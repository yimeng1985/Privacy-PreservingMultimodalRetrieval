"""Run the privacy protection pipeline (new 6-stage design).

Pipeline stages:
  1. Shadow reconstructor (pre-trained, loaded from checkpoint)
  2. Block-level occlusion sensitivity analysis  →  p_j
  3. Candidate region screening  →  top-M patches
  4. VLM semantic privacy judgment  →  q_j  (placeholder / mock)
  5. Score fusion  →  s_j
  6. Local adaptive protection  →  x'

Usage:
    python scripts/run_defense.py --data_dir <image_folder> \
        --reconstructor_ckpt checkpoints/reconstructor_best.pth \
        [--output_dir results/defense] [--num_images 100]
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Model.models.encoder import CLIPEncoder
from Model.models.reconstructor import Reconstructor, ImprovedReconstructor
from Model.models.selector import OcclusionAnalyzer
from Model.models.vlm import MockVLMJudge, QwenVLMJudge
from Model.models.fusion import ScoreFusion
from Model.models.protector import AdaptiveProtector
from Model.models.perturber import MaskedPGD
from Model.data.dataset import ImageDataset
from Model.eval.metrics import compute_all_metrics


def load_reconstructor(ckpt_path, config, device):
    """Load reconstructor from checkpoint, auto-detecting model type."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get('model_type', 'basic')
    print(f"Loading {model_type} reconstructor from {ckpt_path}")

    if model_type == 'improved':
        rec_cfg = config['reconstructor']
        model = ImprovedReconstructor(
            embed_dim=rec_cfg['embed_dim'],
            base_channels=rec_cfg['base_channels'],
            num_res_blocks=rec_cfg.get('num_res_blocks', 2),
            dropout=0.0,
            attention_resolutions=tuple(
                rec_cfg.get('attention_resolutions', [7, 14])
            ),
        )
    else:
        model = Reconstructor(
            embed_dim=config['reconstructor']['embed_dim'],
            base_channels=config['reconstructor']['base_channels'],
        )

    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print("  Using EMA weights")
    else:
        model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_defense(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ======== Load Models ========
    print("Loading CLIP encoder...")
    encoder = CLIPEncoder(
        model_id=config['encoder']['model_id'],
        device=device,
    )

    reconstructor = load_reconstructor(
        args.reconstructor_ckpt, config, device)

    # ======== Initialize Pipeline Modules ========
    sel_cfg = config.get('analyzer', config.get('selector', {}))
    analyzer = OcclusionAnalyzer(
        patch_size=sel_cfg.get('patch_size', 16),
        occlusion_mode=sel_cfg.get('occlusion_mode', 'mean'),
        loss_fn=sel_cfg.get('loss_fn', 'l1'),
        batch_size=sel_cfg.get('batch_size', 64),
        center_prior_weight=sel_cfg.get('center_prior_weight', 0.3),
        center_prior_sigma=sel_cfg.get('center_prior_sigma', 0.4),
    )

    vlm_cfg = config.get('vlm', {})
    vlm_backend = args.vlm_backend or vlm_cfg.get('backend', 'mock')
    if vlm_backend == 'qwen':
        print("Using QwenVLMJudge (DashScope API)")
        vlm_judge = QwenVLMJudge(
            patch_size=sel_cfg.get('patch_size', 16),
            context_margin=vlm_cfg.get('context_margin', 1),
            api_key=vlm_cfg.get('api_key') or os.environ.get('DASHSCOPE_API_KEY'),
            model=vlm_cfg.get('model', 'qwen3.5-plus'),
            max_retries=vlm_cfg.get('max_retries', 3),
        )
    else:
        print("Using MockVLMJudge (testing mode)")
        vlm_judge = MockVLMJudge(
            patch_size=sel_cfg.get('patch_size', 16),
            default_score=vlm_cfg.get('default_score', 0.5),
            default_action=vlm_cfg.get('default_action', 'noise'),
        )

    fusion_cfg = config.get('fusion', {})
    fusion = ScoreFusion(
        alpha=fusion_cfg.get('alpha', 1.0),
        beta=fusion_cfg.get('beta', 1.0),
        mode=fusion_cfg.get('mode', 'multiplicative'),
    )

    prot_cfg = config.get('protector', {})
    protector = AdaptiveProtector(
        patch_size=sel_cfg.get('patch_size', 16),
        epsilon_min=prot_cfg.get('epsilon_min', 0.02),
        epsilon_scale=prot_cfg.get('epsilon_scale', 0.15),
        blur_kernel_size=prot_cfg.get('blur_kernel_size', 7),
        mosaic_block=prot_cfg.get('mosaic_block', 4),
    )

    pgd_cfg = config.get('pgd', {})
    use_pgd = pgd_cfg.get('enabled', False)
    if use_pgd:
        pgd = MaskedPGD(
            epsilon=pgd_cfg.get('epsilon', 0.06),
            alpha=pgd_cfg.get('alpha', 0.008),
            num_steps=pgd_cfg.get('num_steps', 10),
            lambda_util=pgd_cfg.get('lambda_util', 1.0),
            lambda_smooth=pgd_cfg.get('lambda_smooth', 0.01),
        )
        print(f"Using MaskedPGD (eps={pgd.epsilon}, steps={pgd.num_steps})")
    else:
        pgd = None
        print("Using simple AdaptiveProtector (noise/blur)")

    # ======== Dataset ========
    dataset = ImageDataset(args.data_dir, image_size=224)
    num_images = min(len(dataset), args.num_images)
    print(f"Processing {num_images} images from {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    top_m_ratio = sel_cfg.get('top_m_ratio', 0.3)
    top_k_ratio = sel_cfg.get('top_k_ratio', 0.15)
    total_patches = (224 // sel_cfg.get('patch_size', 16)) ** 2
    all_metrics = []

    for idx in range(num_images):
        t0 = time.time()
        image = dataset[idx].unsqueeze(0).to(device)

        # ---- Phase 1: baseline encoding ----
        z_orig = encoder.encode(image)

        # ---- Phase 2: occlusion sensitivity analysis ----
        p_scores = analyzer.compute_sensitivity(image, encoder, reconstructor)

        # ---- Phase 3: candidate screening ----
        candidates = analyzer.select_candidates(p_scores, top_m_ratio)

        # ---- Phase 4: VLM semantic privacy judgment ----
        candidates = vlm_judge.judge_patches(image, candidates)

        # ---- Phase 5: score fusion ----
        candidates = fusion.fuse(candidates)
        selected = fusion.select_final(
            candidates, top_k_ratio=top_k_ratio,
            total_patches=total_patches)

        # ---- Phase 6: local adaptive protection ----
        if use_pgd:
            mask = protector.build_mask(selected, image.shape).to(device)
            image_protected = pgd.perturb(
                image, mask, z_orig, encoder, reconstructor)
        else:
            image_protected = protector.protect_image(
                image, selected,
                default_action=vlm_cfg.get('default_action', 'noise'))

        # ---- Encode protected image ----
        z_protected = encoder.encode(image_protected)

        # ---- Evaluate ----
        with torch.no_grad():
            recon_orig = reconstructor(z_orig.float())
            recon_protected = reconstructor(z_protected.float())

        metrics = compute_all_metrics(
            image, recon_orig, recon_protected, z_orig, z_protected)
        metrics['image_idx'] = idx
        metrics['time_sec'] = time.time() - t0
        metrics['num_candidates'] = len(candidates)
        metrics['num_protected'] = len(selected)
        all_metrics.append(metrics)

        # ---- Visualization ----
        if idx < args.num_vis:
            mask = protector.build_mask(selected, image.shape).to(device)
            mask_vis = mask.expand_as(image)
            diff_vis = ((image_protected - image).abs() * 10).clamp(0, 1)
            grid = torch.cat([
                image, image_protected, mask_vis,
                diff_vis, recon_orig, recon_protected,
            ], dim=0)
            vutils.save_image(
                grid,
                os.path.join(vis_dir, f'{idx:04d}.png'),
                nrow=6, padding=2,
            )

        print(f"[{idx+1}/{num_images}] "
              f"patches={len(selected)}/{total_patches}  "
              f"cos_drift={metrics['cos_drift']:.4f}  "
              f"psnr_prot={metrics['psnr_protected']:.2f}  "
              f"lpips_prot={metrics['lpips_protected']:.4f}  "
              f"time={metrics['time_sec']:.2f}s")

    # ======== Summary ========
    print("\n" + "=" * 60)
    print("AVERAGE METRICS")
    print("=" * 60)

    summary = {}
    metric_keys = [k for k in all_metrics[0]
                   if k not in ('image_idx', 'time_sec',
                                'num_candidates', 'num_protected')]
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        summary[key] = sum(vals) / len(vals)
        print(f"  {key:20s}: {summary[key]:.4f}")

    summary['total_time'] = sum(m['time_sec'] for m in all_metrics)
    summary['avg_time'] = summary['total_time'] / num_images
    summary['avg_protected_patches'] = sum(
        m['num_protected'] for m in all_metrics) / num_images
    print(f"  {'avg_time':20s}: {summary['avg_time']:.2f}s")
    print(f"  {'avg_protected':20s}: {summary['avg_protected_patches']:.1f} patches")

    results_path = os.path.join(args.output_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'per_image': all_metrics}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run privacy protection pipeline')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--reconstructor_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/defense')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--num_vis', type=int, default=20,
                        help='Number of images to save visualizations for')
    parser.add_argument('--vlm_backend', type=str, default=None,
                        choices=['mock', 'qwen'],
                        help='VLM backend: mock (default) or qwen (DashScope)')
    args = parser.parse_args()
    run_defense(args)

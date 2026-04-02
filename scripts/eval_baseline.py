"""Evaluate baselines for comparison with the new privacy protection pipeline.

Baselines:
  1. no_defense:    original embedding (attack upper bound)
  2. gaussian_img:  add Gaussian noise to full image before encoding
  3. gaussian_emb:  add Gaussian noise to embedding directly
  4. random_patch:  randomly select patches + default protection
  5. recon_only:    our method with only p_j (no VLM, uniform q)
  6. full_pipeline: our full method (mock VLM for automated testing)

Usage:
    python scripts/eval_baseline.py --data_dir <image_folder> \
        --reconstructor_ckpt checkpoints/reconstructor_best.pth
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
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
    """Load reconstructor from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = ckpt.get('model_type', 'basic')

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
    else:
        model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def evaluate_baselines(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Models
    encoder = CLIPEncoder(
        model_id=config['encoder']['model_id'],
        device=device,
    )
    reconstructor = load_reconstructor(
        args.reconstructor_ckpt, config, device)

    # Pipeline components
    sel_cfg = config.get('analyzer', config.get('selector', {}))
    patch_size = sel_cfg.get('patch_size', 16)
    top_m_ratio = sel_cfg.get('top_m_ratio', 0.3)
    top_k_ratio = sel_cfg.get('top_k_ratio', 0.15)

    analyzer = OcclusionAnalyzer(
        patch_size=patch_size,
        occlusion_mode=sel_cfg.get('occlusion_mode', 'mean'),
        batch_size=sel_cfg.get('batch_size', 64),
        center_prior_weight=sel_cfg.get('center_prior_weight', 0.3),
        center_prior_sigma=sel_cfg.get('center_prior_sigma', 0.4),
    )
    vlm_cfg = config.get('vlm', {})
    vlm_backend = args.vlm_backend or vlm_cfg.get('backend', 'mock')
    if vlm_backend == 'qwen':
        vlm_judge = QwenVLMJudge(
            patch_size=patch_size,
            api_key=vlm_cfg.get('api_key') or os.environ.get('DASHSCOPE_API_KEY'),
            model=vlm_cfg.get('model', 'qwen3.5-plus'),
        )
    else:
        vlm_judge = MockVLMJudge(patch_size=patch_size, default_score=0.5)
    fusion = ScoreFusion(alpha=1.0, beta=1.0, mode='multiplicative')

    prot_cfg = config.get('protector', {})
    protector = AdaptiveProtector(
        patch_size=patch_size,
        epsilon_min=prot_cfg.get('epsilon_min', 0.02),
        epsilon_scale=prot_cfg.get('epsilon_scale', 0.15),
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
    else:
        pgd = None

    dataset = ImageDataset(args.data_dir, image_size=224)
    num_images = min(len(dataset), args.num_images)
    noise_sigma = args.noise_sigma
    nh = nw = 224 // patch_size
    total_patches = nh * nw

    results = {name: [] for name in [
        'no_defense', 'gaussian_img', 'gaussian_emb',
        'random_patch', 'recon_only', 'full_pipeline',
    ]}

    for idx in range(num_images):
        image = dataset[idx].unsqueeze(0).to(device)
        z_orig = encoder.encode(image)

        with torch.no_grad():
            recon_orig = reconstructor(z_orig.float())

        # --- 1. No defense ---
        m = compute_all_metrics(image, recon_orig, recon_orig,
                                z_orig, z_orig)
        results['no_defense'].append(m)

        # --- 2. Gaussian noise on image ---
        noise = torch.randn_like(image) * noise_sigma
        noisy_img = torch.clamp(image + noise, 0, 1)
        z_noisy = encoder.encode(noisy_img)
        with torch.no_grad():
            recon_noisy = reconstructor(z_noisy.float())
        m = compute_all_metrics(image, recon_orig, recon_noisy,
                                z_orig, z_noisy)
        results['gaussian_img'].append(m)

        # --- 3. Gaussian noise on embedding ---
        z_emb_noisy = z_orig + torch.randn_like(z_orig) * noise_sigma
        z_emb_noisy = F.normalize(z_emb_noisy, dim=-1)
        with torch.no_grad():
            recon_emb_noisy = reconstructor(z_emb_noisy.float())
        m = compute_all_metrics(image, recon_orig, recon_emb_noisy,
                                z_orig, z_emb_noisy)
        results['gaussian_emb'].append(m)

        # --- 4. Random patch selection + protection ---
        k = max(1, int(total_patches * top_k_ratio))
        rand_idx = torch.randperm(total_patches)[:k]
        rand_patches = []
        for ri in rand_idx:
            row = ri.item() // nw
            col = ri.item() % nw
            rand_patches.append({
                'row': row, 'col': col,
                'y0': row * patch_size, 'y1': (row + 1) * patch_size,
                'x0': col * patch_size, 'x1': (col + 1) * patch_size,
                's_score': 0.5, 'action': 'noise',
            })
        img_rand = protector.protect_image(image, rand_patches)
        if use_pgd:
            rand_mask = protector.build_mask(rand_patches, image.shape).to(device)
            img_rand = pgd.perturb(image, rand_mask, z_orig, encoder, reconstructor)
        z_rand = encoder.encode(img_rand)
        with torch.no_grad():
            recon_rand = reconstructor(z_rand.float())
        m = compute_all_metrics(image, recon_orig, recon_rand,
                                z_orig, z_rand)
        results['random_patch'].append(m)

        # --- 5. Reconstruction-only selection (no VLM) ---
        p_scores = analyzer.compute_sensitivity(image, encoder, reconstructor)
        cands = analyzer.select_candidates(p_scores, top_m_ratio)
        # Assign uniform q_score = 1.0 (skip VLM)
        for c in cands:
            c['q_score'] = 1.0
            c['action'] = 'noise'
        cands = fusion.fuse(cands)
        sel = fusion.select_final(cands, top_k_ratio, total_patches)
        if use_pgd:
            ro_mask = protector.build_mask(sel, image.shape).to(device)
            img_recon_only = pgd.perturb(image, ro_mask, z_orig, encoder, reconstructor)
        else:
            img_recon_only = protector.protect_image(image, sel)
        z_recon_only = encoder.encode(img_recon_only)
        with torch.no_grad():
            recon_ro = reconstructor(z_recon_only.float())
        m = compute_all_metrics(image, recon_orig, recon_ro,
                                z_orig, z_recon_only)
        results['recon_only'].append(m)

        # --- 6. Full pipeline (mock VLM) ---
        cands2 = analyzer.select_candidates(p_scores, top_m_ratio)
        cands2 = vlm_judge.judge_patches(image, cands2)
        cands2 = fusion.fuse(cands2)
        sel2 = fusion.select_final(cands2, top_k_ratio, total_patches)
        if use_pgd:
            full_mask = protector.build_mask(sel2, image.shape).to(device)
            img_full = pgd.perturb(image, full_mask, z_orig, encoder, reconstructor)
        else:
            img_full = protector.protect_image(image, sel2)
        z_full = encoder.encode(img_full)
        with torch.no_grad():
            recon_full = reconstructor(z_full.float())
        m = compute_all_metrics(image, recon_orig, recon_full,
                                z_orig, z_full)
        results['full_pipeline'].append(m)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{num_images}")

    # ---- Aggregate and print ----
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    summary = {}
    metric_keys = [k for k in results['no_defense'][0]
                   if k not in ('image_idx', 'time_sec')]

    header = f"{'Method':20s}" + "".join(f"{k:>16s}" for k in metric_keys)
    print(header)
    print("-" * len(header))

    for method, metrics_list in results.items():
        row = f"{method:20s}"
        method_summary = {}
        for key in metric_keys:
            avg = sum(m[key] for m in metrics_list) / len(metrics_list)
            method_summary[key] = avg
            row += f"{avg:16.4f}"
        summary[method] = method_summary
        print(row)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'baseline_comparison.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/baseline_comparison.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate baselines')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--reconstructor_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/baselines')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--noise_sigma', type=float, default=0.05,
                        help='Noise std for Gaussian baselines')
    parser.add_argument('--vlm_backend', type=str, default=None,
                        choices=['mock', 'qwen'],
                        help='VLM backend: mock (default) or qwen (DashScope)')
    args = parser.parse_args()
    evaluate_baselines(args)

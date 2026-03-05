"""Run SPAG defense pipeline and evaluate.

Usage:
    python scripts/run_defense.py --data_dir <image_folder> \
        --reconstructor_ckpt checkpoints/reconstructor_epoch50.pth \
        [--output_dir results/defense] [--num_images 100]
"""
import argparse
import json
import os
import sys
import time

import torch
import torchvision.utils as vutils
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spag.models.encoder import CLIPEncoder
from spag.models.reconstructor import Reconstructor
from spag.models.selector import OcclusionSelector
from spag.models.perturber import MaskedPGD
from spag.data.dataset import ImageDataset
from spag.eval.metrics import compute_all_metrics


def run_defense(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- Load models ----
    print("Loading CLIP encoder...")
    encoder = CLIPEncoder(
        model_id=config['encoder']['model_id'],
        device=device,
    )

    print("Loading reconstructor...")
    reconstructor = Reconstructor(
        embed_dim=config['reconstructor']['embed_dim'],
        base_channels=config['reconstructor']['base_channels'],
    ).to(device)
    ckpt = torch.load(args.reconstructor_ckpt, map_location=device, weights_only=False)
    reconstructor.load_state_dict(ckpt['model_state_dict'])
    reconstructor.eval()
    for p in reconstructor.parameters():
        p.requires_grad = False

    # ---- Defense modules ----
    selector = OcclusionSelector(
        patch_size=config['selector']['patch_size'],
        top_k_ratio=config['selector']['top_k_ratio'],
        mode=config['selector']['occlusion_mode'],
        eps=config['selector']['eps'],
    )
    perturber = MaskedPGD(
        epsilon=config['perturber']['epsilon'],
        alpha=config['perturber']['alpha'],
        num_steps=config['perturber']['num_steps'],
        lambda_util=config['perturber']['lambda_util'],
        lambda_smooth=config['perturber']['lambda_smooth'],
    )

    # ---- Dataset ----
    dataset = ImageDataset(args.data_dir, image_size=224)
    num_images = min(len(dataset), args.num_images)
    print(f"Processing {num_images} images from {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    all_metrics = []

    for idx in range(num_images):
        t0 = time.time()
        image = dataset[idx].unsqueeze(0).to(device)

        # 1. Original embedding
        z_orig = encoder.encode(image)

        # 2. Score patches
        scores, u_scores, p_scores = selector.score_patches(
            image, encoder, reconstructor)

        # 3. Select top-k patches → mask
        mask = selector.select_top_k(scores, image.shape)

        # 4. Masked PGD perturbation
        image_perturbed = perturber.perturb(
            image, mask, z_orig, encoder, reconstructor)

        # 5. Protected embedding
        z_protected = encoder.encode(image_perturbed)

        # 6. Evaluate
        with torch.no_grad():
            recon_orig = reconstructor(z_orig.float())
            recon_protected = reconstructor(z_protected.float())

        metrics = compute_all_metrics(
            image, recon_orig, recon_protected, z_orig, z_protected)
        metrics['image_idx'] = idx
        metrics['time_sec'] = time.time() - t0
        all_metrics.append(metrics)

        # 7. Save visualizations (first N images)
        if idx < args.num_vis:
            # Row: original | perturbed | mask | recon_orig | recon_protected
            mask_vis = mask.expand_as(image)
            perturbation_vis = ((image_perturbed - image).abs() * 10).clamp(0, 1)
            grid = torch.cat([
                image, image_perturbed, mask_vis,
                perturbation_vis, recon_orig, recon_protected
            ], dim=0)
            vutils.save_image(
                grid,
                os.path.join(vis_dir, f'{idx:04d}.png'),
                nrow=6, padding=2,
            )

        print(f"[{idx+1}/{num_images}] "
              f"cos_drift={metrics['cos_drift']:.4f}  "
              f"psnr_orig={metrics['psnr_orig']:.2f}  "
              f"psnr_prot={metrics['psnr_protected']:.2f}  "
              f"lpips_orig={metrics['lpips_orig']:.4f}  "
              f"lpips_prot={metrics['lpips_protected']:.4f}  "
              f"time={metrics['time_sec']:.2f}s")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("AVERAGE METRICS")
    print("=" * 60)

    summary = {}
    metric_keys = [k for k in all_metrics[0] if k not in ('image_idx', 'time_sec')]
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        summary[key] = sum(vals) / len(vals)
        print(f"  {key:20s}: {summary[key]:.4f}")

    summary['total_time'] = sum(m['time_sec'] for m in all_metrics)
    summary['avg_time'] = summary['total_time'] / num_images
    print(f"  {'avg_time':20s}: {summary['avg_time']:.2f}s")

    # Save results
    results_path = os.path.join(args.output_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'per_image': all_metrics}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SPAG defense')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--reconstructor_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/defense')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--num_images', type=int, default=100)
    parser.add_argument('--num_vis', type=int, default=20,
                        help='Number of images to save visualizations for')
    args = parser.parse_args()
    run_defense(args)

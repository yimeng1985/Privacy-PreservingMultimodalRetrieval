"""Evaluate baselines for comparison with SPAG defense.

Baselines:
  1. no_defense:  original embedding (attack upper bound)
  2. gaussian_img:  add Gaussian noise to image before encoding
  3. gaussian_emb:  add Gaussian noise to embedding directly
  4. random_mask:   randomly mask patches (no scoring)
  5. spag:          our method (occlusion scoring + masked PGD)

Usage:
    python scripts/eval_baseline.py --data_dir <image_folder> \
        --reconstructor_ckpt checkpoints/reconstructor_epoch50.pth
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spag.models.encoder import CLIPEncoder
from spag.models.reconstructor import Reconstructor
from spag.models.selector import OcclusionSelector
from spag.models.perturber import MaskedPGD
from spag.data.dataset import ImageDataset
from spag.eval.metrics import compute_all_metrics


def make_random_mask(image_shape, patch_size, ratio, device):
    """Generate a random binary patch mask."""
    _, _, H, W = image_shape
    nh, nw = H // patch_size, W // patch_size
    num_patches = nh * nw
    k = max(1, int(num_patches * ratio))

    indices = torch.randperm(num_patches, device=device)[:k]
    mask = torch.zeros(1, 1, H, W, device=device)
    for idx in indices:
        i, j = idx // nw, idx % nw
        mask[:, :, i*patch_size:(i+1)*patch_size,
             j*patch_size:(j+1)*patch_size] = 1.0
    return mask


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
    reconstructor = Reconstructor(
        embed_dim=config['reconstructor']['embed_dim'],
        base_channels=config['reconstructor']['base_channels'],
    ).to(device)
    ckpt = torch.load(args.reconstructor_ckpt, map_location=device, weights_only=False)
    reconstructor.load_state_dict(ckpt['model_state_dict'])
    reconstructor.eval()
    for p in reconstructor.parameters():
        p.requires_grad = False

    # Defense components (for SPAG baseline)
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

    dataset = ImageDataset(args.data_dir, image_size=224)
    num_images = min(len(dataset), args.num_images)

    noise_sigma = args.noise_sigma
    patch_size = config['selector']['patch_size']
    top_k_ratio = config['selector']['top_k_ratio']

    results = {name: [] for name in [
        'no_defense', 'gaussian_img', 'gaussian_emb',
        'random_mask', 'spag'
    ]}

    for idx in range(num_images):
        image = dataset[idx].unsqueeze(0).to(device)
        z_orig = encoder.encode(image)

        with torch.no_grad():
            recon_orig = reconstructor(z_orig.float())

        # --- Baseline 1: No defense ---
        m = compute_all_metrics(image, recon_orig, recon_orig,
                                z_orig, z_orig)
        results['no_defense'].append(m)

        # --- Baseline 2: Gaussian noise on image ---
        noise = torch.randn_like(image) * noise_sigma
        noisy_img = torch.clamp(image + noise, 0, 1)
        z_noisy = encoder.encode(noisy_img)
        with torch.no_grad():
            recon_noisy = reconstructor(z_noisy.float())
        m = compute_all_metrics(image, recon_orig, recon_noisy,
                                z_orig, z_noisy)
        results['gaussian_img'].append(m)

        # --- Baseline 3: Gaussian noise on embedding ---
        z_emb_noisy = z_orig + torch.randn_like(z_orig) * noise_sigma
        z_emb_noisy = F.normalize(z_emb_noisy, dim=-1)
        with torch.no_grad():
            recon_emb_noisy = reconstructor(z_emb_noisy.float())
        m = compute_all_metrics(image, recon_orig, recon_emb_noisy,
                                z_orig, z_emb_noisy)
        results['gaussian_emb'].append(m)

        # --- Baseline 4: Random patch masking + PGD ---
        rand_mask = make_random_mask(image.shape, patch_size,
                                     top_k_ratio, device)
        img_rand = perturber.perturb(image, rand_mask, z_orig,
                                     encoder, reconstructor)
        z_rand = encoder.encode(img_rand)
        with torch.no_grad():
            recon_rand = reconstructor(z_rand.float())
        m = compute_all_metrics(image, recon_orig, recon_rand,
                                z_orig, z_rand)
        results['random_mask'].append(m)

        # --- Baseline 5: SPAG (our method) ---
        scores, _, _ = selector.score_patches(image, encoder, reconstructor)
        mask = selector.select_top_k(scores, image.shape)
        img_spag = perturber.perturb(image, mask, z_orig,
                                     encoder, reconstructor)
        z_spag = encoder.encode(img_spag)
        with torch.no_grad():
            recon_spag = reconstructor(z_spag.float())
        m = compute_all_metrics(image, recon_orig, recon_spag,
                                z_orig, z_spag)
        results['spag'].append(m)

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
    args = parser.parse_args()
    evaluate_baselines(args)

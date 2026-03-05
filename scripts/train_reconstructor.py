"""Train shadow reconstructor: CLIP embedding → image.

Usage:
    python scripts/train_reconstructor.py --data_dir <image_folder> \
        [--output_dir checkpoints] [--config configs/default.yaml]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spag.models.encoder import CLIPEncoder
from spag.models.reconstructor import Reconstructor
from spag.data.dataset import ImageDataset


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- Models ----
    print("Loading CLIP encoder...")
    encoder = CLIPEncoder(
        model_id=config['encoder']['model_id'],
        device=device,
    )

    reconstructor = Reconstructor(
        embed_dim=config['reconstructor']['embed_dim'],
        base_channels=config['reconstructor']['base_channels'],
    ).to(device)
    print(f"Reconstructor params: "
          f"{sum(p.numel() for p in reconstructor.parameters()):,}")

    # ---- Data ----
    dataset = ImageDataset(args.data_dir, image_size=224)
    print(f"Dataset: {len(dataset)} images from {args.data_dir}")

    num_workers = 0 if os.name == 'nt' else 4
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Optimizer & Loss ----
    optimizer = torch.optim.Adam(
        reconstructor.parameters(),
        lr=config['training']['learning_rate'],
    )
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    lambda_lpips = config['training']['lambda_lpips']

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        reconstructor.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training ----
    num_epochs = config['training']['num_epochs']
    save_every = config['training'].get('save_every', 5)
    log_every = config['training'].get('log_every', 50)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        reconstructor.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, images in enumerate(dataloader):
            images = images.to(device)

            with torch.no_grad():
                embeddings = encoder.encode(images)

            recon = reconstructor(embeddings.float())

            loss_l1 = F.l1_loss(recon, images)
            loss_lp = lpips_fn(recon * 2 - 1, images * 2 - 1).mean()
            loss = loss_l1 + lambda_lpips * loss_lp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % log_every == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}  Step {step+1}/{len(dataloader)}  "
                      f"loss={loss.item():.4f}  L1={loss_l1.item():.4f}  "
                      f"LPIPS={loss_lp.item():.4f}")

        avg_loss = epoch_loss / max(len(dataloader), 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.4f}  "
              f"time={elapsed:.1f}s")

        if (epoch + 1) % save_every == 0 or epoch + 1 == num_epochs:
            path = os.path.join(args.output_dir,
                                f'reconstructor_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': reconstructor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, path)
            print(f"  Saved checkpoint → {path}")

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train shadow reconstructor')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(args)

"""Train shadow reconstructor: CLIP embedding → image.

Supports both basic Reconstructor and ImprovedReconstructor.

Usage:
    python scripts/train_reconstructor.py --data_dir <image_folder> \
        [--model_type improved] [--output_dir checkpoints] \
        [--config configs/default.yaml]
"""
import argparse
import copy
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lpips
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Model.models.encoder import CLIPEncoder
from Model.models.reconstructor import Reconstructor, ImprovedReconstructor
from Model.data.dataset import ImageDataset


class EMA:
    """Exponential Moving Average of model parameters for smoother output."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(),
                                    model.parameters()):
            s_param.data.mul_(self.decay).add_(
                m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def build_model(config, model_type):
    """Build reconstructor model based on type."""
    if model_type == 'basic':
        return Reconstructor(
            embed_dim=config['reconstructor']['embed_dim'],
            base_channels=config['reconstructor']['base_channels'],
        )
    elif model_type == 'improved':
        return ImprovedReconstructor(
            embed_dim=config['reconstructor']['embed_dim'],
            base_channels=config['reconstructor']['base_channels'],
            num_res_blocks=config['reconstructor'].get('num_res_blocks', 2),
            dropout=config['reconstructor'].get('dropout', 0.0),
            attention_resolutions=tuple(
                config['reconstructor'].get('attention_resolutions', [7, 14])
            ),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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

    model_type = args.model_type or config['reconstructor'].get('type', 'improved')
    print(f"Building {model_type} reconstructor...")
    reconstructor = build_model(config, model_type).to(device)
    param_count = sum(p.numel() for p in reconstructor.parameters())
    print(f"Reconstructor params: {param_count:,}")

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

    # ---- Optimizer, Scheduler & Loss ----
    lr = config['training']['learning_rate']
    optimizer = torch.optim.AdamW(
        reconstructor.parameters(),
        lr=lr,
        weight_decay=config['training'].get('weight_decay', 1e-4),
    )
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    lambda_lpips = config['training']['lambda_lpips']
    num_epochs = config['training']['num_epochs']

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # EMA
    use_ema = config['training'].get('use_ema', True)
    ema = EMA(reconstructor, decay=config['training'].get('ema_decay', 0.999)) \
        if use_ema else None

    # Gradient clipping
    grad_clip = config['training'].get('grad_clip', 1.0)

    # ---- Resume ----
    start_epoch = 0
    best_loss = float('inf')
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        reconstructor.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if ema and 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training ----
    save_every = config['training'].get('save_every', 5)
    log_every = config['training'].get('log_every', 50)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        reconstructor.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_lp = 0.0
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
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    reconstructor.parameters(), grad_clip)
            optimizer.step()

            # Update EMA
            if ema:
                ema.update(reconstructor)

            epoch_loss += loss.item()
            epoch_l1 += loss_l1.item()
            epoch_lp += loss_lp.item()

            if (step + 1) % log_every == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}  "
                      f"Step {step+1}/{len(dataloader)}  "
                      f"loss={loss.item():.4f}  L1={loss_l1.item():.4f}  "
                      f"LPIPS={loss_lp.item():.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Step scheduler
        scheduler.step()

        n_steps = max(len(dataloader), 1)
        avg_loss = epoch_loss / n_steps
        avg_l1 = epoch_l1 / n_steps
        avg_lp = epoch_lp / n_steps
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.4f}  "
              f"L1={avg_l1:.4f}  LPIPS={avg_lp:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"time={elapsed:.1f}s")

        # Save checkpoint data
        ckpt_data = {
            'epoch': epoch + 1,
            'model_type': model_type,
            'model_state_dict': reconstructor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'config': config,
        }
        if ema:
            ckpt_data['ema_state_dict'] = ema.state_dict()

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_data['best_loss'] = best_loss
            best_path = os.path.join(args.output_dir, 'reconstructor_best.pth')
            torch.save(ckpt_data, best_path)
            print(f"  ★ New best → {best_path} (loss={best_loss:.4f})")

        if (epoch + 1) % save_every == 0 or epoch + 1 == num_epochs:
            path = os.path.join(args.output_dir,
                                f'reconstructor_epoch{epoch+1}.pth')
            torch.save(ckpt_data, path)
            print(f"  Saved → {path}")

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
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['basic', 'improved'],
                        help='Reconstructor type (default: from config)')
    args = parser.parse_args()
    train(args)

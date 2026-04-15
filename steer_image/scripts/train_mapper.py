"""
train_mapper.py
Trains a linear or MLP mapper to align local embeddings to server embedding space.
Supports multiple loss functions: MSE, Cosine, Huber, MSE+Cosine.

Usage:
    python -m steer_image.scripts.train_mapper --config steer_image/configs/default.yaml
    python -m steer_image.scripts.train_mapper --config steer_image/configs/default.yaml --mapper_type linear
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from steer_image.models.mappers import build_mapper


class AlignmentLoss(nn.Module):
    """Combined loss for embedding space alignment."""

    def __init__(self, loss_type: str = "mse+cosine", cosine_weight: float = 0.5):
        super().__init__()
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "cosine":
            return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target)
        elif self.loss_type == "mse+cosine":
            mse = F.mse_loss(pred, target)
            cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
            return (1.0 - self.cosine_weight) * mse + self.cosine_weight * cos
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def train(cfg: dict, mapper_type: str = None, device: str = "cuda"):
    """Train the mapper."""
    train_cfg = cfg["training"]
    paths = cfg["paths"]

    os.makedirs(paths["checkpoints_dir"], exist_ok=True)

    # Override mapper type if specified via CLI
    if mapper_type is not None:
        cfg["mapper"]["type"] = mapper_type

    # Load alignment embeddings
    emb_dir = paths["embeddings_dir"]
    local_align = np.load(os.path.join(emb_dir, "local_align.npy"))
    server_align = np.load(os.path.join(emb_dir, "server_align.npy"))

    print(f"Alignment data: {local_align.shape[0]} samples")
    print(f"Local dim: {local_align.shape[1]}, Server dim: {server_align.shape[1]}")

    # Build tensors
    local_t = torch.from_numpy(local_align).float()
    server_t = torch.from_numpy(server_align).float()

    # Train/val split (90/10)
    n_total = len(local_t)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val

    dataset = TensorDataset(local_t, server_t)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(train_cfg.get("seed", 42))
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                            shuffle=False, pin_memory=True)

    # Build mapper
    input_dim = local_align.shape[1]
    output_dim = server_align.shape[1]
    mapper = build_mapper(cfg, input_dim, output_dim).to(device)

    mapper_name = cfg["mapper"]["type"]
    print(f"Mapper type: {mapper_name}")
    print(f"Parameters: {sum(p.numel() for p in mapper.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    num_epochs = train_cfg["num_epochs"]
    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    elif train_cfg.get("scheduler") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=num_epochs // 3, gamma=0.1
        )

    # Warmup
    warmup_epochs = train_cfg.get("warmup_epochs", 0)

    # Loss
    criterion = AlignmentLoss(
        loss_type=train_cfg.get("loss", "mse+cosine"),
        cosine_weight=train_cfg.get("cosine_weight", 0.5),
    )

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_cosine_sim": []}

    for epoch in range(1, num_epochs + 1):
        # Warmup learning rate
        if epoch <= warmup_epochs:
            warmup_lr = train_cfg["learning_rate"] * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # Train
        mapper.train()
        train_losses = []
        for local_batch, server_batch in train_loader:
            local_batch = local_batch.to(device)
            server_batch = server_batch.to(device)

            pred = mapper(local_batch)
            loss = criterion(pred, server_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if scheduler is not None and epoch > warmup_epochs:
            scheduler.step()

        avg_train_loss = np.mean(train_losses)

        # Validate
        mapper.eval()
        val_losses = []
        val_cosines = []
        with torch.no_grad():
            for local_batch, server_batch in val_loader:
                local_batch = local_batch.to(device)
                server_batch = server_batch.to(device)

                pred = mapper(local_batch)
                val_loss = criterion(pred, server_batch)
                cos_sim = F.cosine_similarity(pred, server_batch, dim=-1).mean()

                val_losses.append(val_loss.item())
                val_cosines.append(cos_sim.item())

        avg_val_loss = np.mean(val_losses)
        avg_val_cosine = np.mean(val_cosines)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_cosine_sim"].append(avg_val_cosine)

        if epoch % train_cfg.get("log_every", 1) == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Val Cosine: {avg_val_cosine:.4f} | "
                  f"LR: {lr:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(
                paths["checkpoints_dir"], f"{mapper_name}_mapper_best.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": mapper.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_cosine": avg_val_cosine,
                "config": cfg,
            }, ckpt_path)

        # Periodic save
        if train_cfg.get("save_every") and epoch % train_cfg["save_every"] == 0:
            ckpt_path = os.path.join(
                paths["checkpoints_dir"], f"{mapper_name}_mapper_epoch{epoch}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": mapper.state_dict(),
                "val_loss": avg_val_loss,
                "config": cfg,
            }, ckpt_path)

    # Save training history
    history_path = os.path.join(paths["results_dir"], f"{mapper_name}_training_history.json")
    os.makedirs(paths["results_dir"], exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {paths['checkpoints_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Train embedding mapper")
    parser.add_argument("--config", type=str, default="steer_image/configs/default.yaml")
    parser.add_argument("--mapper_type", type=str, default=None,
                        choices=["linear", "mlp"], help="Override mapper type")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, mapper_type=args.mapper_type, device=args.device)


if __name__ == "__main__":
    main()

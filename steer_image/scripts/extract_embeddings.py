"""
extract_embeddings.py
Extracts embeddings from both local and server encoders for:
  - Training set (gallery) images
  - Query (test) images
  - Alignment subset (for mapper training)

Saves .npy files to the configured embeddings directory.

Usage:
    python -m steer_image.scripts.extract_embeddings --config steer_image/configs/default.yaml
"""

import os
import sys
import argparse

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from steer_image.models.encoders import build_encoders
from steer_image.data.dataset import ImageRetrievalDataset


def extract_all(cfg: dict, device: str = "cuda"):
    """Extract embeddings for train (gallery) and query (test) sets."""
    os.makedirs(cfg["paths"]["embeddings_dir"], exist_ok=True)

    print("Loading encoders...")
    local_encoder, server_encoder = build_encoders(cfg, device=device)

    local_transform = local_encoder.model.visual.preprocess if hasattr(
        local_encoder.model.visual, 'preprocess') else None
    server_transform = server_encoder.model.visual.preprocess if hasattr(
        server_encoder.model.visual, 'preprocess') else None

    # Use OpenCLIP preprocess transforms
    _, _, local_preprocess = __import__('open_clip', fromlist=['create_model_and_transforms']).create_model_and_transforms(
        cfg["encoders"]["local"]["name"], pretrained=cfg["encoders"]["local"]["pretrained"]
    )
    _, _, server_preprocess = __import__('open_clip', fromlist=['create_model_and_transforms']).create_model_and_transforms(
        cfg["encoders"]["server"]["name"], pretrained=cfg["encoders"]["server"]["pretrained"]
    )

    ds_cfg = cfg["dataset"]
    batch_size = cfg["training"]["batch_size"]

    for split_name, split in [("train", "train"), ("query", "test")]:
        print(f"\nExtracting {split_name} embeddings...")

        dataset = ImageRetrievalDataset(
            name=ds_cfg["name"],
            root=ds_cfg["data_root"],
            split=split,
            local_transform=local_preprocess,
            server_transform=server_preprocess,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=ds_cfg.get("num_workers", 4),
            pin_memory=True,
        )

        local_embeds = []
        server_embeds = []
        labels = []

        for img_local, img_server, label, idx in tqdm(loader, desc=split_name):
            with torch.no_grad():
                e_l = local_encoder.encode(img_local)
                e_s = server_encoder.encode(img_server)
            local_embeds.append(e_l.cpu().numpy())
            server_embeds.append(e_s.cpu().numpy())
            labels.append(label.numpy())

        local_embeds = np.concatenate(local_embeds, axis=0)
        server_embeds = np.concatenate(server_embeds, axis=0)
        labels = np.concatenate(labels, axis=0)

        emb_dir = cfg["paths"]["embeddings_dir"]
        np.save(os.path.join(emb_dir, f"local_{split_name}.npy"), local_embeds)
        np.save(os.path.join(emb_dir, f"server_{split_name}.npy"), server_embeds)
        np.save(os.path.join(emb_dir, f"labels_{split_name}.npy"), labels)

        print(f"  Saved: local_{split_name}.npy  shape={local_embeds.shape}")
        print(f"  Saved: server_{split_name}.npy shape={server_embeds.shape}")
        print(f"  Saved: labels_{split_name}.npy shape={labels.shape}")

    # Build alignment subset from training embeddings
    print("\nBuilding alignment subset...")
    local_train = np.load(os.path.join(cfg["paths"]["embeddings_dir"], "local_train.npy"))
    server_train = np.load(os.path.join(cfg["paths"]["embeddings_dir"], "server_train.npy"))

    n_total = len(local_train)
    n_align = int(n_total * ds_cfg.get("alignment_ratio", 0.5))
    rng = np.random.RandomState(cfg["training"].get("seed", 42))
    align_indices = rng.choice(n_total, size=n_align, replace=False)

    np.save(os.path.join(cfg["paths"]["embeddings_dir"], "local_align.npy"),
            local_train[align_indices])
    np.save(os.path.join(cfg["paths"]["embeddings_dir"], "server_align.npy"),
            server_train[align_indices])
    np.save(os.path.join(cfg["paths"]["embeddings_dir"], "align_indices.npy"),
            align_indices)

    print(f"  Alignment subset: {n_align} / {n_total} samples")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings")
    parser.add_argument("--config", type=str, default="steer_image/configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    extract_all(cfg, device=args.device)


if __name__ == "__main__":
    main()

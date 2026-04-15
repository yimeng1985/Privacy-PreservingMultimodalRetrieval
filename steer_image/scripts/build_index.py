"""
build_index.py
Builds a FAISS index from server gallery embeddings for efficient retrieval.

Usage:
    python -m steer_image.scripts.build_index --config steer_image/configs/default.yaml
"""

import os
import sys
import argparse

import numpy as np
import faiss
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def build_index(cfg: dict):
    """Build FAISS index from server gallery (train) embeddings."""
    paths = cfg["paths"]
    retrieval_cfg = cfg["retrieval"]

    os.makedirs(paths["index_dir"], exist_ok=True)

    # Load server gallery embeddings
    server_gallery = np.load(
        os.path.join(paths["embeddings_dir"], "server_train.npy")
    ).astype(np.float32)

    print(f"Gallery embeddings shape: {server_gallery.shape}")

    # Normalize if configured
    if retrieval_cfg.get("normalize", True):
        norms = np.linalg.norm(server_gallery, axis=1, keepdims=True)
        server_gallery = server_gallery / np.maximum(norms, 1e-8)

    dim = server_gallery.shape[1]
    index_type = retrieval_cfg.get("index_type", "flatip").lower()

    if index_type == "flatip":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "flatl2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index.add(server_gallery)

    index_path = os.path.join(paths["index_dir"], "gallery.index")
    faiss.write_index(index, index_path)

    print(f"Index built: {index.ntotal} vectors, dim={dim}, type={index_type}")
    print(f"Saved to: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument("--config", type=str, default="steer_image/configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    build_index(cfg)


if __name__ == "__main__":
    main()

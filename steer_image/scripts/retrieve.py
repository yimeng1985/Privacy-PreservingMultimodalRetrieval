"""
retrieve.py
Performs image retrieval using the trained mapper and FAISS index.
Supports single-image retrieval and batch retrieval for evaluation.

Usage:
    python -m steer_image.scripts.retrieve --config steer_image/configs/default.yaml
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import faiss
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from steer_image.models.mappers import build_mapper


def load_mapper(cfg: dict, device: str = "cuda"):
    """Load trained mapper from checkpoint."""
    paths = cfg["paths"]
    mapper_name = cfg["mapper"]["type"]

    # Determine dimensions from saved embeddings
    local_emb = np.load(os.path.join(paths["embeddings_dir"], "local_align.npy"))
    server_emb = np.load(os.path.join(paths["embeddings_dir"], "server_align.npy"))
    input_dim = local_emb.shape[1]
    output_dim = server_emb.shape[1]

    mapper = build_mapper(cfg, input_dim, output_dim)

    ckpt_path = os.path.join(paths["checkpoints_dir"], f"{mapper_name}_mapper_best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    mapper.load_state_dict(checkpoint["model_state_dict"])
    mapper = mapper.to(device)
    mapper.eval()

    print(f"Loaded mapper from {ckpt_path} (epoch {checkpoint['epoch']})")
    return mapper


def retrieve_batch(cfg: dict, device: str = "cuda"):
    """Run batch retrieval on query set."""
    paths = cfg["paths"]
    retrieval_cfg = cfg["retrieval"]

    os.makedirs(paths["results_dir"], exist_ok=True)

    # Load mapper
    mapper = load_mapper(cfg, device=device)

    # Load query embeddings and labels
    local_query = np.load(os.path.join(paths["embeddings_dir"], "local_query.npy"))
    query_labels = np.load(os.path.join(paths["embeddings_dir"], "labels_query.npy"))
    gallery_labels = np.load(os.path.join(paths["embeddings_dir"], "labels_train.npy"))

    print(f"Query set: {len(local_query)} images")
    print(f"Gallery set: {len(gallery_labels)} images")

    # Map local query embeddings to server space
    print("Mapping query embeddings to server space...")
    with torch.no_grad():
        local_t = torch.from_numpy(local_query).float().to(device)
        # Process in batches to avoid OOM
        batch_size = cfg["training"]["batch_size"]
        mapped_queries = []
        for i in range(0, len(local_t), batch_size):
            batch = local_t[i:i + batch_size]
            mapped = mapper(batch)
            mapped_queries.append(mapped.cpu().numpy())
        mapped_queries = np.concatenate(mapped_queries, axis=0).astype(np.float32)

    # Normalize if configured
    if retrieval_cfg.get("normalize", True):
        norms = np.linalg.norm(mapped_queries, axis=1, keepdims=True)
        mapped_queries = mapped_queries / np.maximum(norms, 1e-8)

    # Load FAISS index
    index_path = os.path.join(paths["index_dir"], "gallery.index")
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded: {index.ntotal} vectors")

    # Retrieve
    max_k = max(retrieval_cfg.get("top_k", [1, 5, 10, 20]))
    print(f"Retrieving top-{max_k} for {len(mapped_queries)} queries...")

    distances, indices = index.search(mapped_queries, max_k)

    # Save retrieval results
    results = {
        "distances": distances,
        "indices": indices,
        "query_labels": query_labels,
        "gallery_labels": gallery_labels,
    }

    results_path = os.path.join(paths["results_dir"], "retrieval_results.npz")
    np.savez(results_path, **results)
    print(f"Results saved to: {results_path}")

    return distances, indices, query_labels, gallery_labels


# ============ Baseline: Direct server-space retrieval (upper bound) ============

def retrieve_oracle(cfg: dict):
    """Oracle retrieval: uses actual server query embeddings (upper bound)."""
    paths = cfg["paths"]
    retrieval_cfg = cfg["retrieval"]

    server_query = np.load(
        os.path.join(paths["embeddings_dir"], "server_query.npy")
    ).astype(np.float32)

    if retrieval_cfg.get("normalize", True):
        norms = np.linalg.norm(server_query, axis=1, keepdims=True)
        server_query = server_query / np.maximum(norms, 1e-8)

    index_path = os.path.join(paths["index_dir"], "gallery.index")
    index = faiss.read_index(index_path)

    max_k = max(retrieval_cfg.get("top_k", [1, 5, 10, 20]))
    distances, indices = index.search(server_query, max_k)

    query_labels = np.load(os.path.join(paths["embeddings_dir"], "labels_query.npy"))
    gallery_labels = np.load(os.path.join(paths["embeddings_dir"], "labels_train.npy"))

    return distances, indices, query_labels, gallery_labels


def main():
    parser = argparse.ArgumentParser(description="Image retrieval")
    parser.add_argument("--config", type=str, default="steer_image/configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--oracle", action="store_true",
                        help="Run oracle retrieval (server-space upper bound)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.oracle:
        retrieve_oracle(cfg)
    else:
        retrieve_batch(cfg, device=args.device)


if __name__ == "__main__":
    main()

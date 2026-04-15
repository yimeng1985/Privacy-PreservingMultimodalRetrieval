"""
evaluate.py
Comprehensive evaluation script combining:
  - Retrieval metrics: Recall@k, mAP
  - Alignment metrics: MSE, cosine similarity, top-k neighbor consistency
  - Oracle comparison (upper bound)

Usage:
    python -m steer_image.scripts.evaluate --config steer_image/configs/default.yaml
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
import faiss
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from steer_image.scripts.retrieve import load_mapper, retrieve_batch, retrieve_oracle


# ============ Retrieval Metrics ============

def compute_recall_at_k(indices: np.ndarray, query_labels: np.ndarray,
                        gallery_labels: np.ndarray, k: int) -> float:
    """Compute Recall@k: fraction of queries where at least one
    correct result appears in top-k."""
    n_queries = len(query_labels)
    hits = 0
    for i in range(n_queries):
        top_k_labels = gallery_labels[indices[i, :k]]
        if query_labels[i] in top_k_labels:
            hits += 1
    return hits / n_queries


def compute_precision_at_k(indices: np.ndarray, query_labels: np.ndarray,
                           gallery_labels: np.ndarray, k: int) -> float:
    """Compute Precision@k."""
    n_queries = len(query_labels)
    total_precision = 0.0
    for i in range(n_queries):
        top_k_labels = gallery_labels[indices[i, :k]]
        correct = np.sum(top_k_labels == query_labels[i])
        total_precision += correct / k
    return total_precision / n_queries


def compute_map(indices: np.ndarray, query_labels: np.ndarray,
                gallery_labels: np.ndarray, max_k: int = None) -> float:
    """Compute Mean Average Precision (mAP)."""
    n_queries = len(query_labels)
    if max_k is None:
        max_k = indices.shape[1]

    aps = []
    for i in range(n_queries):
        query_label = query_labels[i]
        retrieved_labels = gallery_labels[indices[i, :max_k]]
        relevant = (retrieved_labels == query_label).astype(float)

        if relevant.sum() == 0:
            aps.append(0.0)
            continue

        cumsum = np.cumsum(relevant)
        precision_at_j = cumsum / np.arange(1, max_k + 1)
        ap = np.sum(precision_at_j * relevant) / relevant.sum()
        aps.append(ap)

    return np.mean(aps)


# ============ Alignment Metrics ============

def compute_alignment_metrics(cfg: dict, device: str = "cuda"):
    """Evaluate how well the mapper aligns local to server embeddings."""
    paths = cfg["paths"]

    mapper = load_mapper(cfg, device=device)

    # Use query set for alignment evaluation (unseen during training)
    local_query = np.load(os.path.join(paths["embeddings_dir"], "local_query.npy"))
    server_query = np.load(os.path.join(paths["embeddings_dir"], "server_query.npy"))

    with torch.no_grad():
        local_t = torch.from_numpy(local_query).float().to(device)
        server_t = torch.from_numpy(server_query).float().to(device)

        # Map in batches
        batch_size = cfg["training"]["batch_size"]
        mapped = []
        for i in range(0, len(local_t), batch_size):
            batch = local_t[i:i + batch_size]
            mapped.append(mapper(batch))
        mapped_t = torch.cat(mapped, dim=0)

        # MSE
        mse = F.mse_loss(mapped_t, server_t).item()

        # Cosine similarity
        cos_sim = F.cosine_similarity(mapped_t, server_t, dim=-1).mean().item()

        # Per-sample cosine similarity stats
        per_sample_cos = F.cosine_similarity(mapped_t, server_t, dim=-1)
        cos_std = per_sample_cos.std().item()
        cos_min = per_sample_cos.min().item()
        cos_max = per_sample_cos.max().item()

    # Top-k neighbor consistency (how many of the true server-space neighbors
    # are preserved after mapping)
    mapped_np = mapped_t.cpu().numpy().astype(np.float32)
    server_np = server_query.astype(np.float32)

    # Normalize
    mapped_np = mapped_np / np.maximum(
        np.linalg.norm(mapped_np, axis=1, keepdims=True), 1e-8)
    server_np = server_np / np.maximum(
        np.linalg.norm(server_np, axis=1, keepdims=True), 1e-8)

    # Build small index for neighbor consistency check
    # Use gallery embeddings
    server_gallery = np.load(
        os.path.join(paths["embeddings_dir"], "server_train.npy")
    ).astype(np.float32)
    server_gallery = server_gallery / np.maximum(
        np.linalg.norm(server_gallery, axis=1, keepdims=True), 1e-8)

    dim = server_gallery.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(server_gallery)

    k_consistency = 10
    # True neighbors using real server query embeddings
    _, true_nn = index.search(server_np, k_consistency)
    # Mapped neighbors
    _, mapped_nn = index.search(mapped_np, k_consistency)

    # Compute overlap
    overlaps = []
    for i in range(len(true_nn)):
        true_set = set(true_nn[i])
        mapped_set = set(mapped_nn[i])
        overlap = len(true_set & mapped_set) / k_consistency
        overlaps.append(overlap)

    neighbor_consistency = np.mean(overlaps)

    return {
        "mse": mse,
        "cosine_similarity_mean": cos_sim,
        "cosine_similarity_std": cos_std,
        "cosine_similarity_min": cos_min,
        "cosine_similarity_max": cos_max,
        "neighbor_consistency_top10": neighbor_consistency,
    }


# ============ Main Evaluation ============

def evaluate(cfg: dict, device: str = "cuda"):
    """Run full evaluation pipeline."""
    paths = cfg["paths"]
    retrieval_cfg = cfg["retrieval"]
    top_k_list = retrieval_cfg.get("top_k", [1, 5, 10, 20])
    max_k = max(top_k_list)

    os.makedirs(paths["results_dir"], exist_ok=True)

    results = {}

    # 1. Retrieval with mapper
    print("=" * 60)
    print("Evaluating mapper-based retrieval...")
    print("=" * 60)

    distances, indices, query_labels, gallery_labels = retrieve_batch(
        cfg, device=device)

    retrieval_metrics = {}
    for k in top_k_list:
        recall = compute_recall_at_k(indices, query_labels, gallery_labels, k)
        precision = compute_precision_at_k(indices, query_labels, gallery_labels, k)
        retrieval_metrics[f"recall@{k}"] = recall
        retrieval_metrics[f"precision@{k}"] = precision
        print(f"  Recall@{k}: {recall:.4f}  |  Precision@{k}: {precision:.4f}")

    mAP = compute_map(indices, query_labels, gallery_labels, max_k=max_k)
    retrieval_metrics["mAP"] = mAP
    print(f"  mAP@{max_k}: {mAP:.4f}")

    results["mapper_retrieval"] = retrieval_metrics

    # 2. Oracle retrieval (upper bound)
    print("\n" + "=" * 60)
    print("Evaluating oracle retrieval (upper bound)...")
    print("=" * 60)

    oracle_dist, oracle_idx, _, _ = retrieve_oracle(cfg)

    oracle_metrics = {}
    for k in top_k_list:
        recall = compute_recall_at_k(oracle_idx, query_labels, gallery_labels, k)
        precision = compute_precision_at_k(oracle_idx, query_labels, gallery_labels, k)
        oracle_metrics[f"recall@{k}"] = recall
        oracle_metrics[f"precision@{k}"] = precision
        print(f"  Recall@{k}: {recall:.4f}  |  Precision@{k}: {precision:.4f}")

    mAP_oracle = compute_map(oracle_idx, query_labels, gallery_labels, max_k=max_k)
    oracle_metrics["mAP"] = mAP_oracle
    print(f"  mAP@{max_k}: {mAP_oracle:.4f}")

    results["oracle_retrieval"] = oracle_metrics

    # 3. Alignment quality
    print("\n" + "=" * 60)
    print("Evaluating alignment quality...")
    print("=" * 60)

    alignment = compute_alignment_metrics(cfg, device=device)
    for key, val in alignment.items():
        print(f"  {key}: {val:.6f}")

    results["alignment"] = alignment

    # 4. Gap analysis
    print("\n" + "=" * 60)
    print("Gap Analysis (mapper vs oracle):")
    print("=" * 60)

    gap = {}
    for k in top_k_list:
        rk = f"recall@{k}"
        g = oracle_metrics[rk] - retrieval_metrics[rk]
        gap[rk] = g
        print(f"  {rk} gap: {g:.4f}")

    map_gap = oracle_metrics["mAP"] - retrieval_metrics["mAP"]
    gap["mAP"] = map_gap
    print(f"  mAP gap: {map_gap:.4f}")

    results["gap"] = gap

    # Save all results
    results_path = os.path.join(paths["results_dir"], "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and alignment")
    parser.add_argument("--config", type=str, default="steer_image/configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, device=args.device)


if __name__ == "__main__":
    main()

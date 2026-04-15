"""
run_pipeline.py
One-click pipeline: extract embeddings → train mapper → build index → evaluate.

Usage:
    python -m steer_image.scripts.run_pipeline --config steer_image/configs/default.yaml
    python -m steer_image.scripts.run_pipeline --config steer_image/configs/default.yaml --skip_extract
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(description="Run full STEER image retrieval pipeline")
    parser.add_argument("--config", type=str, default="steer_image/configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip embedding extraction (use existing)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip mapper training (use existing checkpoint)")
    parser.add_argument("--mapper_type", type=str, default=None,
                        choices=["linear", "mlp"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mapper_type:
        cfg["mapper"]["type"] = args.mapper_type

    # Create output directories
    for key in ["output_dir", "embeddings_dir", "checkpoints_dir",
                "index_dir", "results_dir"]:
        os.makedirs(cfg["paths"][key], exist_ok=True)

    # Step 1: Extract embeddings
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("Step 1: Extracting embeddings")
        print("=" * 60)
        from steer_image.scripts.extract_embeddings import extract_all
        extract_all(cfg, device=args.device)
    else:
        print("Skipping embedding extraction.")

    # Step 2: Train mapper
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("Step 2: Training mapper")
        print("=" * 60)
        from steer_image.scripts.train_mapper import train
        train(cfg, device=args.device)
    else:
        print("Skipping mapper training.")

    # Step 3: Build FAISS index
    print("\n" + "=" * 60)
    print("Step 3: Building FAISS index")
    print("=" * 60)
    from steer_image.scripts.build_index import build_index
    build_index(cfg)

    # Step 4: Evaluate
    print("\n" + "=" * 60)
    print("Step 4: Evaluation")
    print("=" * 60)
    from steer_image.scripts.evaluate import evaluate
    results = evaluate(cfg, device=args.device)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

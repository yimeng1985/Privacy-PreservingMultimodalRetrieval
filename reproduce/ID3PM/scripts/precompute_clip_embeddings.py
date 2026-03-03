"""
Pre-compute CLIP embeddings for all images in a dataset directory.

For each image file (e.g., `image_001.png`), this script saves a corresponding
`.npy` file (e.g., `image_001.npy`) containing the L2-normalized CLIP embedding.

This is the recommended approach for training since it avoids repeated CLIP
inference and allows the CLIP model to be freed from GPU memory.

Usage:
  python scripts/precompute_clip_embeddings.py \
    --data_dir /path/to/images \
    --clip_model_name ViT-L-14 \
    --clip_pretrained datacomp_xl_s13b_b90k \
    --batch_size 64
"""

import argparse
import os

import numpy as np
import torch as th
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import blobfile as bf


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp", "webp"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImagePathDataset(Dataset):
    """Dataset that returns (clip_preprocessed_image, image_path) pairs."""

    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.preprocess(img)
            return img_tensor, path
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Return a dummy tensor on error
            return th.zeros(3, 224, 224), path


def main():
    args = create_argparser().parse_args()

    print(f"Loading CLIP model: {args.clip_model_name} ({args.clip_pretrained})...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained=args.clip_pretrained
    )
    model.eval()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Scanning images in {args.data_dir}...")
    all_files = _list_image_files_recursively(args.data_dir)
    print(f"Found {len(all_files)} images")

    # Filter out images that already have embeddings (if not forcing recompute)
    if not args.force:
        files_to_process = []
        for f in all_files:
            embed_path = os.path.splitext(f)[0] + ".npy"
            if not os.path.exists(embed_path):
                files_to_process.append(f)
        print(f"Skipping {len(all_files) - len(files_to_process)} images with existing embeddings")
        all_files = files_to_process

    if not all_files:
        print("No images to process. Done!")
        return

    print(f"Processing {len(all_files)} images...")

    dataset = ImagePathDataset(all_files, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    total_saved = 0
    for batch_images, batch_paths in tqdm(loader, desc="Computing CLIP embeddings"):
        batch_images = batch_images.to(device)

        with th.no_grad():
            embeddings = model.encode_image(batch_images)
            # L2-normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        embeddings_np = embeddings.cpu().numpy().astype(np.float32)

        for emb, path in zip(embeddings_np, batch_paths):
            embed_path = os.path.splitext(path)[0] + ".npy"
            np.save(embed_path, emb)
            total_saved += 1

    print(f"Saved {total_saved} embeddings. Done!")


def create_argparser():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP embeddings for training images."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14",
                        help="CLIP model architecture")
    parser.add_argument("--clip_pretrained", type=str, default="datacomp_xl_s13b_b90k",
                        help="CLIP pretrained weights")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for CLIP inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--force", action="store_true",
                        help="Force re-compute even if embeddings exist")
    return parser


if __name__ == "__main__":
    main()

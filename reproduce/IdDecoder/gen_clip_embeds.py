"""
Generate CLIP embeddings (per-image .npy) for a folder tree.

Usage examples:
  python gen_clip_embeds.py --img_root celeba_hq --out_root embeds_clip
  python gen_clip_embeds.py --img_root celeba_hq/train --out_root embeds_clip/train --batch_size 16

Defaults match the project settings:
- Model: laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K (768-dim output)
- Output mirrors the image subfolder structure, only the suffix changes to .npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: Path) -> list[Path]:
    """Recursively list image files under root, sorted for stable pairing."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=Path, default=Path("celeba_hq"), help="Root containing train/val or any subfolders of images")
    parser.add_argument("--out_root", type=Path, default=Path("embeds_clip"), help="Where to write .npy embeddings")
    parser.add_argument("--model_id", default="laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", help="open_clip model id")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if target .npy exists")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (slow)")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}, model={args.model_id}")

    model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{args.model_id}")
    model = model.to(device).eval()

    paths = list_images(args.img_root)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.img_root}")
    print(f"[info] found {len(paths)} images under {args.img_root}")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    bs = max(1, args.batch_size)
    autocast_enabled = device == "cuda"

    for i in tqdm(range(0, len(paths), bs), desc="embedding"):
        batch_paths = paths[i : i + bs]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                imgs.append(preprocess(im.convert("RGB")))
        x = torch.stack(imgs, dim=0).to(device)

        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=autocast_enabled):
            emb = model.encode_image(x)
        emb = emb.float().cpu().numpy().astype(np.float32)  # shape [B, 768]

        for path_img, vec in zip(batch_paths, emb):
            rel = path_img.relative_to(args.img_root).with_suffix(".npy")
            out_path = out_root / rel
            if out_path.exists() and not args.overwrite:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, vec)

    print(f"[ok] embeddings written to {out_root} (mirrors {args.img_root})")


if __name__ == "__main__":
    main()

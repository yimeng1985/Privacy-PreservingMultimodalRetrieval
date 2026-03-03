"""
Generate images conditioned on CLIP embeddings using classifier-free guidance.

Supports three embedding sources:
  1. --reference_dir : randomly sample images from a directory, extract CLIP
     embeddings on-the-fly (or load paired .npy files), and generate.
  2. --embedding_dir : randomly sample pre-computed .npy embedding files.
  3. --text_prompt   : use CLIP text encoder (separate prompts with |).

When using --reference_dir, if a .npy file with the same stem exists next to
the image, it is loaded directly instead of running CLIP inference.

Output:
  - logs/results/  contains side-by-side comparison PNGs (reference | generated)
    when reference images are available.
  - logs/samples_NxHxWx3.npz  contains all generated images.

Usage:
  # Randomly pick 16 images from training set and generate:
  python scripts/clip_sample.py --model_path ./logs/best_ema_0.9999.pt \\
    --reference_dir images1024x1024 --num_samples 16 --guidance_scale 2.0

  # From a directory of pre-computed embeddings:
  python scripts/clip_sample.py --model_path ./logs/best_ema_0.9999.pt \\
    --embedding_dir /path/to/embeddings --num_samples 16

  # From text prompt:
  python scripts/clip_sample.py --model_path ./logs/best_ema_0.9999.pt \\
    --text_prompt "a photo of a person smiling" --guidance_scale 2.0
"""

import argparse
import os
import random

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.cfg_util import ClassifierFreeGuidanceModel
from guided_diffusion.script_util import (
    clip_model_and_diffusion_defaults,
    create_clip_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'}


def load_clip_for_encoding(clip_model_name="ViT-L-14", clip_pretrained="datacomp_xl_s13b_b90k"):
    """Load CLIP model for extracting embeddings from images or text."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    model.eval()
    return model, preprocess, tokenizer


def get_embeddings_from_images(image_paths, clip_model, preprocess, device):
    """Extract CLIP image embeddings from a list of image paths."""
    embeddings = []
    for path in image_paths:
        # Check for paired .npy file first
        npy_path = os.path.splitext(path)[0] + ".npy"
        if os.path.exists(npy_path):
            emb = th.from_numpy(np.load(npy_path)).float()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)
        else:
            img = Image.open(path).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)
            with th.no_grad():
                emb = clip_model.encode_image(img_input)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())
    return th.cat(embeddings, dim=0)


def get_embeddings_from_text(text_prompts, clip_model, tokenizer, device):
    """Extract CLIP text embeddings from a list of text prompts."""
    tokens = tokenizer(text_prompts).to(device)
    with th.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu()


def discover_images(directory):
    """Recursively find all image files in a directory."""
    image_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
    return sorted(image_files)


def discover_embeddings(directory):
    """Recursively find all .npy files in a directory."""
    npy_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))
    return sorted(npy_files)


def main():
    args = create_argparser().parse_args()

    # Set random seed for reproducibility if specified
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating CLIP-conditioned model and diffusion...")
    model, diffusion = create_clip_model_and_diffusion(
        **args_to_dict(args, clip_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Wrap model with classifier-free guidance
    cfg_model = ClassifierFreeGuidanceModel(
        model,
        guidance_scale=args.guidance_scale,
        clip_embed_dim=args.clip_embed_dim,
    )

    # ---- Prepare CLIP embeddings ----
    logger.log("preparing CLIP embeddings...")
    ref_image_paths = None  # For side-by-side comparison

    if args.reference_dir:
        # Discover all images, randomly sample num_samples of them
        all_images = discover_images(args.reference_dir)
        if not all_images:
            raise ValueError(f"No images found in {args.reference_dir}")
        n_pick = min(args.num_samples, len(all_images))
        selected_images = random.sample(all_images, n_pick)
        ref_image_paths = selected_images
        logger.log(f"randomly selected {n_pick} images from {len(all_images)} total")

        # Check if any need CLIP inference (no paired .npy)
        need_clip = any(
            not os.path.exists(os.path.splitext(p)[0] + ".npy")
            for p in selected_images
        )
        if need_clip:
            clip_model_enc, preprocess, _ = load_clip_for_encoding(
                args.clip_model_name, args.clip_pretrained
            )
            clip_model_enc.to(dist_util.dev())
        else:
            clip_model_enc, preprocess = None, None

        embeddings = get_embeddings_from_images(
            selected_images, clip_model_enc, preprocess, dist_util.dev()
        )
        if clip_model_enc is not None:
            del clip_model_enc
            th.cuda.empty_cache()

    elif args.embedding_dir:
        # Discover all .npy files, randomly sample
        all_npys = discover_embeddings(args.embedding_dir)
        if not all_npys:
            raise ValueError(f"No .npy files found in {args.embedding_dir}")
        n_pick = min(args.num_samples, len(all_npys))
        selected_npys = random.sample(all_npys, n_pick)
        logger.log(f"randomly selected {n_pick} embeddings from {len(all_npys)} total")

        emb_list = []
        for npy_path in selected_npys:
            emb = th.from_numpy(np.load(npy_path)).float()
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            emb_list.append(emb)
        embeddings = th.cat(emb_list, dim=0)

        # Try to find paired images for comparison
        ref_image_paths = []
        for npy_path in selected_npys:
            stem = os.path.splitext(npy_path)[0]
            found = False
            for ext in IMAGE_EXTENSIONS:
                img_path = stem + ext
                if os.path.exists(img_path):
                    ref_image_paths.append(img_path)
                    found = True
                    break
            if not found:
                ref_image_paths.append(None)
        # If no images found at all, disable comparison
        if all(p is None for p in ref_image_paths):
            ref_image_paths = None

    elif args.text_prompt:
        clip_model_enc, _, tokenizer = load_clip_for_encoding(
            args.clip_model_name, args.clip_pretrained
        )
        clip_model_enc.to(dist_util.dev())
        prompts = [p.strip() for p in args.text_prompt.split("|")]
        embeddings = get_embeddings_from_text(
            prompts, clip_model_enc, tokenizer, dist_util.dev()
        )
        del clip_model_enc
        th.cuda.empty_cache()
    else:
        raise ValueError(
            "Must specify one of --reference_dir, --embedding_dir, or --text_prompt"
        )

    num_embeds = embeddings.shape[0]
    logger.log(f"prepared {num_embeds} embeddings of dim {embeddings.shape[1]}")

    # ---- Sampling loop ----
    logger.log("sampling...")
    all_images = []
    all_emb_indices = []
    sample_idx = 0

    while sample_idx < args.num_samples:
        actual_bs = min(args.batch_size, args.num_samples - sample_idx)
        batch_embeds = []
        batch_indices = []
        for i in range(actual_bs):
            emb_idx = (sample_idx + i) % num_embeds
            batch_embeds.append(embeddings[emb_idx])
            batch_indices.append(emb_idx)
        batch_embeds = th.stack(batch_embeds).to(dist_util.dev())

        model_kwargs = {"clip_embed": batch_embeds}

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            cfg_model,
            (actual_bs, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        gathered = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, sample)
        all_images.extend([s.cpu().numpy() for s in gathered])
        all_emb_indices.extend(batch_indices)

        sample_idx += actual_bs * dist.get_world_size()
        logger.log(f"created {min(sample_idx, args.num_samples)} / {args.num_samples} samples")

    arr = np.concatenate(all_images, axis=0)[: args.num_samples]
    all_emb_indices = all_emb_indices[: args.num_samples]

    if dist.get_rank() == 0:
        # Save raw .npz
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

        # Save individual images (with side-by-side comparison if references exist)
        results_dir = os.path.join(logger.get_dir(), "results")
        os.makedirs(results_dir, exist_ok=True)
        gen_size = args.image_size

        for i in range(arr.shape[0]):
            gen_img = Image.fromarray(arr[i])

            has_ref = (
                ref_image_paths is not None
                and all_emb_indices[i] < len(ref_image_paths)
                and ref_image_paths[all_emb_indices[i]] is not None
            )

            if has_ref:
                ref_path = ref_image_paths[all_emb_indices[i]]
                ref_img = Image.open(ref_path).convert("RGB")
                ref_img = ref_img.resize((gen_size, gen_size), Image.BICUBIC)

                # Side-by-side: [reference | generated]
                canvas = Image.new("RGB", (gen_size * 2, gen_size))
                canvas.paste(ref_img, (0, 0))
                canvas.paste(gen_img, (gen_size, 0))

                ref_name = os.path.splitext(os.path.basename(ref_path))[0]
                save_name = f"{i:05d}_ref_{ref_name}.png"
            else:
                canvas = gen_img
                save_name = f"{i:05d}.png"

            canvas.save(os.path.join(results_dir, save_name))

        logger.log(f"saved {arr.shape[0]} images to {results_dir}")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=4,
        use_ddim=False,
        model_path="",
        guidance_scale=2.0,
        seed=-1,
        # Source of CLIP embeddings (choose one)
        embedding_dir="",
        reference_dir="",
        text_prompt="",
        # CLIP model for on-the-fly encoding
        clip_model_name="ViT-L-14",
        clip_pretrained="datacomp_xl_s13b_b90k",
    )
    defaults.update(clip_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

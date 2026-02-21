"""
Generate images conditioned on CLIP embeddings using classifier-free guidance.

This can generate images from:
  1. A reference image (extract its CLIP embedding, then generate)
  2. Text prompts (use CLIP text encoder to get embedding)
  3. Pre-computed embedding files (.npy)

The classifier-free guidance formula:
    ε̃ = (1 + w) · ε(x_t, t, c) - w · ε(x_t, t, ∅)

Usage:
  # From reference images:
  python scripts/clip_sample.py --model_path /path/to/model.pt \
    --reference_dir /path/to/ref_images --guidance_scale 3.0

  # From text prompt:
  python scripts/clip_sample.py --model_path /path/to/model.pt \
    --text_prompt "a photo of a cat" --guidance_scale 3.0

  # From pre-computed embeddings:
  python scripts/clip_sample.py --model_path /path/to/model.pt \
    --embedding_path /path/to/embedding.npy --guidance_scale 3.0
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.cfg_util import ClassifierFreeGuidanceModel
from guided_diffusion.script_util import (
    clip_model_and_diffusion_defaults,
    create_clip_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


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
    from PIL import Image
    embeddings = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_input = preprocess(img).unsqueeze(0).to(device)
        with th.no_grad():
            emb = clip_model.encode_image(img_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
        embeddings.append(emb.cpu())
    return th.cat(embeddings, dim=0)


def get_embeddings_from_text(text_prompts, clip_model, tokenizer, device):
    """Extract CLIP text embeddings from a list of text prompts."""
    tokens = tokenizer(text_prompts).to(device)
    with th.no_grad():
        emb = clip_model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
    return emb.cpu()


def main():
    args = create_argparser().parse_args()

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
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Wrap model with classifier-free guidance
    cfg_model = ClassifierFreeGuidanceModel(
        model,
        guidance_scale=args.guidance_scale,
        clip_embed_dim=args.clip_embed_dim,
    )

    # Prepare CLIP embeddings
    logger.log("preparing CLIP embeddings...")
    if args.embedding_path:
        # Load pre-computed embeddings
        embeddings = th.from_numpy(np.load(args.embedding_path)).float()
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
    elif args.reference_dir:
        # Extract from reference images
        clip_model, preprocess, _ = load_clip_for_encoding(
            args.clip_model_name, args.clip_pretrained
        )
        clip_model.to(dist_util.dev())
        import blobfile as bf
        image_files = sorted([
            os.path.join(args.reference_dir, f)
            for f in bf.listdir(args.reference_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ])
        if not image_files:
            raise ValueError(f"No images found in {args.reference_dir}")
        embeddings = get_embeddings_from_images(
            image_files, clip_model, preprocess, dist_util.dev()
        )
        del clip_model  # Free CLIP model memory
        th.cuda.empty_cache() if th.cuda.is_available() else None
    elif args.text_prompt:
        # Extract from text prompt
        clip_model, _, tokenizer = load_clip_for_encoding(
            args.clip_model_name, args.clip_pretrained
        )
        clip_model.to(dist_util.dev())
        prompts = [p.strip() for p in args.text_prompt.split("|")]
        embeddings = get_embeddings_from_text(
            prompts, clip_model, tokenizer, dist_util.dev()
        )
        del clip_model
        th.cuda.empty_cache() if th.cuda.is_available() else None
    else:
        raise ValueError(
            "Must specify one of --embedding_path, --reference_dir, or --text_prompt"
        )

    logger.log(f"prepared {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # Sampling loop
    logger.log("sampling...")
    all_images = []
    sample_idx = 0

    while sample_idx < args.num_samples:
        # Select embeddings for this batch (cycle through provided embeddings)
        batch_embeds = []
        for i in range(args.batch_size):
            emb_idx = (sample_idx + i) % embeddings.shape[0]
            batch_embeds.append(embeddings[emb_idx])
        batch_embeds = th.stack(batch_embeds).to(dist_util.dev())

        model_kwargs = {"clip_embed": batch_embeds}

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            cfg_model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([s.cpu().numpy() for s in gathered_samples])

        sample_idx += args.batch_size * dist.get_world_size()
        logger.log(f"created {min(sample_idx, args.num_samples)} / {args.num_samples} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=4,
        use_ddim=False,
        model_path="",
        guidance_scale=3.0,
        # Source of CLIP embeddings (choose one)
        embedding_path="",
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

"""
Train a CLIP-conditioned diffusion model on images.

This implements the core training loop from the paper
"Controllable Inversion of Black-Box Face Recognition Models via Diffusion".

The model is conditioned on CLIP embeddings that are either:
  - Pre-computed and stored as .npy files alongside images (recommended)
  - Computed on-the-fly during training (slower)

Classifier-free guidance is applied by randomly dropping the CLIP condition
with probability p_uncond during training.

Usage:
  python scripts/clip_train.py --data_dir /path/to/images \
    --image_size 64 --clip_embed_dim 768 --p_uncond 0.1 \
    --batch_size 16 --lr 1e-4
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.clip_image_datasets import load_clip_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    clip_model_and_diffusion_defaults,
    create_clip_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating CLIP-conditioned model and diffusion...")
    model, diffusion = create_clip_model_and_diffusion(
        **args_to_dict(args, clip_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_clip_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        clip_embed_dim=args.clip_embed_dim,
        p_uncond=args.p_uncond,
        precomputed_embeddings=args.precomputed_embeddings,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        early_stop_patience=args.early_stop_patience,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        eval_interval=args.eval_interval,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=3e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        early_stop_patience=0,
        keep_last_n_checkpoints=3,
        eval_interval=0,
        precomputed_embeddings=True,
        clip_model_name="ViT-L-14",
        clip_pretrained="datacomp_xl_s13b_b90k",
    )
    defaults.update(clip_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

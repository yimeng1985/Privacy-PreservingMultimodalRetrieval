"""
Dataset utilities for CLIP-conditioned diffusion training.

Supports two modes:
  1. Pre-computed embeddings: each image `foo.png` has a corresponding `foo.npy`
     file containing its CLIP embedding (faster training, recommended).
  2. On-the-fly CLIP extraction: computes CLIP embeddings during data loading
     (simpler setup, slower training).
"""

import math
import os
import random

import numpy as np
import torch as th
from PIL import Image
import blobfile as bf
from torch.utils.data import DataLoader, Dataset


def load_clip_data(
    *,
    data_dir,
    batch_size,
    image_size,
    clip_embed_dim=768,
    p_uncond=0.1,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    precomputed_embeddings=True,
    clip_model_name="ViT-L-14",
    clip_pretrained="datacomp_xl_s13b_b90k",
    num_workers=4,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each image is an NCHW float tensor. The kwargs dict contains the key
    "clip_embed" mapping to a [N x clip_embed_dim] float tensor of CLIP embeddings.

    With classifier-free guidance, embeddings are randomly replaced with zeros
    with probability `p_uncond`.

    :param data_dir: a dataset directory containing images (and optionally .npy embeddings).
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param clip_embed_dim: dimension of the CLIP embedding vector.
    :param p_uncond: probability of dropping the condition (replacing with zeros)
        for classifier-free guidance training.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param precomputed_embeddings: if True, load pre-computed .npy embedding files;
        otherwise compute embeddings on-the-fly using CLIP.
    :param clip_model_name: the CLIP model architecture name (for on-the-fly mode).
    :param clip_pretrained: the CLIP pretrained weights name (for on-the-fly mode).
    :param num_workers: number of dataloader workers.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = _list_image_files_recursively(data_dir)

    dataset = CLIPImageDataset(
        image_size,
        all_files,
        clip_embed_dim=clip_embed_dim,
        p_uncond=p_uncond,
        random_crop=random_crop,
        random_flip=random_flip,
        precomputed_embeddings=precomputed_embeddings,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )

    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )
    while True:
        yield from loader


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


class CLIPImageDataset(Dataset):
    """
    A dataset returning (image, {"clip_embed": ...}) pairs for CLIP-conditioned
    diffusion training.
    """

    def __init__(
        self,
        resolution,
        image_paths,
        clip_embed_dim=768,
        p_uncond=0.1,
        random_crop=False,
        random_flip=True,
        precomputed_embeddings=True,
        clip_model_name="ViT-L-14",
        clip_pretrained="datacomp_xl_s13b_b90k",
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.clip_embed_dim = clip_embed_dim
        self.p_uncond = p_uncond
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.precomputed_embeddings = precomputed_embeddings

        # For on-the-fly CLIP extraction
        self._clip_model = None
        self._clip_preprocess = None
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained

    def _get_clip_model(self):
        """Lazy-load CLIP model for on-the-fly embedding extraction."""
        if self._clip_model is None:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, pretrained=self.clip_pretrained
            )
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
        return self._clip_model, self._clip_preprocess

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        # Load and process image for diffusion
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        # Load or compute CLIP embedding
        if self.precomputed_embeddings:
            # Look for .npy file alongside the image
            embed_path = os.path.splitext(path)[0] + ".npy"
            if os.path.exists(embed_path):
                clip_embed = np.load(embed_path).astype(np.float32)
            else:
                raise FileNotFoundError(
                    f"Pre-computed embedding not found: {embed_path}. "
                    f"Run scripts/precompute_clip_embeddings.py first, or set "
                    f"--precomputed_embeddings=False for on-the-fly extraction."
                )
        else:
            # On-the-fly CLIP extraction
            clip_model, clip_preprocess = self._get_clip_model()
            # Re-open image for CLIP preprocessing
            with bf.BlobFile(path, "rb") as f:
                clip_pil = Image.open(f)
                clip_pil.load()
            clip_pil = clip_pil.convert("RGB")
            clip_input = clip_preprocess(clip_pil).unsqueeze(0)
            with th.no_grad():
                clip_embed = clip_model.encode_image(clip_input).squeeze(0).cpu().numpy()
            # L2-normalize
            clip_embed = clip_embed / (np.linalg.norm(clip_embed) + 1e-8)
            clip_embed = clip_embed.astype(np.float32)

        # Classifier-free guidance: randomly drop conditioning
        if random.random() < self.p_uncond:
            clip_embed = np.zeros(self.clip_embed_dim, dtype=np.float32)

        out_dict = {"clip_embed": clip_embed}
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


class ImageDataset(Dataset):
    """Simple dataset that loads all images from a directory (recursively).

    Returns tensors in [0, 1] with shape (3, image_size, image_size).
    """

    def __init__(self, root_dir, image_size=224):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize(image_size + 32),          # slight upscale
            T.CenterCrop(image_size),            # crop to exact size
            T.ToTensor(),                        # [0, 1]
        ])

        # Collect all image paths recursively
        self.image_paths = sorted([
            str(p) for p in self.root_dir.rglob('*')
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {root_dir}. "
                f"Supported formats: {IMAGE_EXTENSIONS}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

    def get_path(self, idx):
        return self.image_paths[idx]

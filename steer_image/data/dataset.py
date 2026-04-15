"""
Dataset module: image retrieval dataset and alignment dataset construction.
Supports CIFAR-100, CIFAR-10, Food-101, OxfordPets via torchvision.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T


def _get_torchvision_dataset(name: str, root: str, split: str, transform=None):
    """Load a torchvision dataset by name."""
    name = name.lower()
    train = (split == "train")

    if name == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )
    elif name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
    elif name == "food101":
        ds = torchvision.datasets.Food101(
            root=root, split="train" if train else "test",
            download=True, transform=transform
        )
    elif name == "oxfordpets":
        ds = torchvision.datasets.OxfordIIITPet(
            root=root, split="trainval" if train else "test",
            download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return ds


class ImageRetrievalDataset(Dataset):
    """Wraps a torchvision dataset for dual-encoder retrieval.
    Returns (image_local, image_server, label, index).
    """

    def __init__(self, name: str, root: str, split: str,
                 local_transform=None, server_transform=None):
        # Load raw dataset without transform (we apply two transforms)
        self.raw_dataset = _get_torchvision_dataset(name, root, split, transform=None)
        self.local_transform = local_transform
        self.server_transform = server_transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        img, label = self.raw_dataset[idx]

        # Convert to RGB PIL Image if needed
        if not hasattr(img, 'mode'):
            from PIL import Image
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_local = self.local_transform(img) if self.local_transform else img
        img_server = self.server_transform(img) if self.server_transform else img

        return img_local, img_server, label, idx


class AlignmentDataset(Dataset):
    """Alignment dataset built from precomputed embeddings.
    Used to train the mapper: (local_embedding, server_embedding).
    """

    def __init__(self, local_embeddings: np.ndarray, server_embeddings: np.ndarray):
        assert len(local_embeddings) == len(server_embeddings)
        self.local_embeddings = torch.from_numpy(local_embeddings).float()
        self.server_embeddings = torch.from_numpy(server_embeddings).float()

    def __len__(self):
        return len(self.local_embeddings)

    def __getitem__(self, idx):
        return self.local_embeddings[idx], self.server_embeddings[idx]


def build_dataloaders(cfg: dict, local_transform, server_transform):
    """Build train/query dataloaders from config.

    Returns:
        train_dataset: Full training set for embedding extraction
        query_dataset: Query set for retrieval evaluation
        alignment_loader: DataLoader for mapper training (subset of train)
    """
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

    train_dataset = ImageRetrievalDataset(
        name=ds_cfg["name"],
        root=ds_cfg["data_root"],
        split="train",
        local_transform=local_transform,
        server_transform=server_transform,
    )

    query_dataset = ImageRetrievalDataset(
        name=ds_cfg["name"],
        root=ds_cfg["data_root"],
        split="test",
        local_transform=local_transform,
        server_transform=server_transform,
    )

    # Build alignment subset
    n_total = len(train_dataset)
    n_align = int(n_total * ds_cfg.get("alignment_ratio", 0.5))
    rng = np.random.RandomState(train_cfg.get("seed", 42))
    align_indices = rng.choice(n_total, size=n_align, replace=False)
    alignment_subset = Subset(train_dataset, align_indices)

    alignment_loader = DataLoader(
        alignment_subset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, query_loader, alignment_loader

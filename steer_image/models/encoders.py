"""
Encoders module: local (client) and server image encoders.
Uses OpenCLIP to support a variety of CLIP model variants.
"""

import torch
import torch.nn as nn
import open_clip


class LocalEncoder(nn.Module):
    """Client-side image encoder (smaller model)."""

    def __init__(self, model_name: str, pretrained: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of preprocessed images to embeddings."""
        features = self.model.encode_image(images.to(self.device))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class ServerEncoder(nn.Module):
    """Server-side image encoder (larger model)."""

    def __init__(self, model_name: str, pretrained: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of preprocessed images to embeddings."""
        features = self.model.encode_image(images.to(self.device))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


def build_encoders(cfg: dict, device: str = "cuda"):
    """Build local and server encoders from config."""
    local_cfg = cfg["encoders"]["local"]
    server_cfg = cfg["encoders"]["server"]

    local_encoder = LocalEncoder(
        model_name=local_cfg["name"],
        pretrained=local_cfg["pretrained"],
        device=device,
    )
    server_encoder = ServerEncoder(
        model_name=server_cfg["name"],
        pretrained=server_cfg["pretrained"],
        device=device,
    )
    return local_encoder, server_encoder

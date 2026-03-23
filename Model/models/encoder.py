import torch
import torch.nn.functional as F
import open_clip


class CLIPEncoder:
    """Wrapper around CLIP ViT encoder for image embedding extraction."""

    def __init__(self, model_id='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
                 device='cuda'):
        self.device = device
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            f'hf-hub:{model_id}'
        )
        self.model = self.model.float().to(device).eval()
        self.embed_dim = self.model.visual.output_dim

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Extract normalization constants from preprocess transforms
        self._extract_normalization()

    def _extract_normalization(self):
        """Extract mean/std from the model's preprocess pipeline."""
        mean = std = None
        for t in self.preprocess.transforms:
            if isinstance(t, __import__('torchvision').transforms.Normalize):
                mean = t.mean
                std = t.std
                break
        if mean is None:
            # Fallback to standard CLIP normalization
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def normalize(self, images):
        """Normalize images from [0,1] to CLIP expected input range."""
        return (images - self.mean) / self.std

    @torch.no_grad()
    def encode(self, images):
        """Encode images in [0,1] range to L2-normalized embeddings.

        Args:
            images: (B, 3, 224, 224) tensor in [0, 1]
        Returns:
            embeddings: (B, embed_dim) L2-normalized
        """
        normalized = self.normalize(images)
        features = self.model.encode_image(normalized)
        features = F.normalize(features.float(), dim=-1)
        return features

    def encode_with_grad(self, images):
        """Encode with gradient flow for PGD backpropagation.

        Parameters are frozen, but computation graph is preserved
        so gradients can flow back to the input image / perturbation.
        """
        normalized = self.normalize(images)
        features = self.model.encode_image(normalized)
        features = F.normalize(features.float(), dim=-1)
        return features

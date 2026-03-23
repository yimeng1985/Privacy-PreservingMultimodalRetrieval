"""Local adaptive protection module (Phase 6).

Applies image-level protection to selected patches based on VLM-recommended
actions and fused privacy scores. Supports multiple protection strategies
with adaptive strength proportional to patch sensitivity.

Protection methods:
  - noise:    Gaussian noise injection
  - blur:     Gaussian blur
  - mosaic:   Pixelation (downsample → upsample)
  - suppress: Fade towards local/global mean
"""

import torch
import torch.nn.functional as F
import math


class AdaptiveProtector:
    """Apply per-patch adaptive protection to images.

    Protection strength for each patch is determined by:
        epsilon_j = epsilon_min + epsilon_scale * s_j

    where s_j is the fused privacy score.
    """

    def __init__(self, patch_size=16, epsilon_min=0.02, epsilon_scale=0.15,
                 blur_kernel_size=7, mosaic_block=4):
        """
        Args:
            patch_size:       pixel size of each patch
            epsilon_min:      minimum protection strength
            epsilon_scale:    scaling factor for score-based strength
            blur_kernel_size: kernel size for Gaussian blur (odd number)
            mosaic_block:     block size for mosaic/pixelation
        """
        self.patch_size = patch_size
        self.epsilon_min = epsilon_min
        self.epsilon_scale = epsilon_scale
        self.blur_kernel_size = blur_kernel_size
        self.mosaic_block = mosaic_block

    def _get_strength(self, s_score):
        """Compute protection strength from fused score."""
        return self.epsilon_min + self.epsilon_scale * s_score

    def _apply_noise(self, image, y0, y1, x0, x1, strength):
        """Add Gaussian noise to a patch region."""
        region = image[:, :, y0:y1, x0:x1]
        noise = torch.randn_like(region) * strength
        image[:, :, y0:y1, x0:x1] = (region + noise).clamp(0.0, 1.0)

    def _apply_blur(self, image, y0, y1, x0, x1, strength):
        """Apply Gaussian blur to a patch region.

        Strength controls the blur sigma (higher = more blur).
        """
        region = image[:, :, y0:y1, x0:x1]  # (1, 3, ph, pw)
        ks = self.blur_kernel_size
        # Adaptive sigma based on strength
        sigma = max(0.5, strength * 10.0)

        # Create Gaussian kernel
        half = ks // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32, device=image.device)
        kernel_1d = torch.exp(-x ** 2 / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, ks, ks)
        kernel_2d = kernel_2d.expand(3, 1, -1, -1)  # (3, 1, ks, ks)

        # Apply depthwise convolution with padding
        padded = F.pad(region, [half, half, half, half], mode='reflect')
        blurred = F.conv2d(padded, kernel_2d, groups=3)
        image[:, :, y0:y1, x0:x1] = blurred.clamp(0.0, 1.0)

    def _apply_mosaic(self, image, y0, y1, x0, x1, strength):
        """Apply pixelation (mosaic) to a patch region.

        Strength controls how much to downsample (higher = more blocky).
        """
        region = image[:, :, y0:y1, x0:x1]
        h, w = region.shape[2], region.shape[3]
        if h < 2 or w < 2:
            return

        # Adaptive block size: higher strength → larger mosaic blocks
        block = max(2, int(self.mosaic_block * (1.0 + strength * 5.0)))
        block = min(block, min(h, w))

        # Downsample then upsample
        small_h = max(1, h // block)
        small_w = max(1, w // block)
        small = F.interpolate(region, size=(small_h, small_w), mode='area')
        mosaic = F.interpolate(small, size=(h, w), mode='nearest')
        image[:, :, y0:y1, x0:x1] = mosaic

    def _apply_suppress(self, image, y0, y1, x0, x1, strength):
        """Suppress/fade patch towards its local mean.

        Strength controls blend ratio (1.0 = fully replaced by mean).
        """
        region = image[:, :, y0:y1, x0:x1]
        local_mean = region.mean(dim=(2, 3), keepdim=True)
        blend = min(1.0, strength * 3.0)
        blended = region * (1.0 - blend) + local_mean * blend
        image[:, :, y0:y1, x0:x1] = blended.clamp(0.0, 1.0)

    # Action dispatch table
    _ACTION_MAP = {
        'noise': '_apply_noise',
        'blur': '_apply_blur',
        'mosaic': '_apply_mosaic',
        'suppress': '_apply_suppress',
    }

    def protect_image(self, image, selected_patches, default_action='noise'):
        """Apply adaptive protection to selected patches.

        Args:
            image:            (1, 3, H, W) tensor in [0, 1], MODIFIED IN PLACE
            selected_patches: list of dicts from ScoreFusion.select_final,
                              each with 'y0','y1','x0','x1','s_score','action'
            default_action:   fallback action if patch has no 'action' or 'none'

        Returns:
            protected: (1, 3, H, W) protected image tensor in [0, 1]
        """
        protected = image.clone()

        for patch in selected_patches:
            y0, y1 = patch['y0'], patch['y1']
            x0, x1 = patch['x0'], patch['x1']
            s_score = patch.get('s_score', 0.5)
            action = patch.get('action', default_action)

            if action == 'none':
                action = default_action

            strength = self._get_strength(s_score)

            method_name = self._ACTION_MAP.get(action)
            if method_name is None:
                method_name = self._ACTION_MAP[default_action]

            getattr(self, method_name)(protected, y0, y1, x0, x1, strength)

        return protected

    def build_mask(self, selected_patches, image_shape):
        """Build a binary mask from selected patches (for visualization).

        Args:
            selected_patches: list of dicts with 'y0','y1','x0','x1'
            image_shape:      (B, C, H, W)

        Returns:
            mask: (1, 1, H, W) binary tensor
        """
        _, _, H, W = image_shape
        device = 'cpu'
        mask = torch.zeros(1, 1, H, W, device=device)
        for p in selected_patches:
            mask[:, :, p['y0']:p['y1'], p['x0']:p['x1']] = 1.0
        return mask

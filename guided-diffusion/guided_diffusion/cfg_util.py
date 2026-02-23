"""
Classifier-free guidance utilities for CLIP-conditioned diffusion.

Implements the guidance formula from the paper:
    ε̃ = (1 + w) · ε(x_t, t, c) - w · ε(x_t, t, ∅)

where w is the guidance scale, c is the CLIP embedding, and ∅ is the null
(zero) embedding.
"""

import torch as th
import torch.nn as nn


class ClassifierFreeGuidanceModel(nn.Module):
    """
    A wrapper that applies classifier-free guidance at inference time.

    Given a diffusion model conditioned on CLIP embeddings, this wrapper
    runs two forward passes per step:
      1. Conditional:   ε_c = model(x_t, t, clip_embed=c)
      2. Unconditional: ε_u = model(x_t, t, clip_embed=0)

    Then combines them:
      ε̃ = ε_u + guidance_scale * (ε_c - ε_u)
         = (1 + guidance_scale) * ε_c - guidance_scale * ε_u   [when guidance_scale = w]

    Note: guidance_scale=0 gives standard unconditional generation,
          guidance_scale=1 gives standard conditional generation,
          guidance_scale>1 gives enhanced guidance.
    """

    def __init__(self, model, guidance_scale=3.0, clip_embed_dim=768):
        super().__init__()
        self.model = model
        self.guidance_scale = guidance_scale
        self.clip_embed_dim = clip_embed_dim

    def forward(self, x, timesteps, clip_embed=None, **kwargs):
        """
        Apply classifier-free guidance.

        :param x: noisy image [N, C, H, W]
        :param timesteps: [N] timestep tensor
        :param clip_embed: [N, D] CLIP embedding tensor
        :return: guided noise prediction (with optional variance channels)
        """
        assert clip_embed is not None, "clip_embed is required for CFG sampling"

        # Conditional prediction
        cond_output = self.model(x, timesteps, clip_embed=clip_embed, **kwargs)

        if self.guidance_scale == 1.0:
            return cond_output

        # Unconditional prediction (null embedding = zeros)
        null_embed = th.zeros_like(clip_embed)
        uncond_output = self.model(x, timesteps, clip_embed=null_embed, **kwargs)

        # When learn_sigma=True, model outputs [noise, variance] concatenated
        # along channel dim. CFG should only apply to the noise part.
        C = x.shape[1]  # number of image channels (3)
        if cond_output.shape[1] > C:
            # Split into noise prediction and variance prediction
            cond_eps, cond_var = cond_output[:, :C], cond_output[:, C:]
            uncond_eps = uncond_output[:, :C]
            # Apply guidance only to noise prediction
            guided_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
            # Use conditional model's variance prediction (don't guide variance)
            return th.cat([guided_eps, cond_var], dim=1)
        else:
            # No learned variance, apply guidance to full output
            return uncond_output + self.guidance_scale * (cond_output - uncond_output)

import torch
import torch.nn.functional as F


class OcclusionSelector:
    """Patch-level privacy region selector via occlusion scoring.

    For each patch in the image:
      - u_j (utility cost):  cosine drift when patch j is occluded
      - p_j (privacy gain):  reconstruction loss increase when patch j is occluded
      - s_j = p_j / (u_j + eps):  composite score (high = good defense target)

    Selects top-k patches with highest s_j as privacy-critical regions.
    """

    def __init__(self, patch_size=16, top_k_ratio=0.15,
                 mode='mean', eps=1e-3, batch_size=64):
        self.patch_size = patch_size
        self.top_k_ratio = top_k_ratio
        self.mode = mode          # 'mean', 'zero', 'blur'
        self.eps = eps
        self.batch_size = batch_size  # sub-batch for occluded images

    def _occlude_patch(self, image, i, j):
        """Create a copy of image with patch (i, j) occluded."""
        occ = image.clone()
        ph, pw = self.patch_size, self.patch_size
        y0, y1 = i * ph, (i + 1) * ph
        x0, x1 = j * pw, (j + 1) * pw

        if self.mode == 'mean':
            occ[:, :, y0:y1, x0:x1] = image.mean()
        elif self.mode == 'zero':
            occ[:, :, y0:y1, x0:x1] = 0.0
        elif self.mode == 'blur':
            # Simple box blur approximation: replace with local mean
            region = image[:, :, y0:y1, x0:x1]
            occ[:, :, y0:y1, x0:x1] = region.mean(dim=(2, 3), keepdim=True)
        return occ

    @torch.no_grad()
    def score_patches(self, image, encoder, reconstructor):
        """Compute per-patch scores for a single image.

        Args:
            image: (1, 3, H, W) tensor in [0, 1]
            encoder: CLIPEncoder instance
            reconstructor: Reconstructor instance (eval mode)

        Returns:
            scores: (nh, nw) tensor of composite scores s_j
            u_scores: (nh, nw) utility costs
            p_scores: (nh, nw) privacy gains
        """
        device = image.device
        _, _, H, W = image.shape
        ph = pw = self.patch_size
        nh, nw = H // ph, W // pw
        num_patches = nh * nw

        # Original embedding and reconstruction loss
        z_orig = encoder.encode(image)
        recon_orig = reconstructor(z_orig.float())
        loss_base = F.l1_loss(recon_orig, image)

        # Prepare all occluded versions
        occluded_list = []
        for i in range(nh):
            for j in range(nw):
                occluded_list.append(self._occlude_patch(image, i, j))

        # Process in sub-batches
        all_z = []
        all_recon = []
        for start in range(0, num_patches, self.batch_size):
            end = min(start + self.batch_size, num_patches)
            batch = torch.cat(occluded_list[start:end], dim=0)
            z_batch = encoder.encode(batch)
            recon_batch = reconstructor(z_batch.float())
            all_z.append(z_batch)
            all_recon.append(recon_batch)

        z_occ = torch.cat(all_z, dim=0)        # (N, embed_dim)
        recon_occ = torch.cat(all_recon, dim=0)  # (N, 3, H, W)

        # Utility cost: cosine drift per patch
        cos_sims = F.cosine_similarity(z_occ, z_orig.expand_as(z_occ), dim=-1)
        u = 1.0 - cos_sims  # (N,)

        # Privacy gain: reconstruction loss increase per patch
        image_expanded = image.expand(num_patches, -1, -1, -1)
        per_sample_loss = F.l1_loss(
            recon_occ, image_expanded, reduction='none'
        ).mean(dim=(1, 2, 3))  # (N,)
        p = per_sample_loss - loss_base  # (N,)

        # Clamp negative privacy gains (occlusion improved reconstruction = not useful)
        p = p.clamp(min=0.0)

        # Composite score
        scores = p / (u + self.eps)

        return (
            scores.view(nh, nw),
            u.view(nh, nw),
            p.view(nh, nw),
        )

    def select_top_k(self, scores, image_shape):
        """Generate binary mask from top-k scored patches.

        Args:
            scores: (nh, nw) patch scores
            image_shape: (B, C, H, W) for output mask size

        Returns:
            mask: (1, 1, H, W) binary mask, 1 = selected for perturbation
        """
        nh, nw = scores.shape
        num_patches = nh * nw
        k = max(1, int(num_patches * self.top_k_ratio))

        flat = scores.flatten()
        _, topk_indices = torch.topk(flat, k)

        # Build spatial mask
        _, _, H, W = image_shape
        mask = torch.zeros(1, 1, H, W, device=scores.device)
        ph = pw = self.patch_size

        for idx in topk_indices:
            i = idx // nw
            j = idx % nw
            mask[:, :, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = 1.0

        return mask

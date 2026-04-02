"""Block-level occlusion analysis for reconstruction sensitivity scoring.

Implements Phase 2 & 3 of the privacy protection pipeline:
  Phase 2: Compute per-patch reconstruction sensitivity p_j
  Phase 3: Select top-M candidate patches for further VLM analysis
"""

import torch
import torch.nn.functional as F


class OcclusionAnalyzer:
    """Patch-level reconstruction sensitivity analyzer.

    For each patch in the image, measures how much reconstruction quality
    degrades when that patch is occluded — indicating how critical it is
    for the attacker's inversion process.

    Sensitivity score per patch j:
        p_j = L_rec(R(E(x_{-j})), x) - L_rec(R(E(x)), x)

    Higher p_j → the patch is more critical for reconstruction → candidate
    for privacy protection.
    """

    def __init__(self, patch_size=16, occlusion_mode='mean',
                 loss_fn='l1', batch_size=64,
                 center_prior_weight=0.3, center_prior_sigma=0.4):
        """
        Args:
            patch_size:           size of each square patch (pixels)
            occlusion_mode:       how to fill occluded patches ('mean', 'zero', 'blur')
            loss_fn:              reconstruction loss type ('l1', 'l2', 'l1+l2')
            batch_size:           sub-batch size for processing occluded images
            center_prior_weight:  blend weight for spatial center prior (0=off, 1=full)
            center_prior_sigma:   Gaussian spread (fraction of image, smaller=tighter)
        """
        self.patch_size = patch_size
        self.occlusion_mode = occlusion_mode
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.center_prior_weight = center_prior_weight
        self.center_prior_sigma = center_prior_sigma

    def _build_center_prior(self, nh, nw, device):
        """Build a 2D Gaussian spatial prior centered on the image.

        Returns:
            prior: (nh, nw) tensor in [0, 1], peak at center
        """
        sigma = self.center_prior_sigma
        rows = torch.linspace(-1, 1, nh, device=device)
        cols = torch.linspace(-1, 1, nw, device=device)
        gy, gx = torch.meshgrid(rows, cols, indexing='ij')
        prior = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
        # Normalize to [0, 1]
        prior = prior / prior.max()
        return prior

    def _occlude_patch(self, image, row, col):
        """Create image copy with patch at (row, col) occluded."""
        occ = image.clone()
        ps = self.patch_size
        y0, y1 = row * ps, (row + 1) * ps
        x0, x1 = col * ps, (col + 1) * ps

        if self.occlusion_mode == 'mean':
            occ[:, :, y0:y1, x0:x1] = image.mean()
        elif self.occlusion_mode == 'zero':
            occ[:, :, y0:y1, x0:x1] = 0.0
        elif self.occlusion_mode == 'blur':
            region = image[:, :, y0:y1, x0:x1]
            occ[:, :, y0:y1, x0:x1] = region.mean(dim=(2, 3), keepdim=True)
        return occ

    def _compute_rec_loss(self, recon, target):
        """Compute per-sample reconstruction loss."""
        if self.loss_fn == 'l1':
            return F.l1_loss(recon, target, reduction='none').mean(dim=(1, 2, 3))
        elif self.loss_fn == 'l2':
            return F.mse_loss(recon, target, reduction='none').mean(dim=(1, 2, 3))
        elif self.loss_fn == 'l1+l2':
            l1 = F.l1_loss(recon, target, reduction='none').mean(dim=(1, 2, 3))
            l2 = F.mse_loss(recon, target, reduction='none').mean(dim=(1, 2, 3))
            return l1 + l2

    @torch.no_grad()
    def compute_sensitivity(self, image, encoder, reconstructor):
        """Compute reconstruction sensitivity p_j for every patch.

        Args:
            image:          (1, 3, H, W) tensor in [0, 1]
            encoder:        CLIPEncoder instance
            reconstructor:  Reconstructor or ImprovedReconstructor (eval mode)

        Returns:
            p_scores:  (nh, nw) tensor — reconstruction sensitivity per patch
        """
        device = image.device
        _, _, H, W = image.shape
        ps = self.patch_size
        nh, nw = H // ps, W // ps
        num_patches = nh * nw

        # Baseline: reconstruct from original image
        z_orig = encoder.encode(image)
        recon_orig = reconstructor(z_orig.float())
        loss_base = self._compute_rec_loss(
            recon_orig, image
        ).item()

        # Build all occluded image variants
        occluded_list = []
        for i in range(nh):
            for j in range(nw):
                occluded_list.append(self._occlude_patch(image, i, j))

        # Process in sub-batches for memory efficiency
        all_losses = []
        for start in range(0, num_patches, self.batch_size):
            end = min(start + self.batch_size, num_patches)
            batch = torch.cat(occluded_list[start:end], dim=0)
            z_batch = encoder.encode(batch)
            recon_batch = reconstructor(z_batch.float())
            # Compare each reconstruction against the original image
            target = image.expand(end - start, -1, -1, -1)
            losses = self._compute_rec_loss(recon_batch, target)
            all_losses.append(losses)

        per_patch_loss = torch.cat(all_losses, dim=0)  # (num_patches,)

        # Sensitivity: how much reconstruction worsens when patch is removed
        p_scores = per_patch_loss - loss_base
        # Clamp: negative values mean occlusion helped reconstruction (not useful)
        p_scores = p_scores.clamp(min=0.0)

        p_scores = p_scores.view(nh, nw)

        # Apply spatial center prior: boost scores near image center
        if self.center_prior_weight > 0:
            prior = self._build_center_prior(nh, nw, device)
            # Blend: p' = (1 - w) * p_norm + w * prior, then scale back
            p_max = p_scores.max()
            if p_max > 1e-8:
                p_norm = p_scores / p_max
                p_scores = ((1 - self.center_prior_weight) * p_norm
                            + self.center_prior_weight * prior) * p_max

        return p_scores

    def select_candidates(self, p_scores, top_m_ratio=0.3, min_candidates=3):
        """Select top-M candidate patches by reconstruction sensitivity.

        Args:
            p_scores:       (nh, nw) sensitivity scores
            top_m_ratio:    fraction of patches to select as candidates
            min_candidates: minimum number of candidates

        Returns:
            candidates: list of dicts with keys:
                'index', 'row', 'col', 'p_score',
                'y0', 'y1', 'x0', 'x1' (pixel coords)
        """
        nh, nw = p_scores.shape
        num_patches = nh * nw
        m = max(min_candidates, int(num_patches * top_m_ratio))
        m = min(m, num_patches)

        flat = p_scores.flatten()
        topk_vals, topk_idx = torch.topk(flat, m)

        ps = self.patch_size
        candidates = []
        for k in range(m):
            idx = topk_idx[k].item()
            row = idx // nw
            col = idx % nw
            candidates.append({
                'index': idx,
                'row': row,
                'col': col,
                'p_score': topk_vals[k].item(),
                'y0': row * ps,
                'y1': (row + 1) * ps,
                'x0': col * ps,
                'x1': (col + 1) * ps,
            })

        return candidates

"""Score fusion module (Phase 5).

Combines reconstruction sensitivity (p_j) and semantic privacy score (q_j)
into a final composite score (s_j) for each candidate patch, then selects
the top-K patches for protection.
"""

import torch
import numpy as np


class ScoreFusion:
    """Fuse reconstruction sensitivity and VLM semantic privacy scores.

    Supports two fusion modes:
      - multiplicative: s_j = p_j^alpha * q_j^beta
      - linear:         s_j = alpha * p_norm_j + beta * q_norm_j

    Higher fused score → higher priority for protection.
    """

    def __init__(self, alpha=1.0, beta=1.0, mode='multiplicative'):
        """
        Args:
            alpha:  weight for reconstruction sensitivity
            beta:   weight for semantic privacy
            mode:   'multiplicative' or 'linear'
        """
        assert mode in ('multiplicative', 'linear'), \
            f"mode must be 'multiplicative' or 'linear', got '{mode}'"
        self.alpha = alpha
        self.beta = beta
        self.mode = mode

    @staticmethod
    def _normalize(scores):
        """Min-max normalize scores to [0, 1]."""
        s_min = min(scores)
        s_max = max(scores)
        rng = s_max - s_min
        if rng < 1e-8:
            return [0.5] * len(scores)
        return [(s - s_min) / rng for s in scores]

    def fuse(self, candidates):
        """Compute fused scores for candidates.

        Args:
            candidates: list of dicts, each must contain:
                'p_score' (float): reconstruction sensitivity
                'q_score' (float): semantic privacy score from VLM

        Returns:
            candidates: same list with added 's_score' field
        """
        if not candidates:
            return candidates

        p_vals = [c['p_score'] for c in candidates]
        q_vals = [c['q_score'] for c in candidates]

        if self.mode == 'multiplicative':
            # Normalize p to [0, 1] since q is already in [0, 1]
            p_norm = self._normalize(p_vals)
            for i, c in enumerate(candidates):
                p_n = max(p_norm[i], 1e-8)  # avoid zero
                q_n = max(q_vals[i], 1e-8)
                c['s_score'] = (p_n ** self.alpha) * (q_n ** self.beta)

        elif self.mode == 'linear':
            p_norm = self._normalize(p_vals)
            q_norm = self._normalize(q_vals)
            total = self.alpha + self.beta
            for i, c in enumerate(candidates):
                c['s_score'] = (
                    self.alpha * p_norm[i] + self.beta * q_norm[i]
                ) / total

        return candidates

    def select_final(self, candidates, top_k_ratio=0.15,
                     total_patches=None, min_patches=1):
        """Select final patches for protection based on fused scores.

        Args:
            candidates:     list of dicts with 's_score' field
            top_k_ratio:    fraction of TOTAL patches to protect
            total_patches:  total number of patches in the image (for ratio calc)
            min_patches:    minimum number of patches to protect

        Returns:
            selected: sorted list of candidates to protect (highest score first)
        """
        if not candidates:
            return []

        # Determine how many to select
        if total_patches is not None:
            k = max(min_patches, int(total_patches * top_k_ratio))
        else:
            k = max(min_patches, int(len(candidates) * top_k_ratio))
        k = min(k, len(candidates))

        # Sort by s_score descending
        ranked = sorted(candidates, key=lambda c: c['s_score'], reverse=True)
        return ranked[:k]

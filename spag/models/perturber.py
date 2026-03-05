import torch
import torch.nn.functional as F


def total_variation(x):
    """Anisotropic total variation of a tensor."""
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


class MaskedPGD:
    """Masked Projected Gradient Descent perturbation generator.

    Optimizes perturbation delta only within selected mask regions to:
      - Maximize reconstruction loss (make inversion harder)
      - Minimize embedding drift  (preserve retrieval utility)
      - Encourage spatial smoothness (optional TV regularization)

    Loss:  L = -L_rec(R(E(x')), x)  +  lambda_util * (1 - cos(E(x'), z))
                                     +  lambda_smooth * TV(mask * delta)
    """

    def __init__(self, epsilon=0.03, alpha=0.005, num_steps=10,
                 lambda_util=1.0, lambda_smooth=0.01):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.lambda_util = lambda_util
        self.lambda_smooth = lambda_smooth

    def perturb(self, image, mask, z_orig, encoder, reconstructor):
        """Generate adversarially perturbed image.

        Args:
            image:    (1, 3, H, W) original image in [0, 1]
            mask:     (1, 1, H, W) binary mask (1 = perturb this region)
            z_orig:   (1, embed_dim) original embedding (detached)
            encoder:  CLIPEncoder (frozen, used with grad)
            reconstructor: Reconstructor (frozen, used with grad)

        Returns:
            perturbed: (1, 3, H, W) perturbed image in [0, 1]
        """
        device = image.device
        delta = torch.zeros_like(image, device=device, requires_grad=True)
        z_orig = z_orig.detach()

        for _ in range(self.num_steps):
            # Apply masked perturbation
            perturbed = torch.clamp(image + mask * delta, 0.0, 1.0)

            # Forward through encoder and reconstructor
            z_pert = encoder.encode_with_grad(perturbed)
            recon = reconstructor(z_pert)

            # Privacy loss: maximize reconstruction error
            loss_priv = F.l1_loss(recon, image)

            # Utility loss: minimize embedding drift
            cos_sim = F.cosine_similarity(z_pert, z_orig, dim=-1).mean()
            loss_util = 1.0 - cos_sim

            # Smoothness loss
            loss_smooth = total_variation(mask * delta)

            # Total loss (minimize → maximize privacy, minimize utility drift)
            loss = -loss_priv + self.lambda_util * loss_util \
                + self.lambda_smooth * loss_smooth

            loss.backward()

            with torch.no_grad():
                # Signed gradient step (only within mask)
                grad_sign = delta.grad.sign()
                delta.data = delta.data - self.alpha * grad_sign
                # Project onto epsilon-ball
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                # Zero out gradients for next iteration
                delta.grad.zero_()

        # Final perturbed image
        with torch.no_grad():
            perturbed = torch.clamp(image + mask * delta, 0.0, 1.0)

        return perturbed

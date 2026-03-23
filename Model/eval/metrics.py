import torch
import torch.nn.functional as F
import lpips as lpips_pkg
from pytorch_msssim import ssim as _ssim_fn


_lpips_fn = None


def _get_lpips(device):
    global _lpips_fn
    if _lpips_fn is None or next(_lpips_fn.parameters()).device != device:
        _lpips_fn = lpips_pkg.LPIPS(net='alex').to(device).eval()
    return _lpips_fn


def psnr(img1, img2):
    """Peak Signal-to-Noise Ratio. Inputs in [0, 1]."""
    mse = F.mse_loss(img1, img2)
    if mse.item() < 1e-10:
        return 100.0
    return (10.0 * torch.log10(1.0 / mse)).item()


def ssim(img1, img2):
    """Structural Similarity Index. Inputs in [0, 1]."""
    return _ssim_fn(img1, img2, data_range=1.0, size_average=True).item()


def compute_lpips(img1, img2, device='cpu'):
    """Learned Perceptual Image Patch Similarity. Inputs in [0, 1]."""
    fn = _get_lpips(device)
    with torch.no_grad():
        return fn(img1 * 2 - 1, img2 * 2 - 1).mean().item()


def cosine_similarity(z1, z2):
    return F.cosine_similarity(z1, z2, dim=-1).mean().item()


def compute_all_metrics(original, recon_orig, recon_protected,
                        z_orig, z_protected):
    """Compute privacy and utility metrics.

    Args:
        original:        (B, 3, H, W) original images
        recon_orig:      (B, 3, H, W) reconstruction from unprotected embedding
        recon_protected: (B, 3, H, W) reconstruction from protected embedding
        z_orig:          (B, D) original embeddings
        z_protected:     (B, D) protected embeddings

    Returns:
        dict of metric name → float value
    """
    device = original.device
    return {
        # Privacy metrics (how bad is reconstruction?)
        'psnr_orig':      psnr(original, recon_orig),
        'psnr_protected': psnr(original, recon_protected),
        'ssim_orig':      ssim(original, recon_orig),
        'ssim_protected': ssim(original, recon_protected),
        'lpips_orig':      compute_lpips(original, recon_orig, device),
        'lpips_protected': compute_lpips(original, recon_protected, device),
        'mse_orig':      F.mse_loss(original, recon_orig).item(),
        'mse_protected': F.mse_loss(original, recon_protected).item(),
        # Utility metric (embedding drift)
        'cos_sim':   cosine_similarity(z_orig, z_protected),
        'cos_drift': 1.0 - cosine_similarity(z_orig, z_protected),
    }

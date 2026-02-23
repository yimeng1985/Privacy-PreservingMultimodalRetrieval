"""
Helpers for mixed-precision training.

This module uses PyTorch's native AMP (Automatic Mixed Precision) instead of
the legacy manual FP16 approach (which suffered from activation overflow in
FP16 conv layers as parameter norms grew during training).

AMP keeps model weights in FP32 and uses torch.amp.autocast to automatically
run suitable operations (conv, matmul) in FP16 while keeping numerically
sensitive operations (normalization, loss computation) in FP32.
"""

import contextlib
import math

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


# ---------------------------------------------------------------------------
# Legacy helpers – kept for checkpoint compatibility and imports elsewhere.
# ---------------------------------------------------------------------------

def convert_module_to_f16(l):
    """Convert primitive modules to float16."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """Convert primitive modules to float32."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def make_master_params(param_groups_and_shapes):
    """Legacy: Copy model parameters into flattened full-precision parameters."""
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """Legacy: Copy gradients from model parameters to master parameters."""
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """Legacy: Copy master parameter data back into model parameters."""
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    """Convert master params list back to a model state dict for saving."""
    # With AMP, param_groups_and_shapes is None and master_params are
    # the model params themselves, so we always use the simple path.
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    """Convert a state dict to a list of master params (for EMA loading)."""
    # With AMP, master_params are individual FP32 tensors.
    return [state_dict[name] for name, _ in model.named_parameters()]


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


# ---------------------------------------------------------------------------
# MixedPrecisionTrainer – uses PyTorch AMP
# ---------------------------------------------------------------------------

class MixedPrecisionTrainer:
    """
    Mixed precision trainer using PyTorch's native AMP with bfloat16.

    When use_fp16=True:
      - Model weights stay in FP32 (no manual conversion).
      - Forward pass uses torch.amp.autocast(bfloat16) for mixed precision.
      - No GradScaler needed: bfloat16 has the same exponent range as FP32
        (up to ~3.4e38), so gradient underflow/overflow is not a concern.
      - Gradient anomaly detection: steps with gradient norm > GRAD_SPIKE_THRESH
        are skipped entirely to prevent catastrophic parameter updates that can
        cause irreversible model collapse (output → zero).

    When use_fp16=False:
      - Pure FP32 training.
    """

    # Adaptive gradient spike detection:
    # If a step's grad_norm exceeds SPIKE_FACTOR * running_avg AND the
    # absolute threshold, skip it.  Diffusion models have inherently high
    # gradient variance due to timestep sampling (t≈0 → large grads), so
    # normal variance can easily reach 10-30× the EMA.  The original 10x
    # factor caused false positives.  Now the root cause (manual FP16
    # overflow) is fixed by AMP bfloat16, so this is only a safety net for
    # truly catastrophic spikes.
    SPIKE_FACTOR = 50.0
    # Absolute minimum threshold — never flag grad_norm below this value.
    # Normal training range is 0.03-0.5; even timestep-driven variance
    # rarely exceeds 3.0.  Only catastrophic events (NaN-propagation,
    # data corruption) push beyond 5.0.
    SPIKE_ABS_MIN = 5.0

    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,  # kept for interface compat, unused
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,  # kept for compat, unused
    ):
        self.model = model
        self.use_fp16 = use_fp16

        self.model_params = list(self.model.parameters())
        # With AMP, master params ARE the model params (all FP32).
        self.master_params = self.model_params
        # None signals AMP mode for state_dict functions.
        self.param_groups_and_shapes = None

        # Exponential moving average of grad_norm for spike detection.
        self._grad_norm_ema = None  # Initialized on first step

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_amp(opt)
        else:
            return self._optimize_normal(opt)

    @contextlib.contextmanager
    def autocast_ctx(self):
        """Context manager for mixed-precision forward pass.

        Uses bfloat16 instead of float16 because bfloat16 has the same exponent
        range as float32 (up to ~3.4e38), preventing the activation overflow
        that occurs with float16 (max 65504) when parameter norms grow large.
        """
        if self.use_fp16:
            with th.amp.autocast("cuda", dtype=th.bfloat16):
                yield
        else:
            yield

    def _optimize_amp(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()

        if check_overflow(grad_norm):
            logger.log(f"Found NaN/Inf in gradients, skipping step")
            zero_grad(self.master_params)
            return False

        # Adaptive gradient spike detection using exponential moving average.
        # Initialize EMA on first step; then check subsequent steps.
        if self._grad_norm_ema is None:
            self._grad_norm_ema = grad_norm
        else:
            spike_thresh = max(
                self._grad_norm_ema * self.SPIKE_FACTOR,
                self.SPIKE_ABS_MIN,
            )
            if grad_norm > spike_thresh:
                logger.log(
                    f"Gradient spike detected (grad_norm={grad_norm:.4f}, "
                    f"ema={self._grad_norm_ema:.4f}, "
                    f"thresh={spike_thresh:.4f}), skipping step"
                )
                # Don't include spike values in the periodic log dump —
                # they inflate the mean and make the logs misleading.
                # Don't update EMA on spikes — they'd pollute the estimate.
                zero_grad(self.master_params)
                return False
            # Update EMA with current grad_norm (only on non-spike steps).
            self._grad_norm_ema = 0.99 * self._grad_norm_ema + 0.01 * grad_norm

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        # Gradient clipping as additional safety layer.
        th.nn.utils.clip_grad_norm_(self.master_params, 1.0)

        opt.step()
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)

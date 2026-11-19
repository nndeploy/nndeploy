"""Public dispatch for the LTX RMSNorm + AdaLN optimization site."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from nndeploy.e_dit.layers.operator_fusion.config import RMSNORM_ADALN_OP, RMSNORM_ADALN_NATIVE_OP
from nndeploy.e_dit.layers.operator_fusion.registry import OperatorFusionDecision
from nndeploy.e_dit.layers.operator_fusion.runtime import record_operator_fusion_call

_EraserDiT_WIDTH = 2048
_EraserDiT_NORM_EPS = 1e-6


def native_norm_failure(norm):
    from diffusers.models.normalization import RMSNorm
    if type(norm) is not RMSNorm or getattr(norm.forward, '__func__', None) is not RMSNorm.forward:
        return 'norm_implementation'
    if torch.is_grad_enabled():
        return 'autograd'
    return None


def apply_fused_rmsnorm_adaln_native(hidden, scale, shift, norm, *, decision):
    """Keep native variance/rsqrt and fuse the remaining RMSNorm/AdaLN work."""
    failure = _capability_failure(hidden, scale, shift, norm) or native_norm_failure(norm)
    enabled = RMSNORM_ADALN_NATIVE_OP in decision.effective_ops
    if enabled and failure is not None and decision.forced:
        raise RuntimeError(f'forced {RMSNORM_ADALN_NATIVE_OP} capability check failed: {failure}')
    fused = enabled and failure is None
    record_operator_fusion_call(RMSNORM_ADALN_NATIVE_OP, shape=tuple(hidden.shape),
        eligible=failure is None, fused=fused,
        fallback_reason=None if fused else failure or 'op_not_effective')
    if not fused:
        return norm(hidden) * (1 + scale) + shift
    from nndeploy.e_dit.layers.operator_fusion.triton.rmsnorm_native import triton_rmsnorm_adaln_native
    return triton_rmsnorm_adaln_native(hidden, scale, shift, norm.eps)


def _capability_failure(
    normalized_hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm: Any,
) -> str | None:
    tensors = (normalized_hidden_states, scale, shift)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return "tensor_contract"
    if normalized_hidden_states.ndim != 3:
        return "hidden_shape"
    if normalized_hidden_states.shape[0] != 1:
        return "batch_size"
    if normalized_hidden_states.shape[-1] != _EraserDiT_WIDTH:
        return "hidden_width"
    expected_modulation_shape = (1, 1, _EraserDiT_WIDTH)
    if (
        scale.shape != expected_modulation_shape
        or shift.shape != expected_modulation_shape
    ):
        return "modulation_shape"
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        return "tensor_dtype"
    if not all(tensor.is_cuda for tensor in tensors):
        return "tensor_device"
    if (
        scale.device != normalized_hidden_states.device
        or shift.device != normalized_hidden_states.device
    ):
        return "tensor_device"
    if not normalized_hidden_states.is_contiguous():
        return "hidden_layout"
    if scale.stride(-1) != 1 or shift.stride(-1) != 1:
        return "modulation_layout"
    if (
        getattr(norm, "weight", None) is not None
        or getattr(norm, "bias", None) is not None
    ):
        return "norm_affine"
    if float(getattr(norm, "eps", float("nan"))) != _EraserDiT_NORM_EPS:
        return "norm_epsilon"
    return None


def apply_fused_rmsnorm_adaln(
    normalized_hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm: Any,
    *,
    decision: OperatorFusionDecision,
    reference: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Fuse AdaLN modulation after the native, semantics-preserving RMSNorm.

    The public operator represents the complete RMSNorm + AdaLN optimization
    site. The RMSNorm result remains an explicit BF16 tensor because replacing
    diffusers' reduction changes observable model numerics. Triton combines the
    three following BF16 pointwise expressions into one launch while retaining
    each eager rounding boundary.
    """

    failure = _capability_failure(normalized_hidden_states, scale, shift, norm)
    eligible = failure is None
    if RMSNORM_ADALN_OP not in decision.effective_ops:
        record_operator_fusion_call(
            RMSNORM_ADALN_OP,
            shape=tuple(normalized_hidden_states.shape),
            eligible=eligible,
            fused=False,
            fallback_reason=failure or "op_not_effective",
        )
        return reference()
    if failure is not None:
        if decision.forced:
            raise RuntimeError(
                f"forced {RMSNORM_ADALN_OP} capability check failed: {failure}"
            )
        record_operator_fusion_call(
            RMSNORM_ADALN_OP,
            shape=tuple(normalized_hidden_states.shape),
            eligible=False,
            fused=False,
            fallback_reason=failure,
        )
        return reference()

    from nndeploy.e_dit.layers.operator_fusion.triton.rmsnorm_adaln import triton_adaln_modulation

    output = triton_adaln_modulation(normalized_hidden_states, scale, shift)
    record_operator_fusion_call(
        RMSNORM_ADALN_OP,
        shape=tuple(normalized_hidden_states.shape),
        eligible=True,
        fused=True,
    )
    return output

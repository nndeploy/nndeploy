"""Optional BF16 gated residual with the eager multiplication rounding intact."""
import torch

from nndeploy.e_dit.layers.operator_fusion.config import GATED_RESIDUAL_OP
from nndeploy.e_dit.layers.operator_fusion.runtime import record_operator_fusion_call


def _capability_failure(hidden, update, gate):
    if hidden.ndim != 3 or update.shape != hidden.shape or hidden.numel() == 0:
        return 'hidden_shape'
    if gate.shape != (hidden.shape[0], 1, hidden.shape[-1]):
        return 'gate_shape'
    if any(t.dtype != torch.bfloat16 for t in (hidden, update, gate)):
        return 'tensor_dtype'
    if not hidden.is_cuda or any(t.device != hidden.device for t in (update, gate)):
        return 'tensor_device'
    if not all(t.is_contiguous() for t in (hidden, update)) or gate.stride(-1) != 1:
        return 'tensor_layout'
    if torch.is_grad_enabled() and any(t.requires_grad for t in (hidden, update, gate)):
        return 'autograd'
    return None


def apply_fused_gated_residual(hidden, update, gate, *, decision):
    if GATED_RESIDUAL_OP not in decision.requested_ops:
        return hidden + update * gate
    failure = _capability_failure(hidden, update, gate)
    enabled = GATED_RESIDUAL_OP in decision.effective_ops
    if failure is not None and enabled and decision.forced:
        raise RuntimeError(f'forced {GATED_RESIDUAL_OP} capability check failed: {failure}')
    fused = enabled and failure is None
    record_operator_fusion_call(
        GATED_RESIDUAL_OP, shape=tuple(hidden.shape), eligible=failure is None,
        fused=fused, fallback_reason=None if fused else failure or 'op_not_effective',
    )
    if not fused:
        return hidden + update * gate
    from nndeploy.e_dit.layers.operator_fusion.triton.gated_residual import triton_gated_residual
    return triton_gated_residual(hidden, update, gate)

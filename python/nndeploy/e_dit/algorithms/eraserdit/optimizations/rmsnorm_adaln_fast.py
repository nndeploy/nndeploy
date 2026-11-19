"""Optional approximate full RMSNorm + AdaLN for single-GPU inference."""
import torch
from diffusers.models.normalization import RMSNorm

from nndeploy.e_dit.layers.operator_fusion.config import RMSNORM_ADALN_FAST_OP
from nndeploy.e_dit.algorithms.eraserdit.optimizations.rmsnorm_adaln import _capability_failure
from nndeploy.e_dit.layers.operator_fusion.runtime import record_operator_fusion_call


def apply_fused_rmsnorm_adaln_fast(hidden, scale, shift, norm, *, decision):
    # Unlike the exact modulation dispatcher, this API receives raw hidden states.
    failure = _capability_failure(hidden, scale, shift, norm)
    if failure is None:
        if type(norm) is not RMSNorm:
            failure = 'norm_implementation'
        elif hidden.numel() == 0:
            failure = 'empty_input'
        elif torch.is_grad_enabled() and any(t.requires_grad for t in (hidden, scale, shift)):
            failure = 'autograd'
    enabled = RMSNORM_ADALN_FAST_OP in decision.effective_ops
    if enabled and failure is not None and decision.forced:
        raise RuntimeError(f'forced {RMSNORM_ADALN_FAST_OP} capability check failed: {failure}')
    fused = enabled and failure is None
    record_operator_fusion_call(
        RMSNORM_ADALN_FAST_OP, shape=tuple(hidden.shape), eligible=failure is None,
        fused=fused, fallback_reason=None if fused else failure or 'op_not_effective',
    )
    if not fused:
        return norm(hidden) * (1 + scale) + shift
    from nndeploy.e_dit.layers.operator_fusion.triton.rmsnorm_adaln_fast import triton_rmsnorm_adaln_fast
    return triton_rmsnorm_adaln_fast(hidden, scale, shift, norm.eps)

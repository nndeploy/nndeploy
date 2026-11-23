"""Public dispatch and capability contract for paired Q/K RMSNorm + RoPE."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from nndeploy.e_dit.layers.operator_fusion.config import QK_RMSNORM_ROPE_OP, QK_RMSNORM_ROPE_FAST_OP, QK_RMSNORM_ROPE_NATIVE_OP
from nndeploy.e_dit.layers.operator_fusion.registry import OperatorFusionDecision
from nndeploy.e_dit.layers.operator_fusion.runtime import record_operator_fusion_call

_EraserDiT_WIDTH = 2048
_EraserDiT_QK_NORM_EPS = 1e-5


def _capability_failure(
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm: Any,
    key_norm: Any,
    freqs: tuple[torch.Tensor, torch.Tensor] | None,
) -> str | None:
    if freqs is None:
        return "rotary_missing"
    if not isinstance(freqs, tuple) or len(freqs) != 2:
        return "rotary_contract"
    cos, sin = freqs
    tensors = (query, key, cos, sin)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return "tensor_contract"
    if query.ndim != 3 or key.shape != query.shape:
        return "qk_shape"
    if query.shape[-1] != _EraserDiT_WIDTH:
        return "hidden_width"
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        return "qk_dtype"
    if not query.is_cuda or not key.is_cuda:
        return "qk_device"
    if query.device != key.device:
        return "qk_device"
    if not query.is_contiguous() or not key.is_contiguous():
        return "qk_layout"
    if cos.shape != query.shape or sin.shape != query.shape:
        return "rotary_shape"
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        return "rotary_dtype"
    if cos.device != query.device or sin.device != query.device:
        return "rotary_device"
    if not cos.is_contiguous() or not sin.is_contiguous():
        return "rotary_layout"

    query_weight = getattr(query_norm, "weight", None)
    key_weight = getattr(key_norm, "weight", None)
    if not isinstance(query_weight, torch.Tensor) or not isinstance(
        key_weight, torch.Tensor
    ):
        return "norm_weight_missing"
    for weight in (query_weight, key_weight):
        if weight.shape != (_EraserDiT_WIDTH,):
            return "norm_weight_shape"
        if weight.dtype != torch.bfloat16:
            return "norm_weight_dtype"
        if weight.device != query.device:
            return "norm_weight_device"
        if not weight.is_contiguous():
            return "norm_weight_layout"
    if getattr(query_norm, "bias", None) is not None or getattr(
        key_norm, "bias", None
    ) is not None:
        return "norm_bias"
    query_eps = float(getattr(query_norm, "eps", float("nan")))
    key_eps = float(getattr(key_norm, "eps", float("nan")))
    if query_eps != key_eps or query_eps != _EraserDiT_QK_NORM_EPS:
        return "norm_epsilon"
    return None


def apply_fused_qk_rmsnorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm: Any,
    key_norm: Any,
    freqs: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    decision: OperatorFusionDecision,
    reference: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the fused path when selected, otherwise execute the exact caller reference."""

    fast = QK_RMSNORM_ROPE_FAST_OP in decision.effective_ops
    native = QK_RMSNORM_ROPE_NATIVE_OP in decision.effective_ops
    op = QK_RMSNORM_ROPE_NATIVE_OP if native else QK_RMSNORM_ROPE_FAST_OP if fast else QK_RMSNORM_ROPE_OP
    failure = _capability_failure(query, key, query_norm, key_norm, freqs)
    if native and failure is None:
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.rmsnorm_adaln import native_norm_failure
        failure = native_norm_failure(query_norm) or native_norm_failure(key_norm)
    if fast and failure is None:
        from diffusers.models.normalization import RMSNorm
        if type(query_norm) is not RMSNorm or type(key_norm) is not RMSNorm:
            failure = "norm_implementation"
        elif torch.is_grad_enabled() and any(
            t.requires_grad for t in (query, key, query_norm.weight, key_norm.weight, *freqs)
        ):
            failure = "autograd"
    eligible = failure is None
    if op not in decision.effective_ops:
        record_operator_fusion_call(
            op,
            shape=tuple(query.shape),
            eligible=eligible,
            fused=False,
            fallback_reason=failure or "op_not_effective",
        )
        return reference()
    if failure is not None:
        if decision.forced:
            raise RuntimeError(
                f"forced {op} capability check failed: {failure}"
            )
        record_operator_fusion_call(
            op,
            shape=tuple(query.shape),
            eligible=False,
            fused=False,
            fallback_reason=failure,
        )
        return reference()

    cos, sin = freqs
    if native:
        from nndeploy.e_dit.layers.operator_fusion.triton.rmsnorm_native import triton_qk_rmsnorm_rope_native
        fused_query, fused_key = triton_qk_rmsnorm_rope_native(
            query, key, query_norm.weight, key_norm.weight, cos, sin, query_norm.eps)
    elif fast:
        from nndeploy.e_dit.layers.operator_fusion.triton.qk_rmsnorm_rope_fast import triton_qk_rmsnorm_rope_fast
        fused_query, fused_key = triton_qk_rmsnorm_rope_fast(
            query, key, query_norm.weight, key_norm.weight, cos, sin, query_norm.eps,
        )
    else:
        from nndeploy.e_dit.layers.operator_fusion.triton.qk_rmsnorm_rope import triton_qk_rope
        fused_query, fused_key = triton_qk_rope(
            query_norm(query).contiguous(), key_norm(key).contiguous(), cos, sin,
        )
    record_operator_fusion_call(
        op,
        shape=tuple(query.shape),
        eligible=True,
        fused=True,
    )
    return fused_query, fused_key

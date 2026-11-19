"""EraserDiT fused-op topology and compilation acceptance policy."""

from __future__ import annotations

import importlib.util
from typing import Any

from nndeploy.e_dit.layers.operator_fusion.registry import (
    OperatorFusionRegistration,
    OperatorFusionDecision,
)

from nndeploy.e_dit.layers.operator_fusion.config import (
    GATED_RESIDUAL_OP,
    QK_RMSNORM_ROPE_OP,
    QK_RMSNORM_ROPE_FAST_OP,
    QK_RMSNORM_ROPE_NATIVE_OP,
    RMSNORM_ADALN_OP,
    RMSNORM_ADALN_FAST_OP,
    RMSNORM_ADALN_NATIVE_OP,
    normalize_operator_fusion_backend,
    normalize_operator_fusion_ops,
)


# Topologies supported by the local tensor contracts. This is not a claim that
# every backend/topology combination has passed end-to-end video acceptance.
# Runtime checks still enforce layout, dtype, width and device requirements;
# the global backend default remains ``disabled``.
_REGISTRY = {
    QK_RMSNORM_ROPE_NATIVE_OP: OperatorFusionRegistration(
        name=QK_RMSNORM_ROPE_NATIVE_OP, signed_sp_degrees=frozenset({1, 2, 4})),
    RMSNORM_ADALN_NATIVE_OP: OperatorFusionRegistration(
        name=RMSNORM_ADALN_NATIVE_OP, signed_sp_degrees=frozenset({1, 2, 4})),
    RMSNORM_ADALN_FAST_OP: OperatorFusionRegistration(
        name=RMSNORM_ADALN_FAST_OP,
        signed_sp_degrees=frozenset({1}),
    ),
    QK_RMSNORM_ROPE_FAST_OP: OperatorFusionRegistration(
        name=QK_RMSNORM_ROPE_FAST_OP,
        signed_sp_degrees=frozenset({1}),
    ),
    GATED_RESIDUAL_OP: OperatorFusionRegistration(
        name=GATED_RESIDUAL_OP,
        signed_sp_degrees=frozenset({1}),
    ),
    QK_RMSNORM_ROPE_OP: OperatorFusionRegistration(
        name=QK_RMSNORM_ROPE_OP,
        signed_sp_degrees=frozenset({1, 2, 4}),
    ),
    RMSNORM_ADALN_OP: OperatorFusionRegistration(
        name=RMSNORM_ADALN_OP,
        signed_sp_degrees=frozenset({1, 2, 4}),
    ),
}


def _triton_is_available() -> bool:
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):
        return False


def _resolve_sp_degree(server_args: Any) -> int:
    config = getattr(server_args, 'pipeline_config', None)
    if getattr(config, 'dit_parallel_backend', None) == 'nccl':
        return int(config.sp_degree)
    parallel_context = getattr(server_args, "parallel_context", None)
    plan = getattr(parallel_context, "plan", None)
    if plan is not None:
        return max(1, int(getattr(plan, "sp_degree", 1) or 1))
    return max(1, int(getattr(server_args, "sp_degree", 1) or 1))


def resolve_operator_fusion_decision(server_args: Any) -> OperatorFusionDecision:
    backend = normalize_operator_fusion_backend(
        getattr(server_args, "operator_fusion_backend", "disabled")
    )
    requested_ops = normalize_operator_fusion_ops(
        getattr(server_args, "operator_fusion_ops", None)
    )
    sp_degree = _resolve_sp_degree(server_args)
    # Compilation is confined to block.ff. Attention, AdaLN and residual
    # fusion execute outside that region. Only the single-GPU combination has
    # been validated here; retain the existing restriction for SP compilation.
    compile_sp = bool(getattr(server_args, "enable_torch_compile", False)) and sp_degree > 1
    if backend == "triton" and compile_sp:
        raise ValueError("forced operator fusion with SP torch.compile is unsupported")

    if backend == "disabled":
        return OperatorFusionDecision(
            requested_backend=backend,
            requested_ops=requested_ops,
            effective_ops=(),
            fallback_reasons=("backend_disabled",),
            sp_degree=sp_degree,
            forced=False,
        )

    if not _triton_is_available():
        if backend == "triton":
            raise RuntimeError("Triton is unavailable for forced operator fusion")
        return OperatorFusionDecision(
            requested_backend=backend,
            requested_ops=requested_ops,
            effective_ops=(),
            fallback_reasons=("triton_unavailable",),
            sp_degree=sp_degree,
            forced=False,
        )

    if backend == "triton":
        return OperatorFusionDecision(
            requested_backend=backend,
            requested_ops=requested_ops,
            effective_ops=requested_ops,
            fallback_reasons=(),
            sp_degree=sp_degree,
            forced=True,
        )

    if compile_sp:
        return OperatorFusionDecision(
            requested_backend=backend, requested_ops=requested_ops,
            effective_ops=(), fallback_reasons=("torch_compile_active",),
            sp_degree=sp_degree, forced=False,
        )

    effective_ops: list[str] = []
    fallback_reasons: list[str] = []
    for op_name in requested_ops:
        registration = _REGISTRY[op_name]
        if not registration.signed_sp_degrees:
            fallback_reasons.append("op_not_signed")
        elif sp_degree not in registration.signed_sp_degrees:
            fallback_reasons.append("topology_not_signed")
        else:
            effective_ops.append(op_name)
    return OperatorFusionDecision(
        requested_backend=backend,
        requested_ops=requested_ops,
        effective_ops=tuple(effective_ops),
        fallback_reasons=tuple(dict.fromkeys(fallback_reasons)),
        sp_degree=sp_degree,
        forced=False,
    )


def get_operator_fusion_decision(server_args: Any) -> OperatorFusionDecision:
    decision = getattr(server_args, "operator_fusion_decision", None)
    if isinstance(decision, OperatorFusionDecision):
        return decision
    return resolve_operator_fusion_decision(server_args)

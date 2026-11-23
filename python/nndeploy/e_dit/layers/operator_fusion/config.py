"""Configuration contract for optional fused operators."""

from __future__ import annotations

from collections.abc import Iterable

QK_RMSNORM_ROPE_OP = "qk_rmsnorm_rope"
QK_RMSNORM_ROPE_FAST_OP = "qk_rmsnorm_rope_fast"
QK_RMSNORM_ROPE_NATIVE_OP = "qk_rmsnorm_rope_native"
RMSNORM_ADALN_OP = "rmsnorm_adaln"
RMSNORM_ADALN_FAST_OP = "rmsnorm_adaln_fast"
RMSNORM_ADALN_NATIVE_OP = "rmsnorm_adaln_native"
GATED_RESIDUAL_OP = "gated_residual"

OPERATOR_FUSION_BACKENDS = ("disabled", "auto", "triton")
DEFAULT_OPERATOR_FUSION_OPS = (QK_RMSNORM_ROPE_OP, RMSNORM_ADALN_OP)
OPERATOR_FUSION_OPS = (
    *DEFAULT_OPERATOR_FUSION_OPS, GATED_RESIDUAL_OP,
    QK_RMSNORM_ROPE_FAST_OP, RMSNORM_ADALN_FAST_OP,
    QK_RMSNORM_ROPE_NATIVE_OP, RMSNORM_ADALN_NATIVE_OP,
)


def normalize_operator_fusion_backend(value: object) -> str:
    backend = str(value).strip().lower()
    if backend not in OPERATOR_FUSION_BACKENDS:
        raise ValueError(
            "operator_fusion_backend must be one of "
            f"{OPERATOR_FUSION_BACKENDS}, got {value!r}"
        )
    return backend


def normalize_operator_fusion_ops(
    value: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize explicit selections; unset uses the validated default sites."""

    if value is None:
        requested = DEFAULT_OPERATOR_FUSION_OPS
    elif isinstance(value, str):
        requested = tuple(part.strip() for part in value.split(",") if part.strip())
        if not requested:
            raise ValueError("operator_fusion_ops must not be empty when specified")
    else:
        requested = tuple(str(part).strip() for part in value if str(part).strip())
        if not requested:
            raise ValueError("operator_fusion_ops must not be empty when specified")

    unknown = tuple(op for op in requested if op not in OPERATOR_FUSION_OPS)
    if unknown:
        raise ValueError(
            f"unknown operator fusion op(s): {unknown}; supported={OPERATOR_FUSION_OPS}"
        )
    if len(set(requested) & {QK_RMSNORM_ROPE_OP, QK_RMSNORM_ROPE_FAST_OP, QK_RMSNORM_ROPE_NATIVE_OP}) > 1:
        raise ValueError("select only one Q/K RMSNorm + RoPE implementation")
    if len(set(requested) & {RMSNORM_ADALN_OP, RMSNORM_ADALN_FAST_OP, RMSNORM_ADALN_NATIVE_OP}) > 1:
        raise ValueError("select only one RMSNorm + AdaLN implementation")
    return tuple(dict.fromkeys(requested))

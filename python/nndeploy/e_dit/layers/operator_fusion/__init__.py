"""Optional fused operators for diffusion inference."""

from nndeploy.e_dit.layers.operator_fusion.config import (
    GATED_RESIDUAL_OP,
    OPERATOR_FUSION_BACKENDS,
    QK_RMSNORM_ROPE_OP,
    QK_RMSNORM_ROPE_FAST_OP,
    RMSNORM_ADALN_OP,
    RMSNORM_ADALN_FAST_OP,
    normalize_operator_fusion_backend,
    normalize_operator_fusion_ops,
)
from nndeploy.e_dit.layers.operator_fusion.registry import (
    OperatorFusionDecision,
)

__all__ = [
    "GATED_RESIDUAL_OP",
    "OPERATOR_FUSION_BACKENDS",
    "OperatorFusionDecision",
    "QK_RMSNORM_ROPE_OP",
    "QK_RMSNORM_ROPE_FAST_OP",
    "RMSNORM_ADALN_OP",
    "RMSNORM_ADALN_FAST_OP",
    "normalize_operator_fusion_backend",
    "normalize_operator_fusion_ops",
]

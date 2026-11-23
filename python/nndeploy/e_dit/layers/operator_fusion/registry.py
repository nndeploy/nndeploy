"""Shared fused-op registration and decision data contracts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorFusionRegistration:
    name: str
    signed_sp_degrees: frozenset[int]


@dataclass(frozen=True)
class OperatorFusionDecision:
    requested_backend: str
    requested_ops: tuple[str, ...]
    effective_ops: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    sp_degree: int
    forced: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "requested_ops": list(self.requested_ops),
            "effective_ops": list(self.effective_ops),
            "fallback_reasons": list(self.fallback_reasons),
            "sp_degree": self.sp_degree,
            "forced": self.forced,
        }


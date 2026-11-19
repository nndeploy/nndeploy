"""Request-local observability for fused operators."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

from nndeploy.e_dit.layers.operator_fusion.registry import OperatorFusionDecision


@dataclass
class OperatorFusionRequestStats:
    decision: OperatorFusionDecision
    eligible_calls: dict[str, int] = field(default_factory=dict)
    fused_calls: dict[str, int] = field(default_factory=dict)
    reference_calls: dict[str, int] = field(default_factory=dict)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    observed_shape_buckets: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, object]:
        snapshot = self.decision.as_dict()
        snapshot.update(
            {
                "eligible_calls": dict(self.eligible_calls),
                "fused_calls": dict(self.fused_calls),
                "reference_calls": dict(self.reference_calls),
                "runtime_fallback_reasons": dict(self.fallback_reasons),
                "observed_shape_buckets": dict(self.observed_shape_buckets),
            }
        )
        return snapshot


_CURRENT_STATS: ContextVar[OperatorFusionRequestStats | None] = ContextVar(
    "nndeploy_operator_fusion_stats", default=None
)


@contextmanager
def operator_fusion_request_scope(
    decision: OperatorFusionDecision,
    *, stats: OperatorFusionRequestStats | None = None,
) -> Iterator[OperatorFusionRequestStats]:
    stats = stats if stats is not None else OperatorFusionRequestStats(decision=decision)
    token = _CURRENT_STATS.set(stats)
    try:
        yield stats
    finally:
        _CURRENT_STATS.reset(token)


def record_operator_fusion_call(
    op_name: str,
    *,
    shape: tuple[int, ...],
    eligible: bool,
    fused: bool,
    fallback_reason: str | None = None,
) -> None:
    stats = _CURRENT_STATS.get()
    if stats is None:
        return
    shape_key = f"{op_name}:{'x'.join(str(dim) for dim in shape)}"
    stats.observed_shape_buckets[shape_key] = (
        stats.observed_shape_buckets.get(shape_key, 0) + 1
    )
    target = stats.eligible_calls if eligible else stats.reference_calls
    target[op_name] = target.get(op_name, 0) + 1
    if fused:
        stats.fused_calls[op_name] = stats.fused_calls.get(op_name, 0) + 1
    elif eligible:
        stats.reference_calls[op_name] = stats.reference_calls.get(op_name, 0) + 1
    if fallback_reason is not None:
        stats.fallback_reasons[fallback_reason] = (
            stats.fallback_reasons.get(fallback_reason, 0) + 1
        )

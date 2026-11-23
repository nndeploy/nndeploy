"""Explicit timing scopes and caller-selected stage summaries."""
from __future__ import annotations
import time
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Any
from nndeploy.e_dit.utils.perf_logger import RequestMetrics


@contextmanager
def stage_timer(metrics, stage_name, *, device=None, enabled=False):
    """Opt-in wall timing, including queued CUDA work inside the region.

    Normal inference adds no synchronization. Diagnostic regions are nested
    inside stage timings and must not be added to them as independent work.
    """
    if not enabled or metrics is None:
        yield
        return
    sync = None
    if device is not None:
        import torch
        device = torch.device(device)
        if device.type == 'cuda':
            sync = lambda: torch.cuda.synchronize(device)
    if sync:
        sync()
    started = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            sync()
        metrics.record_stage(stage_name, time.perf_counter() - started)


def _step_stats(metrics: RequestMetrics | None) -> dict[str, Any]:
    """Per-step durations, in order, plus the median.

    The order matters for the report matrix's first-run row: Inductor's
    autotune lands in the leading steps, so the warmup cost is the gap between
    the first entries and the median rather than a separately measurable span.
    """
    steps = list(metrics.steps) if metrics is not None else []
    if not steps:
        return {"step_times_ms": [], "step_median_ms": None, "step_count": 0}
    ordered = sorted(steps)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "step_times_ms": [round(value, 3) for value in steps],
        "step_median_ms": round(median, 3),
        "step_count": len(steps),
    }


def build_pure_timing_payload(
    metrics: RequestMetrics | None,
    *,
    stage_names,
    diagnostic=False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra_payload = dict(extra or {})
    if metrics is None:
        return {
            "request_id": None,
            "pure_inference_seconds": None,
            "pure_inference_stage_names": list(stage_names),
            "pure_inference_stage_breakdown_ms": {},
            "pure_inference_stage_counts": {},
            "pure_inference_operation_counts": {},
            "pipeline_total_seconds": None,
            **_step_stats(None),
            "extra": extra_payload,
        }

    stage_breakdown_ms = {
        stage_name: float(metrics.stages.get(stage_name, 0.0))
        for stage_name in stage_names
        if stage_name in metrics.stages
    }
    stage_counts = {
        stage_name: int(metrics.stage_counts.get(stage_name, 0))
        for stage_name in stage_names
        if stage_name in metrics.stage_counts
    }
    pure_inference_ms = sum(stage_breakdown_ms.values())
    payload = {
        "request_id": metrics.request_id,
        "pure_inference_seconds": pure_inference_ms / 1000.0,
        "pure_inference_stage_names": list(stage_names),
        "pure_inference_stage_breakdown_ms": stage_breakdown_ms,
        "pure_inference_stage_counts": stage_counts,
        "pure_inference_operation_counts": dict(metrics.operation_counts),
        "pipeline_total_seconds": (
            float(metrics.total_duration_ms) / 1000.0
            if metrics.total_duration_ms > 0.0
            else None
        ),
        **_step_stats(metrics),
        "extra": extra_payload,
    }
    if diagnostic:
        payload["diagnostic_stage_breakdown_ms"] = {
            name: float(duration_ms)
            for name, duration_ms in metrics.stages.items()
        }
        payload["diagnostic_stage_counts"] = {
            name: int(count) for name, count in metrics.stage_counts.items()
        }
    return payload


def write_timing_payload(payload: dict[str, Any], output_path) -> str | None:
    if not output_path:
        return None
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=True)
    return str(resolved)

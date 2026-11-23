"""Helpers for writing structured inference timing sidecars."""

from __future__ import annotations

import os
from typing import Any

from nndeploy.e_dit.utils.perf_logger import RequestMetrics

PURE_TIMING_ENV_VAR = "NNDEPLOY_DIT_ERASERDIT_PURE_TIMING_JSON"
DIAGNOSTIC_TIMING_ENV_VAR = "NNDEPLOY_DIT_ERASERDIT_DIAGNOSTIC_TIMING"

# Stage names whose metrics sum to "pure inference" (model work, excluding IO and
# the window/commit machinery).  One entry per supported model family.
PURE_INFERENCE_STAGE_NAMES = (
    "EraseConditionEncodingStage",
    "EraseLatentPreparationStage",
    "EraseTimestepPreparationStage",
    "EraseDenoisingNode",
    "EraseDecodingStage",
    "EraserDiTEraseTextEncodingNode",
    "EraserDiTEraseConditionEncodingNode",
    "EraserDiTEraseLatentPreparationNode",
    "EraserDiTEraseTimestepPreparationNode",
    "EraserDiTEraseDenoisingNode",
    "EraserDiTEraseDecodingNode",
)


def diagnostic_timing_enabled() -> bool:
    return os.environ.get(DIAGNOSTIC_TIMING_ENV_VAR) == "1"


def record_diagnostic_stage(
    metrics: RequestMetrics | None,
    stage_name: str,
    duration_s: float,
) -> None:
    if diagnostic_timing_enabled() and metrics is not None:
        metrics.record_stage(stage_name, duration_s)


def diagnostic_stage_timer(metrics, stage_name, *, device=None):
    from nndeploy.e_dit.utils.inference_timing import stage_timer
    return stage_timer(metrics, stage_name, device=device, enabled=diagnostic_timing_enabled())


def build_pure_timing_payload(metrics, *, extra=None):
    from nndeploy.e_dit.utils.inference_timing import build_pure_timing_payload as build
    return build(metrics, stage_names=PURE_INFERENCE_STAGE_NAMES,
                 diagnostic=diagnostic_timing_enabled(), extra=extra)


def maybe_write_pure_timing_payload(payload):
    from nndeploy.e_dit.utils.inference_timing import write_timing_payload
    return write_timing_payload(payload, os.environ.get(PURE_TIMING_ENV_VAR))

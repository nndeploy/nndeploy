"""Window planning helpers for pipelines.runtime."""

from __future__ import annotations

import torch

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.video.bbox import get_max_bbox
from nndeploy.e_dit.video.windowing import WindowSpec, build_window_specs as plan_windows


def build_window_specs(
    params: EraserDiTEraseSamplingParams,
    scenes: list[tuple[int, int]] | None = None,
) -> list[WindowSpec]:
    return plan_windows(
        num_frames=int(params.num_frames or 0),
        infer_len=int(params.infer_len),
        overlap=int(params.overlap),
        time_sample=int(params.time_sample),
        time_shift=int(params.time_shift),
        scenes=scenes if scenes is not None else params.scenes,
    )


def infer_window_bbox(
    bbox_frames: torch.Tensor | None,
    spec: WindowSpec,
    window_mask: torch.Tensor | None,
) -> tuple[int, int, int, int] | None:
    if bbox_frames is not None:
        bbox_slice = bbox_frames[spec.load_start : spec.load_end].clone()
        if spec.overlap_left > 0:
            bbox_slice[: spec.overlap_left] = 0
        if spec.active_end_offset < bbox_slice.shape[0]:
            bbox_slice[spec.active_end_offset :] = 0
        return get_max_bbox(bbox_slice)

    if window_mask is None:
        raise ValueError("window_mask is required when bbox frames are absent")
    return (
        0,
        0,
        int(window_mask.shape[-1]),
        int(window_mask.shape[-2]),
    )

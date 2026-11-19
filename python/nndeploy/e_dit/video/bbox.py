"""Bounding-box helpers for video processing."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def align_value(value: int, align: int) -> int:
    return int(math.ceil(value / align) * align)


def get_max_bbox(bboxes: torch.Tensor | Iterable[Iterable[int]]) -> tuple[int, int, int, int] | None:
    tensor = torch.as_tensor(list(bboxes) if not isinstance(bboxes, torch.Tensor) else bboxes)
    if tensor.numel() == 0:
        return None
    valid_bboxs = tensor[torch.any(tensor > 0, dim=-1)]
    if valid_bboxs.size(0) < 1:
        return None

    minimum = valid_bboxs.min(dim=0)[0]
    maximum = valid_bboxs.max(dim=0)[0]
    bbox = (
        int(minimum[0]),
        int(minimum[1]),
        int(maximum[2]) - int(minimum[0]),
        int(maximum[3]) - int(minimum[1]),
    )
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return bbox


def scale_and_align_bbox(
    bbox: tuple[int, int, int, int],
    video_width: int,
    video_height: int,
    min_pixels: int = 256 * 256,
    max_pixels: int = 1920 * 1088,
    scale_ratio: float = 1.0,
    align_width: int = 32,
    align_height: int = 32,
    force_align: bool = True,
) -> tuple[int, int, int, int] | None:
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height

    if width <= 0 or height <= 0:
        return None
    if max_pixels > 0 and width * height > max_pixels:
        raise ValueError(
            f"base bbox size out of range: {width}x{height} > max_pixels={max_pixels}"
        )

    bbox_pixels = width * height
    min_scale_ratio = (
        max(math.sqrt(min_pixels / bbox_pixels), 1.0) if min_pixels > 0 else 1.0
    )
    max_scale_ratio = (
        max(math.sqrt(max_pixels / bbox_pixels), min_scale_ratio)
        if max_pixels > 0
        else float("inf")
    )

    if scale_ratio >= 1.0:
        new_scale_ratio = max(min_scale_ratio, min(scale_ratio, max_scale_ratio))
        target_w = min(int(math.ceil(width * new_scale_ratio)), video_width)
        target_h = min(int(math.ceil(height * new_scale_ratio)), video_height)
    elif scale_ratio < 0:
        if (video_width / width) < (video_height / height):
            target_w = video_width
            target_h = max_pixels // target_w
        else:
            target_h = video_height
            target_w = max_pixels // target_h
    else:
        raise ValueError("scale_ratio in (0, 1) is not supported")

    target_w = align_value(target_w, align_width)
    target_h = align_value(target_h, align_height)

    if target_w > target_h:
        target_w = max(
            min(target_w, align_value(int(min_pixels / target_h), align_width)),
            min(align_value(int(width * scale_ratio), align_width), align_value(width, align_width)),
        )
    else:
        target_h = max(
            min(target_h, align_value(int(min_pixels / target_w), align_height)),
            min(align_value(int(height * scale_ratio), align_height), align_value(height, align_height)),
        )

    if max_pixels > 0 and target_w * target_h > max_pixels:
        raise ValueError(
            f"scaled bbox exceeds max_pixels: {target_w}x{target_h} > {max_pixels}"
        )

    if force_align:
        target_w = min(target_w, video_width // align_width * align_width)
        target_h = min(target_h, video_height // align_height * align_height)
    else:
        target_w = min(target_w, video_width)
        target_h = min(target_h, video_height)

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    new_x1 = min(max(0, center_x - target_w // 2), video_width - target_w)
    new_y1 = min(max(0, center_y - target_h // 2), video_height - target_h)
    new_x2 = max(min(new_x1 + target_w, video_width), target_w)
    new_y2 = max(min(new_y1 + target_h, video_height), target_h)
    return (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)


def resolve_single_window_crop_bbox(
    *,
    bbox: tuple[int, int, int, int] | None,
    video_width: int,
    video_height: int,
    align_h: int,
    align_w: int,
    scale_area_ratio: float,
    min_pixels: int,
    max_pixels: int,
    force_crop_align: bool,
) -> tuple[int, int, int, int]:
    source_bbox = bbox or (0, 0, int(video_width), int(video_height))
    aligned_bbox = scale_and_align_bbox(
        bbox=source_bbox,
        video_width=video_width,
        video_height=video_height,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        scale_ratio=scale_area_ratio,
        align_width=align_w,
        align_height=align_h,
        force_align=force_crop_align,
    )
    if aligned_bbox is None:
        raise ValueError("failed to align bbox")
    return aligned_bbox

"""Normalize tensor and file inputs for runtime context preparation."""

from __future__ import annotations

import torch

from nndeploy.e_dit.video.tensors import ensure_5d_video

from nndeploy.e_dit.video.io import (
    binarize_mask_tensor, ensure_nchw_video, read_mask_tensor, read_video_tensor,
)


def _build_runtime_video(
    video_source: torch.Tensor | str,
) -> tuple[torch.Tensor, dict[str, object]]:
    if isinstance(video_source, str):
        video, metadata = read_video_tensor(video_source)
        video = video.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        if video_source.ndim == 4:
            video = ensure_nchw_video(video_source).permute(1, 0, 2, 3).unsqueeze(0)
        elif video_source.ndim == 5:
            video = ensure_5d_video(video_source)
            if video.shape[0] != 1:
                raise ValueError(
                    "Video erase pipeline only supports batch size 1"
                )
        else:
            raise ValueError(
                f"Unsupported video tensor shape: {tuple(video_source.shape)}"
            )
        metadata = {
            "fps": None,
            "codec_name": None,
            "num_frames": int(video.shape[2]),
            "width": int(video.shape[-1]),
            "height": int(video.shape[-2]),
        }
    return ensure_5d_video(video.float(), channels=3), metadata


def _build_runtime_mask(
    mask_source: torch.Tensor | str, num_frames: int
) -> torch.Tensor:
    if isinstance(mask_source, str):
        mask, _metadata = read_mask_tensor(mask_source)
    else:
        if mask_source.ndim == 5:
            mask = mask_source[0].permute(1, 0, 2, 3)
        elif mask_source.ndim == 4:
            mask = ensure_nchw_video(mask_source)
        else:
            raise ValueError(
                f"Unsupported mask tensor shape: {tuple(mask_source.shape)}"
            )
        mask = binarize_mask_tensor(mask)
    if mask.ndim != 4:
        raise ValueError(f"Unsupported mask tensor shape: {tuple(mask.shape)}")
    if mask.shape[0] < num_frames:
        tail = mask[-1:, ...].repeat(num_frames - mask.shape[0], 1, 1, 1)
        mask = torch.cat([mask, tail], dim=0)
    return ensure_5d_video(
        mask[:num_frames].permute(1, 0, 2, 3).unsqueeze(0).float(), channels=1
    )

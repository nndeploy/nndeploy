"""Video tensor shape normalization independent of pipeline policy."""

from __future__ import annotations

import torch


def ensure_5d_video(video: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    """Accept CTHW or BCTHW, adding a batch axis without copying frame data."""
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(f"Expected 4D/5D video tensor, got {tuple(video.shape)}")
    if channels is not None and video.shape[1] != channels:
        raise ValueError(
            f"Expected video channel count {channels}, got {video.shape[1]}"
        )
    return video



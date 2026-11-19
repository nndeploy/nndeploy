"""Latent and video value-range helpers for video processing."""

from __future__ import annotations

import torch


def normalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - latents_mean) * scaling_factor / latents_std


def denormalize_latents(
    latents: torch.Tensor,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    scaling_factor: float = 1.0,
) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * latents_std / scaling_factor + latents_mean


def latent_frame_count(num_frames: int, temporal_ratio: int = 8) -> int:
    return (num_frames - 1) // temporal_ratio + 1


def build_rope_interpolation_scale(
    temporal_ratio: int,
    frame_rate: int,
    spatial_ratio: int,
) -> tuple[float, float, float]:
    return (
        temporal_ratio / float(frame_rate),
        float(spatial_ratio),
        float(spatial_ratio),
    )


def preprocess_video_tensor(video: torch.Tensor) -> torch.Tensor:
    """Match diffusers VideoProcessor preprocess for tensor inputs in [0, 1]."""
    return video.to(torch.float32).mul(2.0).sub(1.0)


def postprocess_video_tensor(video: torch.Tensor) -> torch.Tensor:
    """Match diffusers VideoProcessor postprocess for tensor outputs in [-1, 1]."""
    return video.to(torch.float32).div(2.0).add(0.5).clamp(0.0, 1.0)

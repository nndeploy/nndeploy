"""Common VAE interfaces and parallel tiling primitives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import nn


class ParallelTiledVAE(ABC, nn.Module):
    tile_sample_min_height: int = 0
    tile_sample_min_width: int = 0
    tile_sample_min_num_frames: int = 0
    tile_sample_stride_height: int = 0
    tile_sample_stride_width: int = 0
    tile_sample_stride_num_frames: int = 0
    blend_num_frames: int = 0
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def __init__(self, config: Any, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.config = config
        self.tile_sample_min_height = getattr(config, 'tile_sample_min_height', 0)
        self.tile_sample_min_width = getattr(config, 'tile_sample_min_width', 0)
        self.tile_sample_min_num_frames = getattr(config, 'tile_sample_min_num_frames', 0)
        self.tile_sample_stride_height = getattr(config, 'tile_sample_stride_height', 0)
        self.tile_sample_stride_width = getattr(config, 'tile_sample_stride_width', 0)
        self.tile_sample_stride_num_frames = getattr(config, 'tile_sample_stride_num_frames', 0)
        self.blend_num_frames = getattr(config, 'blend_num_frames', 0)
        self.use_tiling = getattr(config, 'use_tiling', False)
        self.use_temporal_tiling = getattr(config, 'use_temporal_tiling', False)
        self.use_parallel_tiling = getattr(config, 'use_parallel_tiling', False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def temporal_compression_ratio(self) -> int:
        return cast(int, getattr(self.config, 'temporal_compression_ratio', 1))

    @property
    def spatial_compression_ratio(self) -> int:
        return cast(int, getattr(self.config, 'spatial_compression_ratio', 1))

    @property
    def scaling_factor(self) -> float | torch.Tensor:
        return cast(float | torch.Tensor, getattr(self.config, 'scaling_factor', 1.0))

    @abstractmethod
    def _encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        latents = self._encode(x)
        return DiagonalGaussianDistribution(latents)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode(z)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode(z)

    def spatial_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode(z)

    def parallel_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode(z)

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from nndeploy.e_dit.config.model_dits import DiTArchConfig
from nndeploy.e_dit.algorithms.wan.configs.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)


@dataclass
class WanVideoEditArchConfig(WanVideoArchConfig):
    """Wan VideoEdit DiT shape contract.

    VideoEdit concatenates noisy latents, packed masks, and masked-video latents:
    16 + 4 + 16 input channels. The model still predicts 16 latent channels.
    """

    in_channels: int = 36
    out_channels: int = 16
    image_dim: int | None = 1280
    added_kv_proj_dim: int | None = 5120

    def __post_init__(self):
        super().__post_init__()
        self.num_channels_latents = 16


@dataclass
class WanVideoEditConfig(WanVideoConfig):
    arch_config: DiTArchConfig = field(default_factory=WanVideoEditArchConfig)

    prefix: str = "WanVideoEdit"

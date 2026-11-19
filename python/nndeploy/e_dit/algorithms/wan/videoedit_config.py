# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field
import html
import re

import torch

from nndeploy.e_dit.algorithms.wan.encoder_configs import DiTConfig, EncoderConfig, VAEConfig
from nndeploy.e_dit.algorithms.wan.configs.wan_videoedit import (
    WanVideoEditConfig,
)
from nndeploy.e_dit.algorithms.wan.encoder_configs import (
    BaseEncoderOutput,
    CLIPVisionConfig,
)
from nndeploy.e_dit.models.text_encoders.t5_config import T5Config
from nndeploy.e_dit.algorithms.wan.configs.wanvae import WanVAEConfig
from nndeploy.e_dit.config.generation import (
    ModelTaskType,
    PipelineConfig,
)
from nndeploy.e_dit.algorithms.wan.config import t5_postprocess_text


def videoedit_prompt_clean(text: str) -> str:
    try:
        import ftfy
    except ImportError as exc:
        raise RuntimeError(
            "VideoEdit prompt preprocessing requires ftfy==6.3.1; "
            "install nndeploy dependencies"
        ) from exc
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class WanVideoEditPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.VIDEO_EDIT
    dit_config: DiTConfig = field(default_factory=WanVideoEditConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)

    flow_shift: float | None = 5.0
    precision: str = "bf16"
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    generator_device: str | None = "cpu"
    vae_tiling: bool = True
    vae_sp: bool = False
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"
    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(output_hidden_states=True)
    )
    preprocess_text_funcs: tuple[Callable[[str], str] | None, ...] = field(
        default_factory=lambda: (videoedit_prompt_clean,)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    def postprocess_image(self, image):
        return image.hidden_states[-2]

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.arch_config.in_channels = 36
        self.dit_config.arch_config.out_channels = 16
        self.dit_config.arch_config.num_channels_latents = 16
        self.dit_config.arch_config.image_dim = 1280
        self.dit_config.arch_config.added_kv_proj_dim = 5120

    def adjust_num_frames(self, num_frames):
        return num_frames

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        latent_num_frames = (num_frames - 1) // 4 + 1
        return (
            batch_size,
            16,
            latent_num_frames,
            batch.height // 8,
            batch.width // 8,
        )

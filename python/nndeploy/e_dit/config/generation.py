"""Component and tensor contracts for diffusion generation algorithms."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import torch
from .model_dits import DiTConfig
from .model_vaes import VAEConfig

class ModelTaskType(Enum):
    T2V = auto()
    I2V = auto()
    TI2V = auto()
    VIDEO_EDIT = auto()

@dataclass
class PipelineConfig:
    task_type: ModelTaskType = ModelTaskType.T2V
    model_path: str = ''
    flow_shift: float | None = None
    generator_device: str = 'cpu'
    dit_config: Any = field(default_factory=DiTConfig)
    vae_config: Any = field(default_factory=VAEConfig)
    dit_precision: str = 'bf16'
    vae_precision: str = 'fp32'
    vae_tiling: bool = True
    component_loaders: dict = field(default_factory=dict)
    component_architectures: dict = field(default_factory=dict)

    preprocess_text_funcs: tuple = (None,)
    text_encoder_extra_args: list = field(default_factory=lambda: [{}])

    def tokenize_prompt(self, prompt, tokenizer, tok_kwargs):
        return tokenizer(prompt, **tok_kwargs)
    def prepare_latent_shape(self, batch, batch_size, num_frames):
        ratio = self.vae_config.arch_config.spatial_compression_ratio
        return (batch_size, self.dit_config.num_channels_latents, num_frames,
                batch.height // ratio, batch.width // ratio)
    def maybe_pack_latents(self, latents, batch_size, batch):
        return latents
    def adjust_num_frames(self, num_frames):
        return num_frames
    def maybe_prepare_latent_ids(self, latents):
        return None
    def get_latent_dtype(self, prompt_dtype):
        return prompt_dtype
    def preprocess_decoding(self, latents, server_args=None, vae=None):
        return latents
    def shard_latents_for_sp(self, batch, latents):
        from nndeploy.e_dit.distributed.model_parallel import get_sp_world_size, get_sp_parallel_rank
        size = get_sp_world_size()
        if size == 1:
            return latents, False
        if latents.ndim != 5 or latents.shape[2] % size:
            raise ValueError('Temporal shards require divisible BCTHW latents')
        return latents.chunk(size, dim=2)[get_sp_parallel_rank()].contiguous(), True
    def gather_latents_for_sp(self, latents, batch=None):
        from nndeploy.e_dit.distributed.model_parallel import sequence_model_parallel_all_gather
        return sequence_model_parallel_all_gather(latents, dim=2)

# Migrated from SGLang multimodal_gen; see ltx2_migration.json and LICENSE.SGLang.
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from nndeploy.e_dit.config.model_base import ArchConfig, ModelConfig
from nndeploy.e_dit.config.quantization import QuantizationConfig
from nndeploy.e_dit.platforms import AttentionBackendEnum


@dataclass
class DiTArchConfig(ArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)

    # convert weights name from HF-format to SGLang-dit-format
    param_names_mapping: dict = field(default_factory=dict)

    # convert weights name from misc-format to HF-format
    # usually applicable if the LoRA is trained with official repo implementation
    lora_param_names_mapping: dict = field(default_factory=dict)

    # Reverse mapping for saving checkpoints: custom -> hf
    reverse_param_names_mapping: dict = field(default_factory=dict)
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
        }
    )

    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0
    exclude_lora_layers: list[str] = field(default_factory=list)
    boundary_ratio: float | None = None

    def __post_init__(self) -> None:
        if not self._compile_conditions:
            self._compile_conditions = self._fsdp_shard_conditions.copy()


@dataclass
class DiTConfig(ModelConfig):
    arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)

    # sglang-diffusion DiT-specific parameters
    prefix: str = ""
    quant_config: QuantizationConfig | None = None


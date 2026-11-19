"""Common diffusion transformer interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from nndeploy.e_dit.layers.attention import AttentionBackendEnum


class BaseDiT(nn.Module, ABC):
    _fsdp_shard_conditions: list[Any] = []
    _compile_conditions: list[Any] = []
    param_names_mapping: dict[str, str | tuple[str, int, int]] = {}
    reverse_param_names_mapping: dict[str, tuple[str, Any, Any]] = {}
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0
    _supported_attention_backends: set[AttentionBackendEnum] = {
        AttentionBackendEnum.TORCH_SDPA
    }

    def __init__(
        self,
        config: Any,
        hf_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.config = config
        self.hf_config = hf_config or {}

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    def __post_init__(self) -> None:
        return None

    def post_load_weights(self) -> None:
        return None

    @property
    def supported_attention_backends(self) -> set[AttentionBackendEnum]:
        return self._supported_attention_backends

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


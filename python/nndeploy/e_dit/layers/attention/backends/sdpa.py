"""PyTorch SDPA attention backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nndeploy.e_dit.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionBackendEnum,
    AttentionCapability,
    AttentionImpl,
    AttentionMetadata,
    AttentionSupport,
    validate_bshd_qkv,
)


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def probe(capability: AttentionCapability) -> AttentionSupport:
        del capability
        return AttentionSupport(supported=True)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_SDPA

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl


class SDPAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        del num_heads, head_size, num_kv_heads, prefix
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        validate_bshd_qkv(query, key, value)
        if attn_metadata.attn_mask is not None and self.causal:
            raise ValueError("attn_mask cannot be combined with is_causal=True")

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_metadata.attn_mask,
            dropout_p=self.dropout,
            is_causal=self.causal,
            scale=self.softmax_scale,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return output.transpose(1, 2)

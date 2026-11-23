"""Single-device attention layers."""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from nndeploy.e_dit.layers.attention.backends.attention_backend import (
    AttentionBackendEnum,
    AttentionBackendUnavailableError,
    AttentionImpl,
    AttentionMetadata,
    wrap_attention_impl_forward,
)
from nndeploy.e_dit.layers.attention.backends.sdpa import SDPABackend


class _BaseAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        backend: str = "sdpa",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends=None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        super().__init__()
        device = torch.device(device)
        if backend not in {"sdpa", AttentionBackendEnum.TORCH_SDPA.value}:
            raise AttentionBackendUnavailableError(
                backend,
                f"single-device attention layer only supports sdpa on {device}",
            )
        if (
            supported_attention_backends is not None
            and AttentionBackendEnum.TORCH_SDPA not in supported_attention_backends
        ):
            raise AttentionBackendUnavailableError(
                backend,
                "torch_sdpa is not in the model's supported backend set",
            )
        if softmax_scale is None:
            softmax_scale = head_size**-0.5
        if num_kv_heads is None:
            num_kv_heads = num_heads
        del dtype
        attn_backend = SDPABackend
        impl_cls: Type[AttentionImpl] = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.impl",
            **extra_impl_args,
        )
        wrap_attention_impl_forward(self.attn_impl)

    def _metadata(self, attn_mask: torch.Tensor | None = None) -> AttentionMetadata:
        return AttentionMetadata(current_timestep=0, attn_mask=attn_mask)


class UlyssesAttention(_BaseAttention):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = self.attn_impl.forward(q, k, v, self._metadata())
        replicated_output = None
        if replicated_q is not None:
            if replicated_k is None or replicated_v is None:
                raise ValueError(
                    "replicated_k and replicated_v are required when replicated_q is provided"
                )
            replicated_output = self.attn_impl.forward(
                replicated_q, replicated_k, replicated_v, self._metadata()
            )
        return output, replicated_output


class UlyssesAttention_VSA(UlyssesAttention):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        gate_compress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del replicated_q, replicated_k, replicated_v, gate_compress
        return self.attn_impl.forward(q, k, v, self._metadata())


class LocalAttention(_BaseAttention):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_mask is not None:
            floating = torch.is_floating_point(attn_mask)
            attn_mask = attn_mask.to(device=q.device, dtype=q.dtype)
            if attn_mask.ndim == 2:
                attn_mask = attn_mask[:, None, None, :]
            elif attn_mask.ndim == 3:
                attn_mask = attn_mask[:, None, :, :]
            if not floating:
                attn_mask = (attn_mask - 1.0) * torch.finfo(q.dtype).max
        return self.attn_impl.forward(q, k, v, self._metadata(attn_mask))


class USPAttention(_BaseAttention):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
    ):
        output = self.attn_impl.forward(q, k, v, self._metadata())
        if replicated_q is None:
            return output
        if replicated_k is None or replicated_v is None:
            raise ValueError(
                "replicated_k and replicated_v are required when replicated_q is provided"
            )
        replicated_output = self.attn_impl.forward(
            replicated_q, replicated_k, replicated_v, self._metadata()
        )
        return output, replicated_output

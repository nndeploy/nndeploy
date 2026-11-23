"""FlashAttention backend with lazy optional dependency loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch

from nndeploy.e_dit.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionBackendEnum,
    AttentionBackendUnavailableError,
    AttentionCapability,
    AttentionImpl,
    AttentionMetadata,
    AttentionSupport,
    validate_bshd_qkv,
)

FlashAttentionFunc = Callable[..., torch.Tensor]


def _load_flash_attn_func() -> tuple[FlashAttentionFunc | None, str | None]:
    try:
        module = importlib.import_module("flash_attn")
    except (ImportError, OSError) as exc:
        return None, f"flash_attn_func is unavailable: {exc}"

    flash_attn_func = getattr(module, "flash_attn_func", None)
    if not callable(flash_attn_func):
        return None, "flash_attn_func is unavailable from the flash_attn package"
    return flash_attn_func, None


class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def probe(capability: AttentionCapability) -> AttentionSupport:
        if capability.device.type != "cuda":
            return AttentionSupport(
                supported=False,
                reason="flash_attn requires a CUDA device",
            )
        if capability.dtype not in (torch.float16, torch.bfloat16):
            return AttentionSupport(
                supported=False,
                reason=(
                    "flash_attn requires dtype torch.float16 or "
                    f"torch.bfloat16, got {capability.dtype}"
                ),
            )
        if (
            capability.head_size <= 0
            or capability.head_size % 8 != 0
            or capability.head_size > 256
        ):
            return AttentionSupport(
                supported=False,
                reason=(
                    "flash_attn head_size must be positive, divisible by 8, "
                    "and at most 256"
                ),
            )
        if capability.has_attn_mask:
            return AttentionSupport(
                supported=False,
                reason="flash_attn does not support an arbitrary attention mask",
            )

        try:
            major, minor = torch.cuda.get_device_capability(capability.device)
        except (AssertionError, RuntimeError, ValueError) as exc:
            return AttentionSupport(
                supported=False,
                reason=f"flash_attn requires sm80 or newer: {exc}",
            )
        if (major, minor) < (8, 0):
            return AttentionSupport(
                supported=False,
                reason=f"flash_attn requires sm80 or newer, got sm{major}{minor}",
            )

        _, reason = _load_flash_attn_func()
        if reason is not None:
            return AttentionSupport(supported=False, reason=reason)
        return AttentionSupport(supported=True)

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FLASH_ATTN

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl


class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        del num_heads, head_size, num_kv_heads, prefix
        flash_attn_func, reason = _load_flash_attn_func()
        if flash_attn_func is None:
            raise AttentionBackendUnavailableError(
                AttentionBackendEnum.FLASH_ATTN,
                reason or "flash_attn_func is unavailable",
            )
        self.flash_attn_func = flash_attn_func
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
        if attn_metadata.attn_mask is not None:
            raise AttentionBackendUnavailableError(
                AttentionBackendEnum.FLASH_ATTN,
                "arbitrary attention mask is unsupported",
            )

        return self.flash_attn_func(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            dropout_p=self.dropout,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )

"""SageAttention backend with lazy public dispatcher loading."""

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

SageAttentionDispatcher = Callable[..., torch.Tensor]

_SUPPORTED_ARCHITECTURES = {
    (8, 0),
    (8, 6),
    (8, 9),
    (9, 0),
    (12, 0),
}
_SUPPORTED_HEAD_SIZES = {64, 128}


def _load_sageattn() -> tuple[SageAttentionDispatcher | None, str | None]:
    try:
        module = importlib.import_module("sageattention")
    except (ImportError, OSError) as exc:
        return None, f"sageattention.sageattn is unavailable: {exc}"

    sageattn = getattr(module, "sageattn", None)
    if not callable(sageattn):
        return None, "sageattention.sageattn public dispatcher is unavailable"
    return sageattn, None


class SageAttentionBackend(AttentionBackend):
    @staticmethod
    def probe(capability: AttentionCapability) -> AttentionSupport:
        if capability.device.type != "cuda":
            return AttentionSupport(
                supported=False,
                reason="sage_attn requires a CUDA device",
            )
        if capability.dtype not in (torch.float16, torch.bfloat16):
            return AttentionSupport(
                supported=False,
                reason=(
                    "sage_attn requires dtype torch.float16 or "
                    f"torch.bfloat16, got {capability.dtype}"
                ),
            )
        if capability.head_size not in _SUPPORTED_HEAD_SIZES:
            return AttentionSupport(
                supported=False,
                reason="sage_attn head_size must be 64 or 128",
            )
        if capability.has_attn_mask:
            return AttentionSupport(
                supported=False,
                reason="sage_attn does not support an arbitrary attention mask",
            )

        try:
            architecture = torch.cuda.get_device_capability(capability.device)
        except (AssertionError, RuntimeError, ValueError) as exc:
            return AttentionSupport(
                supported=False,
                reason=f"sage_attn GPU architecture query failed: {exc}",
            )
        if architecture not in _SUPPORTED_ARCHITECTURES:
            major, minor = architecture
            return AttentionSupport(
                supported=False,
                reason=(
                    "sage_attn public dispatcher does not support GPU "
                    f"architecture sm{major}{minor}"
                ),
            )

        _, reason = _load_sageattn()
        if reason is not None:
            return AttentionSupport(supported=False, reason=reason)
        return AttentionSupport(supported=True)

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SAGE_ATTN

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl):
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
        del num_heads, head_size, num_kv_heads, prefix, extra_impl_args
        sageattn, reason = _load_sageattn()
        if sageattn is None:
            raise AttentionBackendUnavailableError(
                AttentionBackendEnum.SAGE_ATTN,
                reason or "sageattention.sageattn is unavailable",
            )
        self.sageattn = sageattn
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        key_already_smoothed: bool = False,
    ) -> torch.Tensor:
        validate_bshd_qkv(query, key, value)
        if attn_metadata.attn_mask is not None:
            raise AttentionBackendUnavailableError(
                AttentionBackendEnum.SAGE_ATTN,
                "arbitrary attention mask is unsupported",
            )

        return self.sageattn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            tensor_layout="NHD",
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
            return_lse=False,
            smooth_k=not key_already_smoothed,
        )

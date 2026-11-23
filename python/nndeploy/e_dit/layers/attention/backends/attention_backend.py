"""Minimal attention backend interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

import torch


class AttentionBackendEnum(str, Enum):
    TORCH_SDPA = "torch_sdpa"
    FLASH_ATTN = "flash_attn"
    SAGE_ATTN = "sage_attn"
    SAGE_FP8 = "sage_fp8"


@dataclass(frozen=True)
class AttentionCapability:
    device: torch.device
    dtype: torch.dtype
    head_size: int
    causal: bool = False
    has_attn_mask: bool = False


@dataclass(frozen=True)
class AttentionSupport:
    supported: bool
    reason: str | None = None


@dataclass(frozen=True)
class AttentionSelection:
    requested: str
    selected: AttentionBackendEnum
    fallback_reasons: tuple[str, ...] = ()
    probed_backends: tuple[AttentionBackendEnum, ...] = ()


class AttentionBackendUnavailableError(RuntimeError):
    def __init__(
        self,
        requested: str | AttentionBackendEnum,
        reason: str,
    ) -> None:
        requested_name = (
            requested.value
            if isinstance(requested, AttentionBackendEnum)
            else requested
        )
        self.requested = requested_name
        self.reason = reason
        super().__init__(
            f"Attention backend {requested_name!r} is unavailable: {reason}"
        )


def validate_bshd_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must use BSHD rank-4 layout, got rank {tensor.ndim}"
            )

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Q/K/V batch dimensions must match")
    if query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
        raise ValueError("Q/K/V head_dim dimensions must match")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Q/K/V dtype values must match")
    if query.device != key.device or query.device != value.device:
        raise ValueError("Q/K/V device values must match")
    if key.shape[1] != value.shape[1]:
        raise ValueError("K/V sequence dimensions must match")
    if key.shape[2] != value.shape[2]:
        raise ValueError("K/V heads must match")

    query_heads = query.shape[2]
    kv_heads = key.shape[2]
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads != 0:
        raise ValueError(
            "Q heads must equal K/V heads or be a positive integer multiple"
        )


class AttentionBackend(ABC):
    accept_output_buffer: bool = False

    @staticmethod
    @abstractmethod
    def probe(capability: AttentionCapability) -> AttentionSupport:
        """Report whether this backend supports a capability without side effects."""
        del capability
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_enum() -> AttentionBackendEnum:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"] | None:
        return None


@dataclass
class AttentionMetadata:
    current_timestep: int = 0
    attn_mask: torch.Tensor | None = None

    def asdict_zerocopy(self, skip_fields: set[str] | None = None) -> dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        result = {}
        if "current_timestep" not in skip_fields:
            result["current_timestep"] = self.current_timestep
        if "attn_mask" not in skip_fields:
            result["attn_mask"] = self.attn_mask
        return result


T = TypeVar("T", bound=AttentionMetadata)


class AttentionMetadataBuilder(ABC, Generic[T]):
    def prepare(self) -> None:
        return None

    def build(self, **kwargs: dict[str, Any]) -> AttentionMetadata:
        return AttentionMetadata(**kwargs)


class AttentionImpl(ABC, Generic[T]):
    @abstractmethod
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
        raise NotImplementedError

    def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: T) -> torch.Tensor:
        del attn_metadata
        return qkv

    def postprocess_output(
        self, output: torch.Tensor, attn_metadata: T
    ) -> torch.Tensor:
        del attn_metadata
        return output

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError


def wrap_attention_impl_forward(attn_impl: AttentionImpl) -> AttentionImpl:
    return attn_impl

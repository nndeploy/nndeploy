"""Common request contract for approximate Transformer caches."""

from __future__ import annotations

from enum import Enum


class TransformerCacheMode(str, Enum):
    OFF = "off"
    TEACACHE = "teacache"
    CACHE_DIT = "cache_dit"


def resolve_transformer_cache_mode(value: object) -> TransformerCacheMode:
    if isinstance(value, TransformerCacheMode):
        return value
    if not isinstance(value, str):
        raise TypeError(
            "transformer_cache_mode must be a string or TransformerCacheMode"
        )
    try:
        return TransformerCacheMode(value.strip().lower())
    except ValueError as error:
        choices = ", ".join(mode.value for mode in TransformerCacheMode)
        raise ValueError(
            f"transformer_cache_mode must be one of {{{choices}}}, got {value!r}"
        ) from error


def validate_transformer_cache_request(
    *,
    mode: TransformerCacheMode | str,
    enable_torch_compile: bool,
    cache_dit_implemented: bool = True,
) -> TransformerCacheMode:
    resolved = resolve_transformer_cache_mode(mode)
    if resolved is not TransformerCacheMode.OFF and enable_torch_compile:
        raise ValueError(
            "approximate Transformer caches are incompatible with whole-"
            "Transformer torch.compile"
        )
    if resolved is TransformerCacheMode.CACHE_DIT and not cache_dit_implemented:
        raise NotImplementedError(
            "Cache-DiT is unavailable in this runtime build"
        )
    return resolved


__all__ = (
    "TransformerCacheMode",
    "resolve_transformer_cache_mode",
    "validate_transformer_cache_request",
)

"""Capability-aware attention backend selection."""

from __future__ import annotations

from nndeploy.e_dit.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionBackendEnum,
    AttentionBackendUnavailableError,
    AttentionCapability,
    AttentionSelection,
    AttentionSupport,
)
from nndeploy.e_dit.layers.attention.backends.flash_attn import FlashAttentionBackend
from nndeploy.e_dit.layers.attention.backends.sage_attn import SageAttentionBackend
from nndeploy.e_dit.layers.attention.backends.sage_fp8 import SageFP8AttentionBackend
from nndeploy.e_dit.layers.attention.backends.sdpa import SDPABackend

_BACKEND_REGISTRY: dict[AttentionBackendEnum, type[AttentionBackend]] = {
    AttentionBackendEnum.TORCH_SDPA: SDPABackend,
    AttentionBackendEnum.FLASH_ATTN: FlashAttentionBackend,
    AttentionBackendEnum.SAGE_ATTN: SageAttentionBackend,
    AttentionBackendEnum.SAGE_FP8: SageFP8AttentionBackend,
}

_BACKEND_NAMES = {
    "sdpa": AttentionBackendEnum.TORCH_SDPA,
    **{backend.value: backend for backend in AttentionBackendEnum},
}

_AUTO_BACKENDS = (
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.TORCH_SDPA,
)


def backend_name_to_enum(backend_name: str) -> AttentionBackendEnum | None:
    return _BACKEND_NAMES.get(backend_name.lower())


def probe_attention_backend(
    backend: AttentionBackendEnum,
    capability: AttentionCapability,
) -> AttentionSupport:
    backend_cls = _BACKEND_REGISTRY.get(backend)
    if backend_cls is None:
        return AttentionSupport(
            supported=False,
            reason=f"{backend.value} backend implementation is not registered",
        )
    return backend_cls.probe(capability)


def _unsupported_reason(
    backend: AttentionBackendEnum,
    support: AttentionSupport,
) -> str:
    if support.reason is not None:
        return support.reason
    return f"{backend.value} capability probe reported unsupported"


def resolve_attention_backend(
    requested: str,
    capability: AttentionCapability,
    *,
    supported: set[AttentionBackendEnum] | None = None,
    auto_backends: tuple[AttentionBackendEnum, ...] | None = None,
) -> AttentionSelection:
    allowed = set(AttentionBackendEnum) if supported is None else set(supported)
    normalized = requested.lower()
    candidates = _AUTO_BACKENDS if auto_backends is None else auto_backends

    if normalized != "auto":
        backend = backend_name_to_enum(normalized)
        if backend is None:
            raise ValueError(f"Unsupported attention backend: {requested}")
        if backend not in allowed:
            raise AttentionBackendUnavailableError(
                requested,
                f"{backend.value} is not in the model's supported backend set",
            )
        if backend is AttentionBackendEnum.TORCH_SDPA:
            return AttentionSelection(requested=requested, selected=backend)

        support = probe_attention_backend(backend, capability)
        if not support.supported:
            raise AttentionBackendUnavailableError(
                requested,
                _unsupported_reason(backend, support),
            )
        return AttentionSelection(
            requested=requested,
            selected=backend,
            probed_backends=(backend,),
        )

    fallback_reasons: list[str] = []
    probed_backends: list[AttentionBackendEnum] = []
    for backend in candidates:
        if backend not in allowed:
            continue
        if backend is AttentionBackendEnum.TORCH_SDPA:
            return AttentionSelection(
                requested=requested,
                selected=backend,
                fallback_reasons=tuple(fallback_reasons),
                probed_backends=tuple(probed_backends),
            )

        support = probe_attention_backend(backend, capability)
        probed_backends.append(backend)
        if support.supported:
            return AttentionSelection(
                requested=requested,
                selected=backend,
                fallback_reasons=tuple(fallback_reasons),
                probed_backends=tuple(probed_backends),
            )
        fallback_reasons.append(_unsupported_reason(backend, support))

    if fallback_reasons:
        reason = "; ".join(fallback_reasons)
    else:
        candidate_names = ", ".join(backend.value for backend in candidates)
        reason = f"none of the auto candidates ({candidate_names}) are in the supported backend set"
    raise AttentionBackendUnavailableError(requested, reason)

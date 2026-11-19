"""EraserDiT adapter for the local attention backends.

Self-attention routes through the shared capability-aware selector
(``auto | sdpa | flash_attn | sage_attn | sage_fp8``) and, when the operator
fusion decision enables it, through the fused ``qk_rmsnorm_rope`` kernel.
Cross-attention always stays on PyTorch SDPA: flash/sage reject an attention
mask, and the ``qk_rmsnorm_rope`` kernel is defined for a single full-width Q/K
pair, which is exactly what self-attention has.

With ``attention_backend="sdpa"`` and fusion disabled every value is produced by
the same ``F.scaled_dot_product_attention`` call as the model implementation.
"""

from __future__ import annotations

from typing import Optional

import torch
from diffusers.models.attention_processor import Attention

from nndeploy.e_dit.layers.attention.backends.attention_backend import (
    AttentionBackendEnum,
    AttentionCapability,
    AttentionImpl,
    AttentionMetadata,
    AttentionSelection,
)
from nndeploy.e_dit.layers.attention.backends.flash_attn import FlashAttentionBackend
from nndeploy.e_dit.layers.attention.backends.sage_attn import SageAttentionBackend
from nndeploy.e_dit.layers.attention.backends.sage_fp8 import SageFP8AttentionBackend
from nndeploy.e_dit.layers.attention.backends.sdpa import SDPABackend
from nndeploy.e_dit.layers.attention.selector import resolve_attention_backend
from nndeploy.e_dit.layers.operator_fusion.config import QK_RMSNORM_ROPE_OP, QK_RMSNORM_ROPE_FAST_OP, QK_RMSNORM_ROPE_NATIVE_OP
from nndeploy.e_dit.algorithms.eraserdit.optimizations.qk_rmsnorm_rope import apply_fused_qk_rmsnorm_rope
from nndeploy.e_dit.algorithms.eraserdit.optimizations.fusion_policy import get_operator_fusion_decision
from nndeploy.e_dit.config.server_args import get_global_server_args
from nndeploy.e_dit.utils.logging_utils import init_logger

logger = init_logger(__name__)

_BACKEND_CLASSES = {
    AttentionBackendEnum.TORCH_SDPA: SDPABackend,
    AttentionBackendEnum.FLASH_ATTN: FlashAttentionBackend,
    AttentionBackendEnum.SAGE_ATTN: SageAttentionBackend,
    AttentionBackendEnum.SAGE_FP8: SageFP8AttentionBackend,
}

# Preference order: Sage, then FlashAttention, then SDPA.
ERASERDIT_AUTO_BACKENDS = (
    AttentionBackendEnum.SAGE_ATTN,
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.TORCH_SDPA,
)

_ALLOWED_BACKENDS = {"auto", "sdpa", "flash_attn", "sage_attn", "sage_fp8"}


def normalize_attention_backend(value: str | None) -> str:
    backend = str(value or "sdpa").strip().lower()
    if backend not in _ALLOWED_BACKENDS:
        raise ValueError(
            f"Unsupported attention backend: {value!r}; allowed={sorted(_ALLOWED_BACKENDS)}"
        )
    return backend


def apply_rotary_emb(x: torch.Tensor, freqs) -> torch.Tensor:
    """RoPE exactly as the vendored EraserDiT transformer defines it.

    This is *not* ``diffusers.models.embeddings.apply_rotary_emb``: the vendored
    version pairs the feature dim directly (``[B, S, D]`` -> ``[B, S, D // 2, 2]``)
    rather than assuming a ``[B, H, S, D]`` layout.
    """
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)  # [B, S, H, D // 2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class EraserDiTAttentionProcessor:
    """Project LTX Q/K/V and route them to the selected attention backend."""

    def __init__(self, attention_backend: str | None = None) -> None:
        server_args = get_global_server_args()
        if attention_backend is None:
            attention_backend = getattr(server_args, "attention_backend", "sdpa")
        self.attention_backend = normalize_attention_backend(attention_backend)
        self.sage_fp8_options = dict(
            pv_accum_dtype=getattr(server_args, "sage_fp8_accum_dtype", "fp32+fp32"),
            qk_quant_gran=getattr(server_args, "sage_fp8_qk_quant_gran", "per_thread"))
        self.operator_fusion_decision = get_operator_fusion_decision(server_args)
        self._selection_cache: dict[tuple, AttentionSelection] = {}
        self._impl_cache: dict[tuple, AttentionImpl] = {}
        self._latest_selection: AttentionSelection | None = None

    # ── backend resolution ──────────────────────────────────────────────
    @staticmethod
    def _selection_key(requested: str, capability: AttentionCapability) -> tuple:
        return (
            requested,
            capability.device.type,
            capability.device.index,
            capability.dtype,
            capability.head_size,
            capability.causal,
            capability.has_attn_mask,
        )

    def _resolve_self_attention(
        self, query: torch.Tensor, *, has_attn_mask: bool
    ) -> tuple[AttentionSelection, AttentionImpl]:
        capability = AttentionCapability(
            device=query.device,
            dtype=query.dtype,
            head_size=int(query.shape[-1]),
            causal=False,
            has_attn_mask=has_attn_mask,
        )
        selection = self._selection_cache.get(
            self._selection_key(self.attention_backend, capability)
        )
        if selection is None:
            selection = resolve_attention_backend(
                self.attention_backend,
                capability,
                auto_backends=ERASERDIT_AUTO_BACKENDS,
            )
            self._selection_cache[
                self._selection_key(self.attention_backend, capability)
            ] = selection
        self._latest_selection = selection

        impl = self._impl_cache.get(selection.selected)
        if impl is None:
            backend_cls = _BACKEND_CLASSES[selection.selected]
            impl = backend_cls.get_impl_cls()(
                num_heads=capability.head_size,  # unused by the backends
                head_size=capability.head_size,
                softmax_scale=capability.head_size**-0.5,
                causal=False,
                num_kv_heads=capability.head_size,
                prefix="eraserdit_attention.impl",
                dropout_p=0.0,
                **(self.sage_fp8_options if selection.selected is AttentionBackendEnum.SAGE_FP8 else {}),
            )
            self._impl_cache[selection.selected] = impl
        return selection, impl

    @property
    def effective_self_attention_backend(self) -> str | None:
        return (
            self._latest_selection.selected.value
            if self._latest_selection is not None
            else None
        )

    def preflight_self_attention_backend(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        head_size: int = 64,
        num_heads: int = 32,
    ) -> dict[str, object]:
        """Resolve the self-attention backend without running a forward."""
        capability = AttentionCapability(
            device=device,
            dtype=dtype,
            head_size=head_size,
            causal=False,
            has_attn_mask=False,
        )
        selection = resolve_attention_backend(
            self.attention_backend,
            capability,
            auto_backends=ERASERDIT_AUTO_BACKENDS,
        )
        self._latest_selection = selection
        self._selection_cache[self._selection_key(self.attention_backend, capability)] = selection
        backend_cls = _BACKEND_CLASSES[selection.selected]
        impl = backend_cls.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=head_size**-0.5,
            causal=False,
            num_kv_heads=num_heads,
            prefix="eraserdit_attention.preflight",
            dropout_p=0.0,
            **(self.sage_fp8_options if selection.selected is AttentionBackendEnum.SAGE_FP8 else {}),
        )
        self._impl_cache[selection.selected] = impl
        return self.attention_backend_report(impl=impl)

    def attention_backend_report(self, *, impl: AttentionImpl | None = None) -> dict:
        selection = self._latest_selection
        report: dict[str, object] = {
            "requested": self.attention_backend,
            "effective": selection.selected.value if selection else None,
            "cross_attention_backend": "torch_sdpa",
            "fallback_reasons": list(selection.fallback_reasons) if selection else [],
            "fallback_count": len(selection.fallback_reasons) if selection else 0,
        }
        candidates = [impl] if impl is not None else list(self._impl_cache.values())
        strict = [
            candidate.report()
            for candidate in candidates
            if callable(getattr(candidate, "report", None))
        ]
        if strict:
            report.update(strict[0])
        return report

    # ── attention paths ─────────────────────────────────────────────────
    @staticmethod
    def _project_output(
        attn: Attention,
        hidden_states: torch.Tensor,
        dtype: torch.dtype,
        *,
        flatten_heads: bool = True,
    ) -> torch.Tensor:
        if flatten_heads:
            hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _to_bshd(
        attn: Attention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unflattened = (
            query.unflatten(2, (attn.heads, -1)),
            key.unflatten(2, (attn.heads, -1)),
            value.unflatten(2, (attn.heads, -1)),
        )
        return unflattened

    def self_attn(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if {QK_RMSNORM_ROPE_OP, QK_RMSNORM_ROPE_FAST_OP, QK_RMSNORM_ROPE_NATIVE_OP}.intersection(
            self.operator_fusion_decision.effective_ops
        ):

            def reference_qk_norm_rope() -> tuple[torch.Tensor, torch.Tensor]:
                reference_query = attn.norm_q(query)
                reference_key = attn.norm_k(key)
                if image_rotary_emb is not None:
                    reference_query = apply_rotary_emb(reference_query, image_rotary_emb)
                    reference_key = apply_rotary_emb(reference_key, image_rotary_emb)
                return reference_query, reference_key

            query, key = apply_fused_qk_rmsnorm_rope(
                query,
                key,
                attn.norm_q,
                attn.norm_k,
                image_rotary_emb,
                decision=self.operator_fusion_decision,
                reference=reference_qk_norm_rope,
            )
        else:
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        query, key, value = self._to_bshd(attn, query, key, value)
        _selection, impl = self._resolve_self_attention(query, has_attn_mask=False)
        sp = getattr(self, "sequence_parallel", None)
        if sp is None:
            output = impl.forward(query, key, value, AttentionMetadata(attn_mask=None))
        else:
            output = sp.attention(query, key, value, impl, AttentionMetadata(attn_mask=None))
        return self._project_output(attn, output, query.dtype)

    def cross_attn(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        text_cache=None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        if text_cache is None:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key, value = text_cache.key_value(attn, encoder_hidden_states)
        # The reference processor applies ``norm_q``/``norm_k`` unconditionally,
        # so cross-attention is normalised too.  Only RoPE is self-attention
        # specific (``attn2`` is always called with ``image_rotary_emb=None``).
        query = attn.norm_q(query)
        if text_cache is None:
            key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        output = output.transpose(1, 2).flatten(2, 3)
        return self._project_output(attn, output, query.dtype, flatten_heads=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        text_cache=None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            return self.self_attn(attn, hidden_states, image_rotary_emb)
        return self.cross_attn(
            attn, hidden_states, encoder_hidden_states, attention_mask, text_cache=text_cache
        )


__all__ = [
    "ERASERDIT_AUTO_BACKENDS",
    "apply_rotary_emb",
    "EraserDiTAttentionProcessor",
    "normalize_attention_backend",
]

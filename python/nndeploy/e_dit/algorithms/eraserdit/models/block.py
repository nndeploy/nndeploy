"""EraserDiT block with optional AdaLN and gated-residual fusion.

The block follows ``LTXVideoTransformerBlock.forward`` with normalization,
AdaLN modulation and gated residuals routed through dispatchers. Disabled
fusion evaluates the original expressions; ``rmsnorm_adaln_fast`` explicitly
allows a different FP32 reduction order while preserving BF16 boundaries.

The ``rmsnorm_adaln`` capability contract (hidden width 2048, eps 1e-6, bf16,
batch 1, non-affine norm, modulation of shape ``(1, 1, 2048)``) is met here: the
denoising stage issues two batch-1 forwards for classifier-free guidance and
``temb`` carries a single timestep, so ``scale_msa``/``shift_msa`` broadcast to
``(1, 1, 2048)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from nndeploy.e_dit.layers.operator_fusion.config import RMSNORM_ADALN_FAST_OP, RMSNORM_ADALN_NATIVE_OP
from nndeploy.e_dit.algorithms.eraserdit.optimizations.rmsnorm_adaln_fast import apply_fused_rmsnorm_adaln_fast
from nndeploy.e_dit.layers.operator_fusion.registry import OperatorFusionDecision
from nndeploy.e_dit.algorithms.eraserdit.optimizations.rmsnorm_adaln import apply_fused_rmsnorm_adaln, apply_fused_rmsnorm_adaln_native
from nndeploy.e_dit.layers.operator_fusion.gated_residual import apply_fused_gated_residual

__all__ = ["forward_eraserdit_block"]


def _normalize_modulate(hidden, scale, shift, norm, decision):
    if RMSNORM_ADALN_NATIVE_OP in decision.effective_ops:
        return apply_fused_rmsnorm_adaln_native(hidden, scale, shift, norm, decision=decision)
    if RMSNORM_ADALN_FAST_OP in decision.effective_ops:
        return apply_fused_rmsnorm_adaln_fast(hidden, scale, shift, norm, decision=decision)
    normalized = norm(hidden)
    return apply_fused_rmsnorm_adaln(
        normalized, scale, shift, norm, decision=decision,
        reference=lambda: normalized * (1 + scale) + shift,
    )


def forward_eraserdit_block(
    block,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    *,
    decision: OperatorFusionDecision,
    text_cache=None,
) -> torch.Tensor:
    batch_size = hidden_states.size(0)

    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None] + temb.reshape(
        batch_size, temb.size(1), num_ada_params, -1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(
        dim=2
    )
    norm_hidden_states = _normalize_modulate(
        hidden_states, scale_msa, shift_msa, block.norm1, decision,
    )

    attn_hidden_states = block.attn1(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=None,
        image_rotary_emb=image_rotary_emb,
    )
    hidden_states = apply_fused_gated_residual(
        hidden_states, attn_hidden_states, gate_msa, decision=decision,
    )

    attn_hidden_states = block.attn2(
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        image_rotary_emb=None,
        attention_mask=encoder_attention_mask,
        text_cache=text_cache,
    )
    hidden_states = hidden_states + attn_hidden_states

    norm_hidden_states = _normalize_modulate(
        hidden_states, scale_mlp, shift_mlp, block.norm2, decision,
    )

    ff_output = block.ff(norm_hidden_states)
    hidden_states = apply_fused_gated_residual(
        hidden_states, ff_output, gate_mlp, decision=decision,
    )
    return hidden_states

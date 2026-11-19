# Adapted from SGLang multimodal_gen tests (Apache-2.0).
"""Regression coverage for reference RMSNorm, output modulation and sigma math."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nndeploy.e_dit.layers.layernorm import RMSNorm
from nndeploy.e_dit.algorithms.wan.models import wan_videoedit
from nndeploy.e_dit.algorithms.wan.models.wanvideo import WanTransformer3DModel
from nndeploy.e_dit.models.schedulers.videoedit_flow_match import (
    VideoEditFlowMatchScheduler,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_videoedit_all_qk_norms_use_torch_without_strict(dtype):
    def fake_base_init(self, **kwargs):
        torch.nn.Module.__init__(self)
        block = torch.nn.Module()
        block.norm_q = RMSNorm(32)
        block.norm_k = RMSNorm(32)
        self.blocks = torch.nn.ModuleList([block])

    def fake_cross_attention(*args, **kwargs):
        attention = torch.nn.Module()
        for name in ("norm_q", "norm_k", "norm_added_k"):
            setattr(attention, name, RMSNorm(32))
        return attention

    config = SimpleNamespace(
        hidden_size=32, num_attention_heads=4, qk_norm=True, eps=1e-6
    )
    with (
        patch.object(WanTransformer3DModel, "__init__", fake_base_init),
        patch.object(wan_videoedit, "WanVideoEditCrossAttention", fake_cross_attention),
    ):
        model = wan_videoedit.WanVideoEditTransformer3DModel(config, {})
    model.to(dtype=dtype)
    assert model.strict_videoedit_math is False
    block = model.blocks[0]
    assert block.strict_videoedit_math is False
    x = torch.randn(2, 7, 32, generator=torch.Generator().manual_seed(42)).to(dtype)
    names = ["blocks.0.norm_q.weight", "blocks.0.norm_k.weight"]
    names += [
        f"blocks.0.attn2.{name}.weight" for name in ("norm_q", "norm_k", "norm_added_k")
    ]
    assert list(model.state_dict()) == names
    for attention in (block, block.attn2):
        for name in ("norm_q", "norm_k", "norm_added_k"):
            norm = getattr(attention, name, None)
            if norm is not None:
                assert isinstance(norm, torch.nn.RMSNorm)
                expected = torch.nn.functional.rms_norm(x, (32,), norm.weight, norm.eps)
                assert torch.equal(norm(x), expected)


@pytest.mark.parametrize("per_token", [False, True])
def test_output_modulation_preserves_fp32_before_bf16_writeback(per_token):
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(1, 7, 32, generator=generator).bfloat16()
    temb = (
        torch.randn(1, 7, 32, generator=generator).bfloat16()
        if per_token
        else torch.randn(1, 32, generator=generator).bfloat16()
    )
    table = torch.randn(1, 2, 32, generator=generator).bfloat16()
    norm = torch.nn.LayerNorm(32, eps=1e-6, elementwise_affine=False)
    model = SimpleNamespace(
        scale_shift_table=table,
        norm_out=SimpleNamespace(norm=norm),
        videoedit_fp32_output_modulation=True,
        strict_videoedit_math=False,
    )
    if per_token:
        shift, scale = (table.float().unsqueeze(0) + temb.float().unsqueeze(2)).unbind(
            2
        )
    else:
        shift, scale = (table.float() + temb.float().unsqueeze(1)).chunk(2, dim=1)
    expected = (norm(hidden.float()) * (1 + scale) + shift).bfloat16()
    actual = WanTransformer3DModel._modulate_output(model, hidden, temb)
    assert torch.equal(actual, expected)
    assert model.scale_shift_table.dtype == torch.bfloat16


def test_other_wan_models_keep_existing_output_norm_path():
    hidden = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
    temb = torch.ones(1, 8, dtype=torch.bfloat16)
    table = torch.ones(1, 2, 8, dtype=torch.bfloat16)
    sentinel = object()
    norm_out = Mock(return_value=sentinel)
    model = SimpleNamespace(scale_shift_table=table, norm_out=norm_out)
    assert WanTransformer3DModel._modulate_output(model, hidden, temb) is sentinel
    norm_hidden, shift, scale = norm_out.call_args.args
    assert norm_hidden is hidden
    assert shift.dtype == scale.dtype == torch.bfloat16
    assert torch.equal(shift, torch.full((1, 1, 8), 2, dtype=torch.bfloat16))


def test_scheduler_sigmas_stay_on_cpu_with_non_cpu_default_device():
    with torch.device("meta"):
        scheduler = VideoEditFlowMatchScheduler()
    assert scheduler.sigmas.device.type == "cpu"


@pytest.mark.parametrize(
    "inverse,reverse", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("to_final", [False, True])
def test_scheduler_matches_reference_cpu_scalars(inverse, reverse, to_final):
    scheduler = VideoEditFlowMatchScheduler(
        inverse_timesteps=inverse, reverse_sigmas=reverse
    )
    scheduler.set_timesteps(40)
    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(2, 16, generator=generator)
    prediction = torch.randn(2, 16, generator=generator).bfloat16()
    for i in (0, 15, 39):
        sigma = scheduler.sigmas[i]
        next_sigma = (
            (1 if inverse or reverse else 0)
            if to_final or i == 39
            else scheduler.sigmas[i + 1]
        )
        expected = sample + prediction * (next_sigma - sigma)
        actual = scheduler.step(
            prediction, scheduler.timesteps[i], sample, to_final=to_final
        )
        assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA scalar promotion"
)
def test_scheduler_cuda_bf16_prediction_uses_cpu_sigma():
    scheduler = VideoEditFlowMatchScheduler()
    scheduler.set_timesteps(40, device="cuda")
    assert scheduler.sigmas.device.type == "cpu"
    assert scheduler.timesteps.device.type == "cuda"
    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(4096, generator=generator).cuda()
    prediction = torch.randn(4096, generator=generator).bfloat16().cuda()
    for i in (0, 15, 39):
        next_sigma = scheduler.sigmas[i + 1] if i < 39 else 0
        expected = sample + prediction * (next_sigma - scheduler.sigmas[i])
        actual = scheduler.step(prediction, scheduler.timesteps[i], sample)
        assert torch.equal(actual, expected)

"""Direct numerical comparison with a local, unmodified EraserDiT checkout.

Set ERASERDIT_SOURCE to the checkout. Small randomly initialized models exercise
the complete transformer without requiring production weights.
"""
import importlib
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize('device,dtype', [('cpu', torch.float32), ('cuda', torch.bfloat16)])
def test_transformer_matches_upstream(monkeypatch, device, dtype):
    source = os.environ.get('ERASERDIT_SOURCE')
    if not source or not (Path(source) / 'models/dits/eraserdit_transformer.py').is_file():
        pytest.skip('set ERASERDIT_SOURCE to an original EraserDiT checkout')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    monkeypatch.syspath_prepend(source)
    original = importlib.import_module('models.dits.eraserdit_transformer')
    from nndeploy.e_dit.algorithms.eraserdit.models import transformer as migrated

    options = dict(in_channels=9, out_channels=4, num_attention_heads=2,
                   attention_head_dim=32, cross_attention_dim=64,
                   caption_channels=32, num_layers=2)
    torch.manual_seed(42)
    reference = original.EraserDiTLTXVideoTransformer3DModel(**options).to(device=device, dtype=dtype).eval()
    actual = migrated.EraserDiTLTXVideoTransformer3DModel(**options).to(device=device, dtype=dtype).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    inputs = dict(hidden_states=torch.randn(1, 4, 2, 2, 2, device=device, dtype=dtype),
                  cond_latents=torch.randn(1, 4, 2, 2, 2, device=device, dtype=dtype),
                  mask_values=torch.ones(1, 1, 2, 2, 2, device=device, dtype=dtype),
                  encoder_hidden_states=torch.randn(1, 5, 32, device=device, dtype=dtype),
                  encoder_attention_mask=torch.tensor([[1, 1, 1, 0, 0]], device=device),
                  timestep=torch.tensor([500.], device=device, dtype=dtype),
                  num_frames=2, height=2, width=2, rope_interpolation_scale=(.32, 32., 32.),
                  return_dict=False)
    with torch.inference_mode():
        expected = reference(**inputs)[0]
        observed = actual(**inputs)[0]
    assert torch.isfinite(observed).all()
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)

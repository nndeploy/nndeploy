# SPDX-License-Identifier: Apache-2.0
"""Small real Wan models exercise offload without production checkpoints."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from nndeploy.e_dit.algorithms.vsr.model import VSRAutoencoder, VSRRestorer


def models():
    torch.manual_seed(17)
    vae = VSRAutoencoder(
        AutoencoderKLWan(
            base_dim=8,
            z_dim=4,
            dim_mult=[1, 2],
            num_res_blocks=1,
            temperal_downsample=[True],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
    )
    dit = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=3,
    )
    return vae, dit


class OffloadConfigTest(unittest.TestCase):
    def test_mutually_exclusive(self):
        vae, dit = models()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            VSRRestorer(
                vae, dit, device="cpu", dit_cpu_offload=True, dit_layerwise_offload=True
            )

    def test_prefetch_validation(self):
        for value in (-1, float("nan"), float("inf")):
            vae, dit = models()
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                VSRRestorer(vae, dit, device="cpu", dit_offload_prefetch_size=value)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class OffloadCudaTest(unittest.TestCase):
    def test_compiled_vae_can_move_repeatedly(self):
        vae, dit = models()
        window = torch.randn(1, 3, 5, 16, 16, device="cuda")
        baseline = VSRRestorer(
            copy.deepcopy(vae),
            copy.deepcopy(dit),
            dtype=torch.float32,
            compile_encoder=True,
            compile_decoder=True,
        )
        expected = baseline.restore_window(window)
        del baseline
        restorer = VSRRestorer(
            vae,
            dit,
            dtype=torch.float32,
            vae_cpu_offload=True,
            dit_cpu_offload=True,
            compile_encoder=True,
            compile_decoder=True,
        )
        for _ in range(2):
            torch.testing.assert_close(
                restorer.restore_window(window), expected, rtol=0, atol=0
            )
            self.assertEqual(next(restorer.vae.parameters()).device.type, "cpu")
        with (
            patch.object(
                restorer.vae.vae.decoder,
                "forward",
                side_effect=RuntimeError("decoder failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "decoder failure"),
        ):
            restorer.restore_window(window)
        self.assertEqual(next(restorer.vae.parameters()).device.type, "cpu")
        torch.testing.assert_close(
            restorer.restore_window(window), expected, rtol=0, atol=0
        )

    def test_repeat_and_failure_cleanup(self):
        vae, dit = models()
        window = torch.randn(1, 3, 5, 16, 16, device="cuda")
        baseline = VSRRestorer(
            copy.deepcopy(vae), copy.deepcopy(dit), dtype=torch.float32
        )
        expected = baseline.restore_window(window)
        del baseline
        for mode in ("whole", "layerwise"):
            for cache in (False, True):
                with self.subTest(mode=mode, cache=cache):
                    restorer = VSRRestorer(
                        copy.deepcopy(vae),
                        copy.deepcopy(dit),
                        dtype=torch.float32,
                        vae_cpu_offload=True,
                        dit_cpu_offload=mode == "whole",
                        dit_layerwise_offload=mode == "layerwise",
                        cache_dit_condition=cache,
                    )
                    for _ in range(2):
                        actual = restorer.restore_window(window)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(
                            next(restorer.vae.parameters()).device.type, "cpu"
                        )
                        if mode == "whole":
                            self.assertEqual(
                                next(restorer.dit.parameters()).device.type, "cpu"
                            )
                        else:
                            self.assertFalse(restorer.dit_offload_manager._gpu_layers)

                    # Fail after the layer pre-hook has loaded weights.
                    def fail(*args):
                        raise RuntimeError("injected block failure")

                    hook = restorer.dit.blocks[1].register_forward_pre_hook(fail)
                    try:
                        with self.assertRaisesRegex(
                            RuntimeError, "injected block failure"
                        ):
                            restorer.restore_window(window)
                    finally:
                        hook.remove()
                    self.assertEqual(next(restorer.vae.parameters()).device.type, "cpu")
                    if mode == "layerwise":
                        self.assertFalse(restorer.dit_offload_manager._gpu_layers)
                    else:
                        self.assertEqual(
                            next(restorer.dit.parameters()).device.type, "cpu"
                        )
                    torch.testing.assert_close(
                        restorer.restore_window(window), expected, rtol=0, atol=0
                    )
                    del restorer


if __name__ == "__main__":
    unittest.main()

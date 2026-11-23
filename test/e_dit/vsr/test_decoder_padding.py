# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from types import MethodType

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from nndeploy.e_dit.algorithms.vsr.compile import _implicit_spatial_padding


class DecoderPaddingTest(unittest.TestCase):
    def test_preserves_causal_convolution_and_cache(self):
        torch.manual_seed(41)
        for temporal_kernel in (1, 3):
            for spatial_stride in (1, 2):
                original = WanCausalConv3d(
                    3,
                    4,
                    (temporal_kernel, 3, 3),
                    stride=(1, spatial_stride, spatial_stride),
                    padding=(temporal_kernel // 2, 1, 1),
                ).eval()
                candidate = copy.deepcopy(original)
                candidate.forward = MethodType(_implicit_spatial_padding, candidate)
                for frames in (1, 4):
                    for cached_frames in (0, 1, 2):
                        x = torch.randn(2, 3, frames, 9, 11)
                        cache = (
                            torch.randn(2, 3, cached_frames, 9, 11)
                            if cached_frames
                            else None
                        )
                        saved = cache.clone() if cache is not None else None
                        with torch.no_grad():
                            expected = original(x, cache)
                            actual = candidate(x, cache)
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-5, atol=1e-6
                        )
                        if cache is not None:
                            self.assertTrue(torch.equal(cache, saved))


if __name__ == "__main__":
    unittest.main()

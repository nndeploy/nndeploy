# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from types import MethodType

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample
from nndeploy.e_dit.algorithms.vsr.compile import _upsample3d_forward


class CompilerCacheTest(unittest.TestCase):
    def test_upsampler_preserves_outputs_and_caches(self):
        torch.manual_seed(9)
        original = WanResample(4, "upsample3d").eval()
        candidate = copy.deepcopy(original)
        candidate.forward = MethodType(_upsample3d_forward, candidate)
        # Includes None -> Rep -> Tensor, one-frame and multi-frame chunks,
        # and a second video after resetting causal state.
        for lengths in ([1, 1, 1, 2], [1, 2, 2]):
            ref_cache, cand_cache = [None], [None]
            for length in lengths:
                x = torch.randn(1, 4, length, 4, 6)
                ref_idx, cand_idx = [0], [0]
                with torch.no_grad():
                    a = original(x, ref_cache, ref_idx)
                    b = candidate(x, cand_cache, cand_idx)
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(ref_idx, cand_idx)
                if isinstance(ref_cache[0], torch.Tensor):
                    self.assertTrue(torch.equal(ref_cache[0], cand_cache[0]))
                else:
                    self.assertEqual(ref_cache, cand_cache)
        x = torch.randn(1, 4, 2, 4, 6)
        self.assertTrue(torch.equal(original(x), candidate(x)))


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import Mock

import torch
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor
from nndeploy.e_dit.algorithms.vsr.condition_cache import _FixedForward


class ConditionCacheTest(unittest.TestCase):
    def test_query_reuse_and_invalidation(self):
        torch.manual_seed(71)
        module = WanAttention(
            32,
            heads=2,
            dim_head=16,
            cross_attention_dim_head=16,
            processor=WanAttnProcessor(),
        ).eval()
        cached = _FixedForward(module, cross_attention=True)
        original = module.forward
        cached.original = Mock(wraps=original)
        text = torch.randn(1, 1, 32)
        with torch.no_grad():
            for _ in range(3):
                query = torch.randn(1, 8, 32)
                actual = cached(query, text, None, None)
                torch.testing.assert_close(
                    actual, original(query, text, None, None), rtol=0, atol=0
                )
            self.assertEqual(cached.original.call_count, 1)
            text.add_(0.1)
            torch.testing.assert_close(
                cached(query, text), original(query, text), rtol=0, atol=0
            )
            self.assertEqual(cached.original.call_count, 2)
            module.to_v.weight.add_(0.1)
            torch.testing.assert_close(
                cached(query, text), original(query, text), rtol=0, atol=0
            )
            self.assertEqual(cached.original.call_count, 3)
            # If a consumer mutates the cached value, recompute it next time.
            cached.value.add_(1)
            torch.testing.assert_close(
                cached(query, text), original(query, text), rtol=0, atol=0
            )
            self.assertEqual(cached.original.call_count, 4)
            longer = torch.randn(1, 2, 32)
            torch.testing.assert_close(cached(query, longer), original(query, longer))
            torch.testing.assert_close(
                cached(query, encoder_hidden_states=text), original(query, text)
            )
        with torch.inference_mode():
            torch.testing.assert_close(
                cached(query, text), original(query, text), rtol=0, atol=0
            )
            unknown_version = text.clone()
            torch.testing.assert_close(
                cached(query, unknown_version), original(query, unknown_version)
            )

        with torch.no_grad():
            module.double()
            query64, text64 = query.double(), text.double()
            torch.testing.assert_close(
                cached(query64, text64), original(query64, text64), rtol=0, atol=0
            )
        query64.requires_grad_(True)
        self.assertTrue(cached(query64, text64).requires_grad)


if __name__ == "__main__":
    unittest.main()

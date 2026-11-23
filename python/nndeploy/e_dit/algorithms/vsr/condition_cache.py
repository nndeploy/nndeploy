# SPDX-License-Identifier: Apache-2.0
"""Bounded caches for VSR's fixed t=1000 and single-token text conditioning.

Only install on the private DiT of VSRRestorer. This is not a general diffusion
cache: VSRRestorer owns the constant timestep. Cross-attention with one key has
softmax weight 1, so its output is independent of the image query. Populate the
cache using the original full query shape to retain the original GEMM rounding.
"""

import torch


def _signature(tensor):
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.data_ptr(),
        tensor._version,
    )


class _FixedForward:
    def __init__(self, module, cross_attention=False):
        self.module = module
        self.original = module.forward
        self.cross_attention = cross_attention
        self.key = None
        self.value = None
        self.references = None
        self.value_versions = None

    def __call__(self, *args, **kwargs):
        if self.module.training or torch.is_grad_enabled() or len(args) < 2:
            return self.original(*args, **kwargs)
        x, text = args[:2]
        extra = args[2:]
        if self.cross_attention:
            supported = (
                text is not None
                and text.ndim == 3
                and text.shape[1] == 1
                and not any(a is not None for a in extra)
                and not kwargs
                and self.module.add_k_proj is None
            )
        else:
            supported = (
                text is not None
                and x.ndim == 1
                and not any(a is not None for a in extra)
                and set(kwargs) <= {"timestep_seq_len"}
                and kwargs.get("timestep_seq_len") is None
            )
        if not supported:
            return self.original(*args, **kwargs)
        params = tuple(self.module.parameters())
        try:
            key = (
                tuple(x.shape),
                x.dtype,
                x.device,
                _signature(text),
                tuple(_signature(p) for p in params),
            )
        except RuntimeError:
            # Inference tensors without version counters cannot be safely
            # tracked for mutation; retain the original behavior in that case.
            return self.original(*args, **kwargs)
        values = self.value if isinstance(self.value, tuple) else (self.value,)
        versions = tuple(
            v._version if isinstance(v, torch.Tensor) else None for v in values
        )
        if key != self.key or versions != self.value_versions:
            # Normal tensors keep version counters even under an outer
            # inference_mode context; this only runs on cache misses.
            with torch.inference_mode(False), torch.no_grad():
                self.value = self.original(*args, **kwargs)
            self.key = key
            # One entry per module. Holding references prevents allocator
            # address reuse from producing a false hit after input replacement.
            self.references = (text, params)
            values = self.value if isinstance(self.value, tuple) else (self.value,)
            self.value_versions = tuple(
                v._version if isinstance(v, torch.Tensor) else None for v in values
            )
        return self.value


def cache_fixed_vsr_condition(dit):
    """Install inference-only caches on this VSR DiT instance, once."""
    modules = [(dit.condition_embedder, False)]
    modules.extend((block.attn2, True) for block in dit.blocks)
    for module, cross in modules:
        if not isinstance(module.forward, _FixedForward):
            module.forward = _FixedForward(module, cross_attention=cross)


def clear_fixed_vsr_condition(model):
    """Drop cached GPU tensors before moving weights or releasing DiT blocks."""
    for module in model.modules():
        if isinstance(module.forward, _FixedForward):
            cached = module.forward
            cached.key = cached.value = cached.references = cached.value_versions = None

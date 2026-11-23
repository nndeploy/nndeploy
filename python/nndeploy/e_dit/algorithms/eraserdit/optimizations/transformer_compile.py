"""Strict whole-DiT compilation; scheduler, I/O and runtime state stay eager."""
import time

import torch


class CompiledTransformer:
    def __init__(self, model, *, mode):
        if mode not in ('default', 'max-autotune-no-cudagraphs'):
            raise ValueError('transformer compile supports default or max-autotune-no-cudagraphs')
        if getattr(model, '_block_compile_report', None):
            raise ValueError('remove FFN compilation before compiling the whole transformer')
        if any(m.enabled for m in getattr(model, 'layerwise_offload_managers', ())):
            raise ValueError('transformer compile cannot capture layerwise offload hooks')
        if model.operator_fusion_decision.effective_ops:
            raise ValueError('disable operator fusion before whole-transformer compilation')
        if any(block.attn1.processor.attention_backend != 'sdpa' for block in model.transformer_blocks):
            raise ValueError('transformer compile currently requires explicit sdpa attention')
        self.model = model
        self._prepared = set()
        self._seen = set()
        self.report = dict(scope='transformer', fullgraph=True, dynamic=False,
                           cudagraphs=False, linear_backend='inductor',
                           successful_forwards=0, first_call_history=[])
        torch._inductor.config.emulate_precision_casts = True
        torch._inductor.config.triton.cudagraphs = False
        self.forward = torch.compile(model.forward, mode=mode, fullgraph=True, dynamic=False)

    def __call__(self, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('transformer compile is inference-only; use torch.no_grad()')
        if any(kwargs.get(key) is not None for key in ('cache_adapter', 'text_cache', 'sequence_parallel')):
            raise ValueError('transformer compile does not support caches or sequence parallelism')
        hidden = kwargs['hidden_states']
        dtype = torch.get_autocast_dtype(hidden.device.type) if torch.is_autocast_enabled(hidden.device.type) else hidden.dtype
        preparation_key = (str(hidden.device), dtype)
        if preparation_key not in self._prepared:
            for block in self.model.transformer_blocks:
                block.attn1.processor.preflight_self_attention_backend(
                    device=hidden.device, dtype=dtype,
                    head_size=self.model.config.attention_head_dim,
                    num_heads=self.model.config.num_attention_heads)
            self._prepared.add(preparation_key)
        # Actual tensor values (especially timesteps) must not become Python
        # constants. These signatures describe diagnostics only, not a cache.
        signature = tuple((key, tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
                          for key, value in sorted(kwargs.items()) if isinstance(value, torch.Tensor))
        first = signature not in self._seen
        if first and hidden.is_cuda:
            torch.cuda.synchronize(hidden.device)
        started = time.perf_counter()
        result = self.forward(**kwargs)
        if first:
            if hidden.is_cuda:
                torch.cuda.synchronize(hidden.device)
            self.report['first_call_history'].append(dict(
                input_shape=list(hidden.shape), seconds=time.perf_counter() - started))
            self._seen.add(signature)
        self.report['successful_forwards'] += 1
        return result


def configure_transformer_compile(model, *, mode='max-autotune-no-cudagraphs'):
    return CompiledTransformer(model, mode=mode)

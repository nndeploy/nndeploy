"""Compile block-local feed-forward compute; residency/cache/collectives stay eager.

The FFN is a substantial, communication-free tensor region. Keeping attention
and modulation outside this first boundary preserves their eager rounding and
avoids capturing peer barriers or mutable cache dictionaries in Dynamo.
"""
import os
from contextlib import ExitStack
from types import MethodType

import torch


@torch.library.custom_op('eraserdit::native_linear', mutates_args=())
def _native_linear(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    # Keep the bias inside the native GEMM's BF16 output rounding. Inductor's
    # addmm -> mm + fused bias/GELU rewrite exceeded the video SSIM gate.
    return torch.nn.functional.linear(value, weight, bias)


@_native_linear.register_fake
def _native_linear_fake(value, weight, bias):
    return value.new_empty((*value.shape[:-1], weight.shape[0]))


def _linear_forward(self, value):
    return _native_linear(value, self.weight, self.bias)


def configure_block_compile(model, *, mode='max-autotune-no-cudagraphs'):
    if mode not in ('default', 'max-autotune-no-cudagraphs'):
        raise ValueError('composable compile supports default or max-autotune-no-cudagraphs; CUDA graphs are disabled')
    linear_backend = os.environ.get('NNDEPLOY_DIT_ERASERDIT_COMPILE_LINEAR_BACKEND', 'native')
    if linear_backend not in ('native', 'inductor'):
        raise ValueError('NNDEPLOY_DIT_ERASERDIT_COMPILE_LINEAR_BACKEND must be native or inductor')
    previous = getattr(model, '_block_compile_report', None)
    if previous is not None:
        if previous['mode'] == mode and previous['linear_backend'] == linear_backend:
            return previous
        remove_block_compile(model)
    config = torch._inductor.config
    config.emulate_precision_casts = True
    config.triton.cudagraphs = False
    count = 0
    for block in model.transformer_blocks:
        ff = block.ff
        if not hasattr(ff, '_eager_forward'):
            ff._compile_warmed_shapes = set()
            if linear_backend == 'native':
                for module in ff.modules():
                    if isinstance(module, torch.nn.Linear):
                        module._precompile_forward = module.forward
                        module.forward = MethodType(_linear_forward, module)
            ff._eager_forward = ff.forward
            ff.forward = torch.compile(ff.forward, mode=mode, fullgraph=True, dynamic=False)
        count += 1
    model._block_compile_report = dict(scope='block_ffn', blocks=count, mode=mode, cudagraphs=False, linear_backend=linear_backend)
    return model._block_compile_report


def remove_block_compile(model):
    for block in model.transformer_blocks:
        block.ff.__dict__.pop('_compile_warmed_shapes', None)
        if hasattr(block.ff, '_eager_forward'):
            block.ff.forward = block.ff._eager_forward
            del block.ff._eager_forward
        for module in block.ff.modules():
            if hasattr(module, '_precompile_forward'):
                module.forward = module._precompile_forward
                del module._precompile_forward
    model.__dict__.pop('_block_compile_report', None)


def prepare_block_compile(model, *, batch_size, sequence_length, device, dtype):
    """Warm every FFN signature before peer threads enter eager/compiled code.

    FX tracing temporarily patches Module.__call__ process-wide in Torch 2.6.
    Serial preparation prevents another rank from entering that trace. No RNG
    is consumed and no attention/collectives run. Offloaded weights are borrowed
    one block at a time; only FFN tensor compute is compiled, without CUDA graphs.
    """
    import time
    if not hasattr(model, '_block_compile_report'):
        return 0.0
    started = time.perf_counter()
    signature = (batch_size, sequence_length, str(device), str(dtype))
    width = model.config.num_attention_heads * model.config.attention_head_dim
    managers = [m for m in getattr(model, 'layerwise_offload_managers', ())
                if m.enabled and m.layers_attr_str == 'transformer_blocks']
    with torch.cuda.device(device), torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), ExitStack() as cleanup:
        for manager in managers:
            cleanup.callback(manager.release_all)
        value = None
        for index, block in enumerate(model.transformer_blocks):
            warmed = getattr(block.ff, '_compile_warmed_shapes', set())
            if signature in warmed:
                continue
            if value is None:
                value = torch.zeros(batch_size, sequence_length, width, device=device, dtype=dtype)
            with ExitStack() as residency:
                for manager in managers:
                    residency.enter_context(manager.layer_residency(index))
                block.ff(value)
            warmed.add(signature)
            block.ff._compile_warmed_shapes = warmed
        if value is not None:
            torch.cuda.current_stream(device).synchronize()
    elapsed = time.perf_counter() - started
    if value is not None:
        report = model._block_compile_report
        report['preparation_seconds_total'] = report.get('preparation_seconds_total', 0.0) + elapsed
        report.setdefault('preparation_history', []).append(dict(
            batch_size=batch_size, sequence_length=sequence_length,
            device=str(device), dtype=str(dtype), seconds=elapsed))
    return elapsed

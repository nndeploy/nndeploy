"""Persistent CPU-created replicas; each rank owns its offload state."""
from nndeploy.e_dit.utils.env import get_env
from copy import deepcopy
import os

import torch


def clone_transformer_cpu(source):
    # Populate deepcopy's memo with CPU tensors first: never transiently clone
    # a complete model on the source GPU. Preserve tied Parameter identities.
    memo = {}
    for tensor in list(source.parameters()) + list(source.buffers()):
        if id(tensor) not in memo:
            value = tensor.detach().to(device='cpu', copy=True)
            memo[id(tensor)] = (torch.nn.Parameter(value, requires_grad=tensor.requires_grad)
                                if isinstance(tensor, torch.nn.Parameter) else value)
    for block in source.transformer_blocks:
        if hasattr(block.ff, '_eager_forward'):
            memo[id(block.ff.forward)] = None
    replica = deepcopy(source, memo).eval()
    from nndeploy.e_dit.algorithms.eraserdit.optimizations.block_compile import remove_block_compile
    remove_block_compile(replica)
    return replica


class EraserDiTReplicaPool:
    def __init__(self, source, plan, args, *, compiled_transformer=None):
        self.models = [source]
        self.compiled_forwards = []
        self.adapters = []
        self.plan = plan
        self.policy = args.resolve_resource_policy()
        self.closed = False
        self.active = []
        try:
            for device in plan['devices'][1:plan['sp'] * plan['cfg']]:
                model = clone_transformer_cpu(source)
                self.models.append(model)
                adapter = None
                if not self.policy.dit_cpu_offload:
                    model.to(device)
                self.adapters.append(adapter)
                if args.enable_torch_compile and args.torch_compile_scope == 'ffn':
                    from nndeploy.e_dit.algorithms.eraserdit.optimizations.block_compile import configure_block_compile
                    mode = get_env('NNDEPLOY_DIT_TORCH_COMPILE_MODE', 'max-autotune-no-cudagraphs').strip()
                    configure_block_compile(model, mode=mode)
            if args.enable_torch_compile and args.torch_compile_scope == 'transformer':
                from nndeploy.e_dit.algorithms.eraserdit.optimizations.compile_config import validate_transformer_compile
                from nndeploy.e_dit.algorithms.eraserdit.optimizations.transformer_compile import configure_transformer_compile
                validate_transformer_compile(args)
                if plan['sp'] != 1:
                    raise ValueError('whole-transformer mesh compilation requires SP1')
                mode = get_env('NNDEPLOY_DIT_TORCH_COMPILE_MODE', 'max-autotune-no-cudagraphs').strip()
                self.compiled_forwards = [
                    compiled_transformer if index == 0 and compiled_transformer is not None
                    else configure_transformer_compile(model, mode=mode)
                    for index, model in enumerate(self.models)]
        except BaseException:
            self.close()
            raise

    def acquire(self):
        if self.closed or self.active:
            raise RuntimeError('replica pool is closed or already acquired')
        try:
            for index, adapter in enumerate(self.adapters, 1):
                self.active.append(index)
                if adapter:
                    adapter.acquire_component_residency('transformer', reason='mesh_window')
                elif self.policy.dit_cpu_offload:
                    self.models[index].to(self.plan['devices'][index])
        except BaseException:
            self.release()
            raise

    def release(self):
        try:
            for index in reversed(self.active):
                adapter = self.adapters[index - 1]
                if adapter:
                    adapter.offload_remainder('transformer')
                elif self.policy.dit_cpu_offload:
                    torch.cuda.synchronize(self.plan['devices'][index])
                    self.models[index].to('cpu')
        finally:
            self.active.clear()

    def snapshot(self):
        return [adapter.snapshot() if adapter else {'backend': 'resident'}
                for adapter in self.adapters]

    def close(self):
        self.release()
        for adapter in self.adapters:
            if adapter:
                adapter.shutdown()
        self.adapters.clear()
        self.compiled_forwards.clear()
        self.models.clear()
        self.closed = True

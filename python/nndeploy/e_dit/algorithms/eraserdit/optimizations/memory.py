"""Pipeline ownership around migrated SGLang managers (no custom scheduler).

DiT uses the upstream hooks and wraparound prefetch. T5 uses FSDP CPU offload.
Other components follow SGLang stage-level Module.to placement. Measurements
observe current storage; they are not a byte budget or allocator peak.
"""
from collections import deque
import os
import tempfile
import threading
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nndeploy.e_dit.memory.backends.fsdp_offload import shard_model, MixedPrecisionPolicy
from nndeploy.e_dit.memory.telemetry import memory_observation

_group_lock = threading.Lock()
_group_users = 0
_group_owned = False
_group_store = None


def _acquire_group():
    global _group_users, _group_owned, _group_store
    with _group_lock:
        if not dist.is_initialized():
            _group_store = tempfile.TemporaryDirectory(prefix='eraserdit-fsdp-')
            try:
                dist.init_process_group('nccl', init_method=f'file://{_group_store.name}/store',
                                        rank=0, world_size=1)
            except BaseException:
                _group_store.cleanup()
                _group_store = None
                raise
            _group_owned = True
        if dist.get_world_size() != 1:
            raise ValueError('This EraserDiT FSDP integration currently requires world_size=1')
        _group_users += 1


def _release_group():
    global _group_users, _group_owned, _group_store
    with _group_lock:
        _group_users -= 1
        if not _group_users and _group_owned:
            dist.destroy_process_group()
            _group_owned = False
            _group_store.cleanup()
            _group_store = None


def _storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        if hasattr(tensor, 'to_local'):
            tensor = tensor.to_local()
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class SGLangMemoryAdapter:
    def __init__(self, modules, args):
        self.modules = modules
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.managers = []
        self.active_component_name = None
        self.events = deque(maxlen=64)
        self.closed = False
        self.group_acquired = False
        self.fsdp_modules = []
        self.peak_observed = 0
        self.remainder = []
        self.vae_residency = None
        try:
            vae_mode = os.environ.get('NNDEPLOY_DIT_ERASERDIT_VAE_OFFLOAD_MODE', 'full')
            if vae_mode not in ('full', 'split', 'cached'):
                raise ValueError('NNDEPLOY_DIT_ERASERDIT_VAE_OFFLOAD_MODE must be full, split or cached')
            if vae_mode != 'full' and args.vae_cpu_offload and 'vae' in modules:
                from nndeploy.e_dit.memory.backends.vae_residency import VAEPhaseResidency
                self.vae_residency = VAEPhaseResidency(
                    modules['vae'], self.device, mode=vae_mode, pin_memory=args.pin_cpu_memory)
            if args.dit_layerwise_offload:
                transformer = modules['transformer']
                with torch.cuda.device(self.device):
                    transformer.configure_layerwise_offload(args)
                self.managers = transformer.layerwise_offload_managers
                managed = {name for m in self.managers for name, _ in m.iter_cpu_weights()}
                self.remainder = [t for name, t in list(transformer.named_parameters()) +
                                  list(transformer.named_buffers()) if name not in managed]
                # SGLang only offloads blocks; the remaining DiT weights stay on device.
                for tensor in self.remainder:
                    tensor.data = tensor.data.to(self.device)
            if args.text_encoder_cpu_offload:
                from transformers.models.t5.modeling_t5 import T5Block, T5EncoderModel
                text = modules['text_encoder']
                if not isinstance(text, T5EncoderModel):
                    raise ValueError('Text CPU offload requires T5EncoderModel shard conditions')
                with torch.cuda.device(self.device):
                    _acquire_group()
                    self.group_acquired = True
                    mesh = init_device_mesh('cuda', (1, 1), mesh_dim_names=('offload', 'replicate'))
                    shard_model(text, cpu_offload=True, reshard_after_forward=True,
                                mesh=mesh['offload'], mp_policy=MixedPrecisionPolicy(),
                                fsdp_shard_conditions=[lambda n, m: isinstance(m, T5Block)],
                                pin_cpu_memory=args.pin_cpu_memory)
                self.fsdp_modules = [m for m in text.modules() if hasattr(m, 'reshard')]
                text._nndeploy_execution_device = self.device
            self.snapshot()
        except BaseException:
            self.shutdown(terminal=True)
            raise

    def acquire_component_residency(self, name, *, reason, phase=None):
        if self.closed or self.active_component_name is not None:
            raise RuntimeError('memory adapter closed or a component is already active')
        started = time.perf_counter()
        try:
            if name == 'vae' and self.vae_residency is not None:
                self.vae_residency.acquire(phase)
            elif name == 'transformer' and self.managers:
                with torch.cuda.device(self.device):
                    self.modules[name].prepare_for_next_req()
            elif name == 'text_encoder' and self.args.text_encoder_cpu_offload:
                pass  # FSDP owns movement, including the root embedding parameters.
            elif self._offloads(name):
                self.modules[name].to(self.device, non_blocking=True)
            self.active_component_name = name
            self._record(name, 'acquire', reason, started)
        except BaseException:
            self._release(name)
            raise

    def _offloads(self, name):
        return getattr(self.args, {'transformer': 'dit_cpu_offload', 'vae': 'vae_cpu_offload',
                                   'text_encoder': 'text_encoder_cpu_offload'}[name])

    def _release(self, name):
        if name == 'vae' and self.vae_residency is not None:
            self.vae_residency.release()
        elif name == 'transformer' and self.managers:
            with torch.cuda.device(self.device):
                for manager in self.managers:
                    manager.release_all()
                torch.cuda.current_stream().synchronize()
        elif name == 'text_encoder' and self.fsdp_modules:
            # Also reshard after a forward exception; FSDP post-hooks may not run.
            for module in reversed(self.fsdp_modules):
                # Torch 2.6 single-module FSDP post-hooks are not always_call.
                # Finish a stranded forward before reshard: the root otherwise
                # ignores reshard while FORWARD with reshard_after_forward=False.
                state = module._get_fsdp_state()
                if state._training_state.name == 'FORWARD':
                    with torch.no_grad():
                        state._post_forward(module, (), None)
                module.reshard()
            torch.cuda.synchronize(self.device)
        elif self._offloads(name):
            self.modules[name].to('cpu', non_blocking=True)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

    def release_component_residency(self, name, *, reason):
        if self.active_component_name != name:
            raise RuntimeError('component release does not match active phase')
        started = time.perf_counter()
        try:
            self._release(name)
        finally:
            self.active_component_name = None
        from nndeploy.e_dit.memory.allocator import release_idle_cuda_cache
        release_idle_cuda_cache(self.args.cuda_memory_limit_gib, self.device)
        self._record(name, 'release', reason, started)

    def settle_component_transfers(self):
        pass  # Releases above settle ownership at the stage boundary.

    def _record(self, name, action, reason, started):
        transfer = {}
        if name == 'vae' and self.args.vae_cpu_offload:
            if self.vae_residency is not None:
                transfer = dict(self.vae_residency.stats)
                transfer['h2d_bytes' if action == 'release' else 'd2h_bytes'] = 0
            else:
                tensors = list(self.modules[name].parameters()) + list(self.modules[name].buffers())
                size = sum(t.numel() * t.element_size() for t in tensors)
                transfer = dict(mode='full', scope='full', active_weight_bytes=size,
                                h2d_bytes=size if action == 'acquire' else 0,
                                d2h_bytes=size if action == 'release' else 0)
        self.events.append(dict(component=name, action=action, reason=reason,
                                seconds=time.perf_counter()-started,
                                memory=memory_observation(self.device),
                                **({'weight_transfers': transfer} if transfer else {})))

    def snapshot(self):
        resident, pinned, layers = 0, 0, 0
        for manager in self.managers:
            layers += len(manager._gpu_layers)
            gpu = [manager.get_target_with_name(n) for i in manager._gpu_layers
                   for n in manager._weight_metadata[i]]
            resident += _storage_bytes(gpu)
            pinned += _storage_bytes(t for _, t in manager.iter_cpu_weights() if t.is_pinned())
        self.peak_observed = max(self.peak_observed, resident)
        return dict(backend='sglang_source', weight_budget_scope=None,
                    resident_bytes=resident, observed_peak_resident_bytes=self.peak_observed,
                    resident_measurement='snapshot_only_excludes_allocator_pending_frees',
                    live_layers=layers, pinned_cpu_bytes=pinned,
                    active_component_name=self.active_component_name,
                    text_encoder_backend='fsdp_cpu_offload' if self.fsdp_modules else 'resident',
                    vae_offload=self.vae_residency.snapshot() if self.vae_residency else {'mode': 'full'},
                    component_transfers=list(self.events), closed=self.closed)

    def shutdown(self, *, terminal=False):
        if self.closed:
            return self.snapshot()
        try:
            if self.active_component_name:
                self.release_component_residency(self.active_component_name, reason='shutdown')
            for manager in self.managers:
                with torch.cuda.device(self.device):
                    manager.release_all()
                    manager.copy_stream.synchronize()
                    torch.cuda.current_stream().synchronize()
                manager.remove_forward_hooks()
                # Shutdown is terminal for this pipeline. No restore/re-registration API.
                for name, cpu in manager.iter_cpu_weights():
                    manager.get_target_with_name(name).data = torch.empty(0, dtype=cpu.dtype)
                manager._consolidated_cpu_weights.clear()
                manager._strided_cpu_weights.clear()
                manager._weight_metadata.clear()
                manager._offload_placeholders.clear()
            self.managers.clear()
            if 'transformer' in self.modules:
                self.modules['transformer'].layerwise_offload_managers = []
            for tensor in self.remainder:
                tensor.data = torch.empty(0, dtype=tensor.dtype)
            self.remainder.clear()
            self.fsdp_modules.clear()
        finally:
            if self.group_acquired:
                _release_group()
                self.group_acquired = False
            self.closed = True
        return self.snapshot()

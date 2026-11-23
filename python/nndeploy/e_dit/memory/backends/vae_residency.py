"""Phase-local VAE placement with optional reusable CPU parameter backing.

The cached path is for eval-only inference. Registered buffers are always
written back; parameter updates through normal in-place ops/load_state_dict
are detected with version counters. Direct .data writes during a lease are
unsupported, as with the immutable CPU backing of layerwise DiT offload.
"""
import torch


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


class VAEPhaseResidency:
    def __init__(self, model, device, *, mode, pin_memory=True):
        if mode not in ('split', 'cached'):
            raise ValueError('VAE phase residency mode must be split or cached')
        if not all(isinstance(getattr(model, name, None), torch.nn.Module)
                   for name in ('encoder', 'decoder')):
            raise ValueError('VAE phase residency requires encoder and decoder modules')
        self.model, self.device = model, torch.device(device)
        self.mode, self.pin_memory = mode, pin_memory
        self.records = []
        self.active = False
        self.stats = {}
        self._previous_device = None
        self._had_device = False

    def _upload(self, tensor):
        return tensor.to(self.device, non_blocking=True)

    def acquire(self, phase):
        if self.active:
            raise RuntimeError('VAE phase already active')
        if self.model.training:
            raise ValueError('VAE phase offload requires eval mode')
        scopes = {'vae_encode': 'encoder', 'vae_decode': 'decoder'}
        if phase not in scopes:
            raise ValueError('VAE offload requires an explicit encode/decode phase')
        scope = scopes[phase]
        self.stats = dict(mode=self.mode, scope=scope, h2d_bytes=0, d2h_bytes=0,
                          active_weight_bytes=0, reused_cpu_bytes=0)
        self._had_device = hasattr(self.model, '_nndeploy_execution_device')
        self._previous_device = getattr(self.model, '_nndeploy_execution_device', None)
        self.active = True
        try:
            # A tied tensor can have an encoder and a decoder name. Select by
            # identity across all aliases before traversing unique tensors.
            selected = set()
            for iterator in (self.model.named_parameters(remove_duplicate=False),
                             self.model.named_buffers(remove_duplicate=False)):
                for name, tensor in iterator:
                    if not name.startswith(('encoder.', 'decoder.')) or name.startswith(scope + '.'):
                        selected.add(id(tensor))
            for is_buffer, values in ((False, self.model.named_parameters()),
                                      (True, self.model.named_buffers())):
                for name, tensor in values:
                    if id(tensor) not in selected:
                        continue
                    if tensor.device.type != 'cpu':
                        raise ValueError(f'VAE offload expected CPU tensor: {name}')
                    cpu = None
                    if self.mode == 'cached':
                        if self.pin_memory and self.device.type == 'cuda' and not tensor.is_pinned():
                            tensor.data = tensor.detach().pin_memory()
                        cpu = tensor.detach()
                    self.records.append((tensor, cpu, _version(tensor), is_buffer))
                    size = tensor.numel() * tensor.element_size()
                    tensor.data = self._upload(tensor.detach())
                    self.stats['active_weight_bytes'] += size
                    self.stats['h2d_bytes'] += size if self.device.type == 'cuda' else 0
                    if cpu is not None:
                        self.stats['reused_cpu_bytes'] += size
            self.model._nndeploy_execution_device = self.device
        except BaseException:
            self.release()
            raise

    def release(self):
        if not self.active:
            return
        pending = []
        try:
            for tensor, cpu, version, is_buffer in self.records:
                gpu = tensor.detach()
                if gpu.device.type == 'cpu':
                    continue  # A partially failed acquire may leave CPU entries.
                size = gpu.numel() * gpu.element_size()
                if cpu is None:
                    tensor.data = gpu.to('cpu', non_blocking=True)
                    self.stats['d2h_bytes'] += size
                else:
                    # Buffers may be mutable even in eval mode. Never discard
                    # them, or tracked parameter updates made during residency.
                    if is_buffer or version is None or _version(tensor) != version:
                        cpu.copy_(gpu, non_blocking=True)
                        self.stats['d2h_bytes'] += size
                    tensor.data = cpu
                pending.append(gpu)
            # A forward can replace a registered buffer rather than mutate it
            # in place. load_state_dict(assign=True) can replace parameters as
            # well. Preserve newly registered GPU tensors via the slow path.
            for tensor in list(self.model.parameters()) + list(self.model.buffers()):
                if tensor.device.type != 'cpu':
                    gpu = tensor.detach()
                    tensor.data = gpu.to('cpu', non_blocking=True)
                    self.stats['d2h_bytes'] += gpu.numel() * gpu.element_size()
                    pending.append(gpu)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
        finally:
            # Retain GPU storages until queued copies have completed above.
            self.records.clear()
            self.active = False
            if self._had_device:
                self.model._nndeploy_execution_device = self._previous_device
            elif hasattr(self.model, '_nndeploy_execution_device'):
                del self.model._nndeploy_execution_device

    def snapshot(self):
        tensors = list(self.model.parameters()) + list(self.model.buffers())
        tensors.extend(cpu for _, cpu, _, _ in self.records if cpu is not None)
        storage = {}
        for tensor in tensors:
            if tensor.device.type == 'cpu':
                value = tensor.untyped_storage()
                storage[value.data_ptr()] = (value.nbytes(), tensor.is_pinned())
        return dict(mode=self.mode, active=self.active,
                    cpu_backing_bytes=sum(size for size, _ in storage.values()),
                    pinned_cpu_backing_bytes=sum(size for size, pinned in storage.values() if pinned),
                    last_phase_transfers=dict(self.stats))

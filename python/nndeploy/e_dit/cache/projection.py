"""Lossless, window/CFG-owned text projection cache for fixed inference weights."""
import torch


class ProjectionCache:
    def __init__(self, on_update=None):
        self.on_update = on_update
        self.source = None
        self.signature = None
        self.projected = None
        self.entries = {}
        self.retained_storages = {}
        self.projection_hits = 0
        self.kv_hits = 0
        self.peak_bytes = 0

    def project(self, source, projection):
        if torch.is_grad_enabled():
            raise RuntimeError('projection caching is inference-only')
        try:
            version = source._version
        except RuntimeError:
            # Inference tensors have no mutation counter: conservatively rebuild.
            version = object()
        device_type = source.device.type
        signature = (version, tuple(source.shape), source.dtype, source.device, id(projection),
                     torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type))
        if self.source is not source or self.signature != signature:
            self.clear()
            self.source, self.signature = source, signature
        if self.projected is None:
            self.projected = projection(source).detach()
            self._record_bytes()
        else:
            self.projection_hits += 1
        return self.projected

    def get_or_compute(self, key, compute):
        if torch.is_grad_enabled():
            raise RuntimeError('projection caching is inference-only')
        if self.projected is None:
            raise RuntimeError('project must establish the conditioning scope first')
        if key not in self.entries:
            self.entries[key] = tuple(value.detach() for value in compute())
            self._record_bytes()
        else:
            self.kv_hits += 1
        return self.entries[key]

    def _record_bytes(self):
        self.retained_storages = {
            (str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
            for t in self.tensors()
        }
        self.peak_bytes = max(self.peak_bytes, sum(self.retained_storages.values()))
        if self.on_update is not None:
            self.on_update()

    def tensors(self):
        yield from (t for t in (self.source, self.projected) if t is not None)
        yield from (tensor for pair in self.entries.values() for tensor in pair)

    def clear(self):
        self.source = self.signature = self.projected = None
        self.entries.clear()
        self.retained_storages.clear()

    def stats(self):
        return {'projection_hits': self.projection_hits, 'kv_hits': self.kv_hits,
                'peak_retained_tensor_bytes': self.peak_bytes}

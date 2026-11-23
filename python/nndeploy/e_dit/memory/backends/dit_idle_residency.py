"""Keep the shared-card DiT rank's weights on CPU outside denoising windows."""
import torch


class DiTIdleResidency:
    def __init__(self, model, device):
        self.model, self.device = model, torch.device(device)
        tensors = list(model.named_parameters()) + list(model.named_buffers())
        if any(t.device.type != 'cpu' for _, t in tensors):
            raise ValueError('capture idle DiT weights before GPU placement')
        # detach creates independent Tensor objects without copying the existing
        # shared checkpoint storage. Module.to replaces model tensor storage;
        # this CPU snapshot survives and avoids a D2H weight copy on release.
        self.cpu = {name: tensor.detach() for name, tensor in tensors}
        self.bytes = sum(t.numel()*t.element_size() for t in self.cpu.values())
        self.active = False
        self.acquires = 0

    def acquire(self):
        if not self.active:
            try:
                self.model.to(self.device, non_blocking=True)
            except BaseException:
                self.release()
                raise
            self.active = True
            self.acquires += 1

    def release(self):
        # Module.to may replace buffer objects. Resolve every tensor through the
        # model rather than assigning stale buffer references after a transfer.
        torch.cuda.synchronize(self.device)
        current = dict(self.model.named_parameters()) | dict(self.model.named_buffers())
        if current.keys() != self.cpu.keys():
            raise RuntimeError('DiT parameters/buffers changed after residency registration')
        for name, cpu in self.cpu.items():
            current[name].data = cpu
        self.active = False

    def report(self):
        return dict(mode='shared_rank_idle_cpu', active=self.active, acquisitions=self.acquires,
                    cpu_weight_bytes=self.bytes, h2d_weight_bytes=self.bytes*self.acquires,
                    d2h_weight_bytes=0)

"""Per-forward scratch padding for the full-GEMM SP reference path."""
import torch
import torch.nn.functional as F


class ReferenceSequencePadding:
    """Reuse zero rows between sequential projections within one forward.

    The returned tensor is scratch storage: the next call may overwrite its
    local rows. Consumers must finish reading it on the calling stream before
    the next call. Linear/FFN projections return independent output storage.
    Never retain this cache across denoising forwards or windows.
    """
    def __init__(self):
        self.buffers = {}

    def clear(self):
        self.buffers.clear()

    def __call__(self, value, start, length):
        if value.ndim != 3 or not value.is_cuda or torch.is_grad_enabled():
            return F.pad(value, (0, 0, start, length - start - value.shape[-2]))
        batch, local, width = value.shape
        if start < 0 or start + local > length:
            raise ValueError('local sequence must fit the padded sequence')
        # Streams have separate scratch storage, so reuse never races a
        # projection previously queued on a different stream.
        key = (value.device, value.dtype, batch, local, width, start, length,
               torch.cuda.current_stream(value.device).cuda_stream)
        padded = self.buffers.get(key)
        if padded is None:
            padded = value.new_zeros((batch, length, width))
            self.buffers[key] = padded
        padded[:, start:start + local].copy_(value)
        return padded

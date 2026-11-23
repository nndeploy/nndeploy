"""CUDA height-shard halo exchange, with zero global boundaries."""
from threading import Barrier
import torch


class SpatialExchange:
    def __init__(self, degree):
        self.degree = degree
        self.slots = [None] * degree
        self.copied = [None] * degree
        self.barrier = Barrier(degree, timeout=120)
        self.calls = [0] * degree

    def abort(self):
        self.barrier.abort()

    def convolution(self, rank, value, original, radius):
        """Supply height padding explicitly; original must have height padding 0."""
        if not radius:
            self.calls[rank] += 1
            return original(value)
        stream = torch.cuda.current_stream(value.device)
        ready = torch.cuda.Event()
        ready.record(stream)
        self.slots[rank] = (value, ready)
        self.barrier.wait()
        try:
            height = value.shape[-2]
            shape = list(value.shape)
            shape[-2] = height + 2 * radius
            local = value.new_empty(shape)
            local[..., radius:radius + height, :].copy_(value)
            if rank:
                source, event = self.slots[rank - 1]
                stream.wait_event(event)
                local[..., :radius, :].copy_(source[..., -radius:, :], non_blocking=True)
            else:
                local[..., :radius, :].zero_()
            if rank + 1 < self.degree:
                source, event = self.slots[rank + 1]
                stream.wait_event(event)
                local[..., -radius:, :].copy_(source[..., :radius, :], non_blocking=True)
            else:
                local[..., -radius:, :].zero_()
            copied = torch.cuda.Event()
            copied.record(stream)
            self.copied[rank] = copied
            self.barrier.wait()
            # GPU-side waits protect source storage without blocking the host.
            # The next publication barrier prevents overwriting these events
            # before every consumer has enqueued its waits.
            for peer in (rank - 1, rank + 1):
                if 0 <= peer < self.degree:
                    stream.wait_event(self.copied[peer])
            self.slots[rank] = None
            result = original(local)
            self.calls[rank] += 1
            return result
        except BaseException:
            self.abort()
            raise

"""Ordered CUDA peer transport and reduction adapter."""
from threading import Barrier
import torch


class PeerExchange:
    """Ordered exchange: producers and all consumers finish before slot reuse."""
    def __init__(self, degree, timeout=120):
        self.degree = degree
        self.barrier = Barrier(degree, timeout=timeout)
        self.slots = [None] * degree
        self.calls = [0] * degree
        self.direct_copies = [0] * degree

    def abort(self):
        self.barrier.abort()

    def exchange(self, rank, values, select, dim):
        device = values[0].device
        try:
            torch.cuda.current_stream(device).synchronize()
            self.slots[rank] = values
            self.barrier.wait()
            result = []
            for k in range(len(values)):
                parts = [select(peer[k]) for peer in self.slots]
                # Large transfers can write directly into the final tensor,
                # avoiding destination temporaries followed by another cat.
                # Small transfers retain cat: strided copy launch overhead
                # outweighed the saved bandwidth on the short-window benchmark.
                size = sum(part.numel() * part.element_size() for part in parts)
                if device.type == 'cuda' and size >= 32 * 1024**2:
                    shape = list(parts[0].shape)
                    shape[dim] = sum(part.shape[dim] for part in parts)
                    output = torch.empty(shape, dtype=parts[0].dtype, device=device)
                    offset = 0
                    for part in parts:
                        output.narrow(dim, offset, part.shape[dim]).copy_(part, non_blocking=True)
                        offset += part.shape[dim]
                    result.append(output)
                    self.direct_copies[rank] += 1
                else:
                    result.append(torch.cat([part.to(device) for part in parts], dim=dim))
            torch.cuda.current_stream(device).synchronize()
            self.barrier.wait()
            self.calls[rank] += 1
            return tuple(result)
        except BaseException:
            self.abort()
            raise


    def rotate(self, rank, values):
        """One ring hop; no full-sequence K/V allocation."""
        device = values[0].device
        try:
            torch.cuda.current_stream(device).synchronize()
            self.slots[rank] = values
            self.barrier.wait()
            result = tuple(x.to(device) for x in self.slots[(rank - 1) % self.degree])
            torch.cuda.current_stream(device).synchronize()
            self.barrier.wait()
            self.calls[rank] += 1
            return result
        except BaseException:
            self.abort()
            raise


class PeerCacheCoordinator:
    """Cache consensus shares the ordered SP transport with attention."""
    def __init__(self, exchange, rank):
        self.exchange, self.rank = exchange, rank
        self.world_size = exchange.degree

    def all_reduce(self, tensor):
        values = self.exchange.exchange(self.rank, (tensor.unsqueeze(0),), lambda x: x, dim=0)[0]
        return values.sum(dim=0)

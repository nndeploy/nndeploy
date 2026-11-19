# SPDX-License-Identifier: Apache-2.0
"""Bounded, ordered spatial-tile inference with one model per CUDA process."""

import traceback
from itertools import islice

import torch
import torch.multiprocessing as mp


def _worker(connection, device, kwargs):
    try:
        from nndeploy.e_dit.algorithms.vsr.model import VSRRestorer

        torch.cuda.set_device(device)
        model = VSRRestorer.from_pretrained(device=device, **kwargs)
        connection.send(("ready", None))
        while True:
            command, window = connection.recv()
            if command == "stop":
                break
            if command == "stats":
                connection.send(("stats", dict(torch._dynamo.utils.counters["stats"])))
                continue
            if command != "restore":
                raise RuntimeError(f"Unexpected worker command: {command}")
            output = model.restore_window(window).contiguous()
            torch.cuda.synchronize()
            connection.send(("result", output))
            # CUDA IPC storage must remain alive until the owner has copied it.
            if connection.recv() != "consumed":
                raise RuntimeError("Missing CUDA IPC consumption acknowledgement")
            del output, window
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        connection.close()


class ParallelVSRRestorer:
    """Static round-robin tiles; at most one in-flight tile per worker.

    Device indices are relative to CUDA_VISIBLE_DEVICES. The first device owns
    blending and also computes tiles. Use as a context manager to release workers.
    Like VSRRestorer, this object is serial-request-only.
    """

    def __init__(self, devices, **kwargs):
        from nndeploy.e_dit.algorithms.vsr.model import VSRRestorer

        indices = [torch.device(d).index for d in devices]
        if len(indices) < 2 or None in indices or len(set(indices)) != len(indices):
            raise ValueError("Provide at least two distinct explicit CUDA devices")
        if any(torch.device(d).type != "cuda" for d in devices):
            raise ValueError("Tile parallelism requires CUDA devices")
        self.workers = []
        self.closed = False
        self.device = torch.device(devices[0])
        context = mp.get_context("spawn")
        try:
            for device in devices[1:]:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, device, kwargs))
                process.start()
                child.close()
                self.workers.append((parent, process))
            torch.cuda.set_device(self.device)
            self.local = VSRRestorer.from_pretrained(device=self.device, **kwargs)
            for connection, _ in self.workers:
                self._receive(connection, "ready")
            for name in (
                "tile_t",
                "tile_h",
                "tile_w",
                "t_overlap",
                "s_overlap",
                "dtype",
            ):
                setattr(self, name, getattr(self.local, name))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection, expected):
        if not connection.poll(1800):
            raise TimeoutError("VSR worker timed out")
        status, payload = connection.recv()
        if status != expected:
            raise RuntimeError(f"VSR worker {status}: {payload}")
        return payload

    def restore_window(self, window):
        return self.local.restore_window(window)

    @torch.no_grad()
    def restore_windows(self, windows):
        if self.closed:
            raise RuntimeError("Parallel VSR restorer is closed")
        windows = iter(windows)
        # Fetching can raise a cooperative cancellation between batches. No
        # worker is active here, so preserve the pool for the next request.
        while batch := list(islice(windows, len(self.workers) + 1)):
            try:
                inputs = [
                    w.to(self.device, dtype=self.dtype).contiguous() for w in batch
                ]
                torch.cuda.synchronize(self.device)
                active = self.workers[: len(inputs) - 1]
                for (connection, _), window in zip(active, inputs[1:]):
                    connection.send(("restore", window))
                results = [self.local.restore_window(inputs[0])]
                for connection, _ in active:
                    remote = self._receive(connection, "result")
                    result = remote.to(self.device, copy=True)
                    torch.cuda.synchronize(self.device)
                    del remote
                    connection.send("consumed")
                    results.append(result)
            except BaseException:
                self.close()
                raise
            # All IPC outputs have been acknowledged before yielding. Early
            # iterator close does not terminate the persistent worker pool.
            yield from results

    def compile_stats(self):
        for connection, _ in self.workers:
            connection.send(("stats", None))
        return [dict(torch._dynamo.utils.counters["stats"])] + [
            self._receive(connection, "stats") for connection, _ in self.workers
        ]

    def warmup(self, window, iterations=4):
        for _ in range(iterations):
            list(self.restore_windows(window for _ in range(len(self.workers) + 1)))
        torch.cuda.synchronize(self.device)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection, process in self.workers:
            if process.is_alive():
                try:
                    connection.send(("stop", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            connection.close()
        self.workers.clear()
        local = getattr(self, "local", None)
        if local is not None:
            local.close()
            self.local = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

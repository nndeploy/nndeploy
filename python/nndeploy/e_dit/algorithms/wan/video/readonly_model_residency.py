# SPDX-License-Identifier: Apache-2.0
"""Stage-scoped GPU copies of immutable inference model weights."""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch

from nndeploy.e_dit.algorithms.wan.video.videoedit_memory import (
    PinnedStagingPool,
    env_nonnegative_int,
)


class ModelResidency:
    """Explicit prefetch/wait/close lifetime for immutable CPU model weights.

    The handle owns GPU allocations and pinned staging until close. Serial
    inference only: consumers must call wait before using the rebound model.
    """

    def __init__(self, model, device, *, dtype=None, asynchronous=False):
        self.model = model
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.asynchronous = asynchronous and self.device.type == "cuda"
        self.parameters = []
        self.buffers = []
        self.gpu_tensors = []
        self.stream = None
        self.ready = None
        self.pool = None
        self.started = False

    def prefetch(self):
        if self.started:
            return self
        self.parameters = [(p, p.data) for p in self.model.parameters()]
        self.buffers = [(m, name, value) for m in self.model.modules()
                        for name, value in m._buffers.items() if value is not None]
        if any(t.device.type != "cpu" for _, t in self.parameters) or any(
            t.device.type != "cpu" for _, _, t in self.buffers
        ):
            raise ValueError("Read-only residency requires CPU originals before loading")
        self.started = True
        try:
            if self.asynchronous:
                self.stream = torch.cuda.Stream(device=self.device)
                budget = env_nonnegative_int("NNDEPLOY_DIT_WAN_VIDEOEDIT_STAGE_PINNED_MB", 64)
                if budget:
                    self.pool = PinnedStagingPool(budget * 1024**2)
                with torch.cuda.stream(self.stream):
                    self._upload()
                    self.ready = torch.cuda.Event()
                    self.ready.record(self.stream)
            else:
                self._upload()
        except BaseException:
            self.close()
            raise
        return self

    def _upload(self):
        if self.stream is not None:
            self._upload_consolidated()
            return

        def upload(original):
            dtype = self.dtype if self.dtype and original.is_floating_point() else original.dtype
            result = original.to(device=self.device, dtype=dtype)
            self.gpu_tensors.append(result)
            return result

        with torch.no_grad():
            for parameter, original in self.parameters:
                parameter.data = upload(original)
            uploaded_buffers = {}
            for module, name, original in self.buffers:
                if id(original) not in uploaded_buffers:
                    uploaded_buffers[id(original)] = upload(original)
                module._buffers[name] = uploaded_buffers[id(original)]

    def _upload_consolidated(self):
        originals = {id(t): t for _, t in self.parameters}
        originals.update({id(t): t for _, _, t in self.buffers})
        groups, uploaded = {}, {}
        with torch.no_grad():
            for key, original in originals.items():
                dtype = self.dtype if self.dtype and original.is_floating_point() else original.dtype
                if original.is_contiguous():
                    groups.setdefault((original.dtype, dtype), []).append((key, original))
                else:
                    target = torch.empty_strided(original.shape, original.stride(),
                                                 device=self.device, dtype=dtype)
                    if self.pool is not None:
                        self.pool.copy_to(target, original, self.stream)
                    else:
                        target.copy_(original, non_blocking=original.is_pinned())
                    self.gpu_tensors.append(target)
                    uploaded[key] = target
            for (_, dtype), entries in groups.items():
                flat = torch.empty(sum(t.numel() for _, t in entries), device=self.device, dtype=dtype)
                self.gpu_tensors.append(flat)
                if self.pool is not None:
                    self.pool.copy_many(flat, [t for _, t in entries], self.stream)
                offset = 0
                for key, original in entries:
                    view = flat[offset : offset + original.numel()].view(original.shape)
                    if self.pool is None:
                        view.copy_(original, non_blocking=original.is_pinned())
                    uploaded[key] = view
                    offset += original.numel()
            for parameter, original in self.parameters:
                parameter.data = uploaded[id(original)]
            for module, name, original in self.buffers:
                module._buffers[name] = uploaded[id(original)]

    def wait(self):
        self.prefetch()
        if self.ready is not None:
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(self.ready)
            for tensor in self.gpu_tensors:
                tensor.record_stream(consumer)
        return self.model

    def close(self):
        if not self.started:
            return
        try:
            if self.stream is not None:
                self.stream.synchronize()
        finally:
            with torch.no_grad():
                for parameter, original in self.parameters:
                    parameter.data = original
                for module, name, original in self.buffers:
                    module._buffers[name] = original
            self.gpu_tensors.clear()
            self.parameters.clear()
            self.buffers.clear()
            if self.pool is not None:
                self.pool.close()
            self.pool = None
            self.ready = None
            self.stream = None
            self.started = False

    def __enter__(self):
        try:
            return self.wait()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *exc):
        self.close()


class StagePrefetch:
    """Pack/copy on one worker while the preceding model computes."""

    def __init__(self, model, device):
        self.residency = ModelResidency(model, device, asynchronous=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="videoedit-h2d")

        def load():
            with torch.cuda.device(self.residency.device):
                return self.residency.prefetch()

        self.future = self.executor.submit(load)

    def __enter__(self):
        try:
            self.future.result()
            return self.residency.wait()
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            self.future.result()
        finally:
            self.executor.shutdown(wait=True)
            self.residency.close()

    def __exit__(self, *exc):
        self.close()


def close_encoder_prefetch(params):
    handle = getattr(params, "runtime_encoder_prefetch", None)
    params.runtime_encoder_prefetch = None
    if handle is not None:
        handle.close()


@contextmanager
def readonly_model_on_device(model: torch.nn.Module, device: torch.device, *, dtype=None):
    """Keep CPU originals alive; restore references without any device-to-host copy.

    Only for serial, inference-only use of a model whose parameters and registered
    buffers are read-only during the scope. Capture anew on each entry so CPU-side
    weight updates between requests are respected. No clone or second CPU model
    is created. GPU-side weight/buffer updates must not use this helper.
    """
    with ModelResidency(
        model, device, dtype=dtype,
        asynchronous=os.environ.get("NNDEPLOY_DIT_WAN_VIDEOEDIT_ASYNC_STAGE_LOAD") == "1",
    ) as loaded:
        yield loaded

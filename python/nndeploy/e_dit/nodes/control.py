"""Cooperative request cancellation at shared execution boundaries.

The historical service token keys remain part of the request contract. The
execution layer does not depend on the HTTP task store or service lifecycle.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


class RequestCancelled(RuntimeError):
    """Raised by every participating rank at the same safe boundary."""


@dataclass
class CancellationToken:
    request_id: str
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    requested_at: float | None = None
    last_checkpoint_at: float = field(default_factory=time.time)

    def request(self) -> None:
        if not self._event.is_set():
            self.requested_at = time.time()
            self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def touch(self) -> None:
        self.last_checkpoint_at = time.time()


def service_checkpoint(batch: Any, server_args: Any, *, phase: str) -> None:
    """Synchronize owner cancellation at a predefined all-rank safe point."""

    extra = getattr(batch, "extra", None)
    token = extra.get("service_cancellation_token") if isinstance(extra, dict) else None
    if token is None and server_args is not None:
        token = getattr(server_args, "_service_cancellation_token", None)
    if token is None:
        return
    cancelled = bool(getattr(token, "requested", False))
    context = getattr(server_args, "parallel_context", None)
    if context is not None and bool(getattr(context, "enabled", False)):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("service cancellation requires torch.distributed")
        group = getattr(context, "control_process_group", None)
        backend = str(
            dist.get_backend(group) if group is not None else dist.get_backend()
        ).lower()
        device = (
            torch.device("cuda", int(getattr(context, "local_rank", 0) or 0))
            if backend == "nccl"
            else torch.device("cpu")
        )
        value = torch.tensor([int(cancelled)], dtype=torch.int64, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
        cancelled = bool(value.item())
    token.touch()
    if cancelled:
        raise RequestCancelled(f"request cancelled at {phase}")


__all__ = ("CancellationToken", "RequestCancelled", "service_checkpoint")

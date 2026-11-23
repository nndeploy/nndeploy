"""Deterministic transfer and reclamation of completed videoerase windows."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch

from nndeploy.e_dit.nodes.schedule_batch import Req

_PRESERVED_EXTRA_KEYS = (
    "cached_text_embeddings",
    "runtime_object_output",
)


@dataclass(frozen=True)
class ReclaimHooks:
    collect: Callable[[], Any] = gc.collect
    synchronize: Callable[[torch.device], Any] = torch.cuda.synchronize
    empty_cache: Callable[[], Any] = torch.cuda.empty_cache


@dataclass
class PendingWindowReclaim:
    batch: Req | None
    window_key: tuple[int, int]
    transferred_cache_keys: tuple[str, ...]


def transfer_completed_window(batch: Req) -> PendingWindowReclaim:
    if not isinstance(batch, Req):
        raise TypeError("batch must be a Req")
    window_key = (
        int(
            batch.extra.get(
                "runtime_object_index",
                batch.extra.get("object_index", -1),
            )
        ),
        int(
            batch.extra.get(
                "runtime_window_index",
                batch.extra.get("window_index", -1),
            )
        ),
    )
    transferred = tuple(
        key for key in _PRESERVED_EXTRA_KEYS if key in batch.extra
    )
    preserved = {key: batch.extra[key] for key in transferred}
    batch.extra.clear()
    batch.extra.update(preserved)
    return PendingWindowReclaim(
        batch=batch,
        window_key=window_key,
        transferred_cache_keys=transferred,
    )


def reclaim_completed_window(
    pending: PendingWindowReclaim,
    *,
    device: torch.device | str,
    snapshot: Callable[..., Any] | None = None,
    hooks: ReclaimHooks | None = None,
    on_drop: Callable[[], Any] | None = None,
) -> None:
    if not isinstance(pending, PendingWindowReclaim):
        raise TypeError("pending must be a PendingWindowReclaim")
    if pending.batch is None:
        return
    using_default_hooks = hooks is None
    hooks = hooks or ReclaimHooks()
    if on_drop is not None:
        on_drop()
    batch = pending.batch
    for name in tuple(batch.__dataclass_fields__):
        if name in {"sampling_params", "extra", "metrics"}:
            continue
        value = getattr(batch, name, None)
        if isinstance(value, torch.Tensor) or isinstance(value, (list, dict, tuple)):
            setattr(batch, name, None)
    batch.extra.clear()
    pending.batch = None
    batch = None
    hooks.collect()
    target = torch.device(device)
    if not using_default_hooks or target.type == "cuda":
        hooks.synchronize(target)
        hooks.empty_cache()
    if snapshot is not None:
        snapshot(target)


__all__ = (
    "PendingWindowReclaim",
    "ReclaimHooks",
    "reclaim_completed_window",
    "transfer_completed_window",
)

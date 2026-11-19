# SPDX-License-Identifier: Apache-2.0
"""Small, explicit collectives used by the VideoEdit pipeline."""

from __future__ import annotations

from typing import Any

import torch


def broadcast_tensor_from_owner(
    group: Any,
    value: torch.Tensor | None,
    owner: int,
    name: str,
) -> torch.Tensor:
    """Broadcast one tensor from a rank expressed in group-local coordinates.

    Tensor payloads always travel through the accelerator process group.  CPU
    tensors are staged through each rank's local accelerator and restored to
    CPU after the collective.  This avoids GroupCoordinator's optional shared
    memory broadcaster, which is not initialized for sequence-parallel groups.
    """

    if group is None or group.world_size == 1:
        if value is None:
            raise RuntimeError(f"{name} owner produced no tensor")
        return value
    if not 0 <= owner < group.world_size:
        raise ValueError(
            f"{name} owner must be in [0, {group.world_size}), got {owner}"
        )

    is_owner = group.rank_in_group == owner
    metadata = None
    if is_owner:
        try:
            if value is None:
                raise RuntimeError(f"{name} owner rank {owner} produced no tensor")
            metadata = (tuple(value.shape), value.dtype, value.device.type == "cpu")
        except Exception as exc:
            metadata = f"{type(exc).__name__}: {exc}"

    metadata_list = [metadata]
    source_rank = group.ranks[owner]
    torch.distributed.broadcast_object_list(
        metadata_list, src=source_rank, group=group.cpu_group
    )
    metadata = metadata_list[0]
    if isinstance(metadata, str):
        raise RuntimeError(metadata)
    if not isinstance(metadata, tuple) or len(metadata) != 3:
        raise RuntimeError(f"{name} owner rank {owner} produced invalid metadata")
    shape, dtype, return_to_cpu = metadata

    # Agree on allocation/copy success before any rank enters the payload
    # collective. Otherwise a local OOM leaves its peers waiting in NCCL.
    wire_value = None
    local_error = None
    try:
        if is_owner:
            wire_value = value.to(device=group.device).contiguous()
        else:
            wire_value = torch.empty(shape, dtype=dtype, device=group.device)
    except Exception as exc:
        local_error = f"rank {group.rank_in_group}: {type(exc).__name__}: {exc}"
    errors = [None] * group.world_size
    torch.distributed.all_gather_object(errors, local_error, group=group.cpu_group)
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError(f"{name} payload preparation failed; " + "; ".join(failures))
    torch.distributed.broadcast(wire_value, src=source_rank, group=group.device_group)
    return wire_value.to("cpu") if return_to_cpu else wire_value

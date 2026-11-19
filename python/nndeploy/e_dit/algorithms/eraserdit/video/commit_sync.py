"""Synchronize the writer's minimal videoerase window commit patch."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.distributed.group_coordinator import GroupCoordinator
from nndeploy.e_dit.distributed.parallel_state import (
    ParallelContext,
    RuntimeGroup,
)
from nndeploy.e_dit.algorithms.eraserdit.video.sp_dispatch import (
    ActiveSPWindowContext as _ActiveCommitContext,
    WindowCommitFatalError,
    _normalize_stage_error,
    synchronize_window_runtime_boundary,
    resolve_active_window_commit_context,
)
from nndeploy.e_dit.parallel.stage_policy import synchronize_stage_error

_PROTOCOL_VERSION = 1
_PATCH_NDIM = 5
_HEADER_SIZE = 15
_MAX_PATCH_NUMEL = 2**31 - 1

_DTYPE_TO_CODE = {
    dtype: code
    for name, code in (
        ("uint8", 1),
        ("float16", 2),
        ("bfloat16", 3),
        ("float32", 4),
        ("float64", 5),
    )
    if (dtype := getattr(torch, name, None)) is not None
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


class WindowCommitPeerError(RuntimeError):
    """A peer failed before the next window commit data collective."""

    def __init__(self, peer_ranks: tuple[int, ...], *, phase: str) -> None:
        self.peer_ranks = peer_ranks
        self.phase = phase
        ranks = ", ".join(str(rank) for rank in peer_ranks)
        rank_label = "rank" if len(peer_ranks) == 1 else "ranks"
        super().__init__(
            f"Window commit {phase} failed on peer {rank_label} {ranks}"
        )


@dataclass(frozen=True)
class _PatchMetadata:
    shape: tuple[int, int, int, int, int]
    dtype: torch.dtype
    numel: int
    crop_bbox: tuple[int, int, int, int]
    patch_commit_only: bool


@dataclass(frozen=True)
class _CommitControlBuffers:
    local_error_flag: torch.Tensor
    gathered_error_flags: torch.Tensor


def _require_non_bool_int(name: str, value) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-bool int")
    return value


def _validate_patch_metadata(
    *,
    shape,
    dtype: torch.dtype,
    numel: int,
    crop_bbox,
    patch_commit_only: bool,
) -> _PatchMetadata:
    if type(shape) not in (tuple, list) or len(shape) != _PATCH_NDIM:
        raise ValueError(f"commit patch must have {_PATCH_NDIM} dimensions")
    normalized_shape = tuple(
        _require_non_bool_int(f"shape[{index}]", value)
        for index, value in enumerate(shape)
    )
    if any(value <= 0 for value in normalized_shape):
        raise ValueError("commit patch shape values must be positive")
    if normalized_shape[0] != 1 or normalized_shape[1] != 3:
        raise ValueError("commit patch shape must be [1, 3, frames, height, width]")
    if dtype not in _DTYPE_TO_CODE:
        raise TypeError(f"unsupported commit patch dtype: {dtype}")
    expected_numel = prod(normalized_shape)
    if numel != expected_numel or not 0 < numel <= _MAX_PATCH_NUMEL:
        raise ValueError(
            "commit patch numel does not match its bounded shape: "
            f"numel={numel}, expected={expected_numel}"
        )
    if type(crop_bbox) not in (tuple, list) or len(crop_bbox) != 4:
        raise ValueError("commit patch crop bbox must contain x, y, width, height")
    bbox = tuple(
        _require_non_bool_int(f"crop_bbox[{index}]", value)
        for index, value in enumerate(crop_bbox)
    )
    x, y, width, height = bbox
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("commit patch crop bbox values are invalid")
    if width != normalized_shape[-1] or height != normalized_shape[-2]:
        raise ValueError("commit patch bbox size must match patch spatial shape")
    if type(patch_commit_only) is not bool:
        raise TypeError("patch_commit_only must be a bool")
    return _PatchMetadata(
        shape=normalized_shape,
        dtype=dtype,
        numel=numel,
        crop_bbox=bbox,
        patch_commit_only=patch_commit_only,
    )


def _build_writer_header_and_payload(
    batch: Req,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch = getattr(batch, "crop_video_modified", None)
    if not isinstance(patch, torch.Tensor):
        raise TypeError("writer window crop patch must be a tensor")
    payload = patch.detach().to(device=device).contiguous()
    metadata = _validate_patch_metadata(
        shape=tuple(payload.shape),
        dtype=payload.dtype,
        numel=payload.numel(),
        crop_bbox=batch.crop_bbox,
        patch_commit_only=batch.extra.get("patch_commit_only", False),
    )
    x, y, width, height = metadata.crop_bbox
    header = torch.tensor(
        [
            _PROTOCOL_VERSION,
            _PATCH_NDIM,
            _DTYPE_TO_CODE[metadata.dtype],
            metadata.numel,
            *metadata.shape,
            x,
            y,
            width,
            height,
            int(metadata.patch_commit_only),
            0,
        ],
        dtype=torch.int64,
        device=device,
    )
    if header.numel() != _HEADER_SIZE:
        raise AssertionError("Window commit header width changed")
    return header, payload


def _decode_header(header: torch.Tensor) -> _PatchMetadata:
    if not isinstance(header, torch.Tensor):
        raise TypeError("commit patch header must be a tensor")
    if header.dtype is not torch.int64 or header.ndim != 1:
        raise TypeError("commit patch header must be a one-dimensional int64 tensor")
    if header.numel() != _HEADER_SIZE:
        raise ValueError("commit patch header has an invalid fixed width")
    values = header.detach().to(device="cpu").tolist()
    if values[0] != _PROTOCOL_VERSION:
        raise ValueError("unsupported commit patch protocol version")
    if values[1] != _PATCH_NDIM:
        raise ValueError("commit patch header ndim is invalid")
    dtype = _CODE_TO_DTYPE.get(values[2])
    if dtype is None:
        raise ValueError("commit patch header dtype code is invalid")
    if values[14] != 0:
        raise ValueError("commit patch header reserved field must be zero")
    if values[13] not in (0, 1):
        raise ValueError("commit patch header patch_commit_only flag is invalid")
    return _validate_patch_metadata(
        shape=values[4:9],
        dtype=dtype,
        numel=values[3],
        crop_bbox=values[9:13],
        patch_commit_only=bool(values[13]),
    )


def _allocate_payload(
    metadata: _PatchMetadata,
    *,
    device: torch.device,
    owner_payload: torch.Tensor | None,
) -> torch.Tensor:
    if owner_payload is not None:
        if (
            tuple(owner_payload.shape) != metadata.shape
            or owner_payload.dtype is not metadata.dtype
            or owner_payload.device != device
            or not owner_payload.is_contiguous()
            or owner_payload.numel() != metadata.numel
        ):
            raise ValueError("writer commit payload changed after header construction")
        return owner_payload
    return torch.empty(metadata.shape, dtype=metadata.dtype, device=device)


def _synchronize_local_error(
    error: BaseException | None,
    *,
    coordinator: GroupCoordinator,
    group: RuntimeGroup,
    control_buffers: _CommitControlBuffers,
    phase: str,
) -> None:
    normalized_error = _normalize_stage_error(error, phase=phase)
    control_buffers.local_error_flag.fill_(1 if normalized_error is not None else 0)
    coordinator.all_gather_into_tensor(
        control_buffers.gathered_error_flags,
        control_buffers.local_error_flag,
    )
    failed_slots = (
        control_buffers.gathered_error_flags.detach()
        .to(device="cpu")
        .ne(0)
        .nonzero()
        .flatten()
        .tolist()
    )
    if normalized_error is not None:
        raise normalized_error
    if failed_slots:
        raise WindowCommitPeerError(
            tuple(group.spec.ranks[slot] for slot in failed_slots),
            phase=phase,
        )


def _allocate_control_buffers(
    *,
    world_size: int,
    device: torch.device,
) -> _CommitControlBuffers:
    return _CommitControlBuffers(
        local_error_flag=torch.empty(1, dtype=torch.int32, device=device),
        gathered_error_flags=torch.empty(
            world_size,
            dtype=torch.int32,
            device=device,
        ),
    )


def _prepare_commit_collective(
    server_args: ServerArgs,
) -> (
    tuple[
        _ActiveCommitContext,
        GroupCoordinator,
        torch.Tensor,
        _CommitControlBuffers,
    ]
    | None
):
    active = resolve_active_window_commit_context(server_args)
    if active is None:
        return None
    coordinator = GroupCoordinator(active.world_data_group)
    header = torch.zeros(
        _HEADER_SIZE,
        dtype=torch.int64,
        device=active.control_device,
    )
    control_buffers = _allocate_control_buffers(
        world_size=coordinator.world_size,
        device=active.control_device,
    )
    return active, coordinator, header, control_buffers


def synchronize_window_commit(
    batch: Req,
    server_args: ServerArgs,
) -> Req:
        prepared_collective = None
        setup_error: BaseException | None = None
        try:
            prepared_collective = _prepare_commit_collective(server_args)
        except BaseException as error:
            setup_error = error
        setup_error = _normalize_stage_error(setup_error, phase="setup")
        parallel_context = getattr(server_args, "parallel_context", None)
        synchronize_stage_error(
            setup_error,
            parallel_context if isinstance(parallel_context, ParallelContext) else None,
        )
        if setup_error is not None:
            raise setup_error
        if prepared_collective is None:
            return batch
        active, coordinator, header, control_buffers = prepared_collective
        if batch.metrics is not None:
            batch.metrics.ensure_operation("commit_patch_broadcast")
        if bool(batch.extra.get("sp_writer_owned_runtime")):
            if not active.is_writer:
                batch.crop_video_modified = None
                batch.crop_bbox = None
                batch.output_video = None
                batch.output = None
                batch.decoded_video = None
            return batch

        payload = None
        preparation_error: BaseException | None = None
        if active.is_writer:
            try:
                header, payload = _build_writer_header_and_payload(
                    batch,
                    device=active.control_device,
                )
            except BaseException as error:
                preparation_error = error
        _synchronize_local_error(
            preparation_error,
            coordinator=coordinator,
            group=active.world_data_group,
            control_buffers=control_buffers,
            phase="header preparation",
        )
        coordinator.broadcast(header, src=active.contract.writer_rank)

        allocation_error: BaseException | None = None
        metadata = None
        try:
            metadata = _decode_header(header)
            payload = _allocate_payload(
                metadata,
                device=active.control_device,
                owner_payload=payload,
            )
        except BaseException as error:
            allocation_error = error
        _synchronize_local_error(
            allocation_error,
            coordinator=coordinator,
            group=active.world_data_group,
            control_buffers=control_buffers,
            phase="payload allocation",
        )
        assert metadata is not None and payload is not None
        coordinator.broadcast(payload, src=active.contract.writer_rank)
        if batch.metrics is not None:
            batch.metrics.record_operation("commit_patch_broadcast")

        batch.crop_video_modified = payload
        batch.crop_bbox = metadata.crop_bbox
        batch.extra["patch_commit_only"] = metadata.patch_commit_only
        batch.extra["crop_video_modified_shape"] = metadata.shape
        batch.output_video = None
        batch.output = payload
        batch.decoded_video = None
        return batch


def release_window_commit_payload(
    batch: Req,
    server_args: ServerArgs,
) -> None:
    """Drop active-P3 per-window payloads after existing commit ops consume them."""
    if resolve_active_window_commit_context(server_args) is None:
        return
    patch = getattr(batch, "crop_video_modified", None)
    if getattr(batch, "output", None) is patch:
        batch.output = None
    batch.crop_video_modified = None
    batch.output_video = None
    batch.decoded_video = None


__all__ = (
    "WindowCommitFatalError",
    "WindowCommitPeerError",
    "release_window_commit_payload",
    "resolve_active_window_commit_context",
    "synchronize_window_commit",
    "synchronize_window_runtime_boundary",
)

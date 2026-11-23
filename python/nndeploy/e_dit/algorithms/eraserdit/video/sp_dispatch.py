"""Dispatch fixed-width video erase SP window commands from the writer rank."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import torch

from nndeploy.e_dit.parallel.stage_policy import synchronize_stage_error
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.config.window_parallel import SequenceParallelContract
from nndeploy.e_dit.distributed.group_coordinator import GroupCoordinator
from nndeploy.e_dit.distributed.parallel_state import (
    ParallelContext,
    RuntimeGroup,
    resolve_group_control_device,
)

_PROTOCOL_VERSION = 1
SP_WINDOW_HEADER_SIZE = 6
_MAX_FIELD_VALUE = 2**31 - 1
_WRITER_EXCEPTION_REASON = 1
_WRITER_FATAL_REASON = 2


class SPWindowCommandCode(IntEnum):
    RUN = 1
    SKIP = 2
    END = 3
    ERROR = 4


@dataclass(frozen=True)
class SPWindowCommand:
    code: SPWindowCommandCode
    object_index: int
    window_index: int
    scene_index: int
    reason_code: int


@dataclass(frozen=True)
class ActiveSPWindowContext:
    contract: SequenceParallelContract
    parallel_context: ParallelContext
    world_data_group: RuntimeGroup
    world_control_group: RuntimeGroup
    sp_group: RuntimeGroup
    cfg_group: RuntimeGroup
    control_device: torch.device
    world_control_device: torch.device

    @property
    def is_writer(self) -> bool:
        return self.parallel_context.global_rank == self.contract.writer_rank

    @property
    def cfg_branch(self) -> str | None:
        if not self.contract.cfg_parallel_active:
            return None
        return "positive" if self.cfg_group.group_rank == 0 else "negative"


class SPWindowControlError(RuntimeError):
    """Normalize a window-command control-plane failure with rank and phase."""

    def __init__(self, *, rank: int, phase: str, original: BaseException) -> None:
        self.rank = rank
        self.phase = phase
        self.original_type = type(original).__name__
        super().__init__(
            f"SP window control failed on rank {rank} during phase {phase}: "
            f"{self.original_type}: {original}"
        )


def resolve_active_window_commit_context(
    server_args: ServerArgs,
) -> ActiveSPWindowContext | None:
    """Resolve the frozen SP/CFG mesh; never infer activity from world size."""
    contract = getattr(server_args, "sequence_parallel_contract", None)
    if contract is None:
        return None
    if not isinstance(contract, SequenceParallelContract):
        raise TypeError(
            "sequence_parallel_contract must be a frozen "
            "SequenceParallelContract"
        )
    if not contract.active:
        return None
    context = getattr(server_args, "parallel_context", None)
    if not isinstance(context, ParallelContext) or not context.enabled:
        raise RuntimeError(
            "active sequence parallel requires an enabled ParallelContext"
        )
    plan = context.plan
    frozen_values = (
        ("world_size", plan.world_size, contract.world_size),
        ("sp_degree", plan.sp_degree, contract.sp_degree),
        ("cfg_degree", plan.cfg_degree, contract.cfg_degree),
        ("vae_degree", plan.vae_degree, contract.vae_degree),
        ("writer_rank", plan.writer_rank, contract.writer_rank),
    )
    for name, actual, expected in frozen_values:
        if actual != expected:
            raise RuntimeError(
                f"parallel context {name} changed after  capability "
                f"freeze: expected {expected}, got {actual}"
            )
    if contract.distributed_compute_mode != "entry_only":
        raise RuntimeError(
            "active commit sync requires distributed_compute_mode='entry_only'"
        )

    world_data_group = context.world_data_group()
    world_control_group = context.world_control_group()
    sp_group = context.current_group("sp_")
    cfg_group = context.current_group("cfg_")
    groups = {
        "world data": world_data_group,
        "world control": world_control_group,
        "SP": sp_group,
        "CFG": cfg_group,
    }
    for name, group in groups.items():
        if not isinstance(group, RuntimeGroup) or not group.is_member:
            raise RuntimeError(f"current {name} group must be a member RuntimeGroup")
        if group.global_rank != context.global_rank:
            raise RuntimeError(
                f"{name} group slot must match parallel context global_rank"
            )
    expected_world_ranks = tuple(range(contract.world_size))
    if world_data_group.spec.ranks != expected_world_ranks:
        raise RuntimeError("world data group must contain every global rank in order")
    if world_control_group.spec.ranks != expected_world_ranks:
        raise RuntimeError("world control group must contain every global rank in order")
    if world_data_group.owns_process_group or world_control_group.owns_process_group:
        raise RuntimeError("world runtime groups must not own process-group lifecycle")
    if sp_group.world_size != contract.sp_degree:
        raise RuntimeError("SP group world_size must match frozen sp_degree")
    if cfg_group.world_size != contract.cfg_degree:
        raise RuntimeError("CFG group world_size must match frozen cfg_degree")
    if context.topology is None:
        raise RuntimeError("active mesh requires a frozen parallel topology")
    if sp_group.spec not in context.topology.sp_groups:
        raise RuntimeError("current SP group is not part of the frozen topology")
    if cfg_group.spec not in context.topology.cfg_groups:
        raise RuntimeError("current CFG group is not part of the frozen topology")
    if context.topology.sp_groups[0].ranks[0] != contract.writer_rank:
        raise RuntimeError(" writer must own positive sequence shard zero")
    return ActiveSPWindowContext(
        contract=contract,
        parallel_context=context,
        world_data_group=world_data_group,
        world_control_group=world_control_group,
        sp_group=sp_group,
        cfg_group=cfg_group,
        control_device=resolve_group_control_device(
            world_data_group,
            local_rank=context.local_rank,
            fallback_device=server_args.device,
        ),
        world_control_device=resolve_group_control_device(
            world_control_group,
            local_rank=context.local_rank,
            fallback_device="cpu",
        ),
    )


def _require_field(name: str, value: int, *, allow_end: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-bool int")
    minimum = -1 if allow_end else 0
    if not minimum <= value <= _MAX_FIELD_VALUE:
        raise ValueError(
            f"{name} must be between {minimum} and {_MAX_FIELD_VALUE}"
        )
    return value


def _validate_command(command: SPWindowCommand) -> SPWindowCommand:
    if not isinstance(command, SPWindowCommand):
        raise TypeError("window command must be SPWindowCommand")
    if not isinstance(command.code, SPWindowCommandCode):
        raise TypeError("window command code must be SPWindowCommandCode")

    terminal = command.code in (
        SPWindowCommandCode.END,
        SPWindowCommandCode.ERROR,
    )
    object_index = _require_field(
        "object_index", command.object_index, allow_end=terminal
    )
    window_index = _require_field(
        "window_index", command.window_index, allow_end=terminal
    )
    scene_index = _require_field(
        "scene_index", command.scene_index, allow_end=terminal
    )
    reason_code = _require_field("reason_code", command.reason_code)

    if terminal and (object_index, window_index, scene_index) != (-1, -1, -1):
        raise ValueError("END and ERROR command indexes must all be -1")
    if command.code in (
        SPWindowCommandCode.RUN,
        SPWindowCommandCode.END,
    ) and reason_code != 0:
        raise ValueError("RUN and END reason_code must be zero")
    if command.code is SPWindowCommandCode.SKIP and reason_code == 0:
        raise ValueError("SKIP reason_code must be positive")
    if command.code is SPWindowCommandCode.ERROR and reason_code not in (
        _WRITER_EXCEPTION_REASON,
        _WRITER_FATAL_REASON,
    ):
        raise ValueError("ERROR reason_code is invalid")
    return command


def encode_sp_window_command(
    command: SPWindowCommand,
    *,
    device: torch.device,
) -> torch.Tensor:
    command = _validate_command(command)
    header = torch.tensor(
        [
            _PROTOCOL_VERSION,
            int(command.code),
            command.object_index,
            command.window_index,
            command.scene_index,
            command.reason_code,
        ],
        dtype=torch.int64,
        device=device,
    )
    if header.numel() != SP_WINDOW_HEADER_SIZE:
        raise AssertionError("SP window header width changed")
    return header


def decode_sp_window_command(
    header: torch.Tensor,
) -> SPWindowCommand:
    if not isinstance(header, torch.Tensor):
        raise TypeError("window command header must be a tensor")
    if header.dtype is not torch.int64 or header.ndim != 1:
        raise TypeError("window command header must be a one-dimensional int64 tensor")
    if header.numel() != SP_WINDOW_HEADER_SIZE:
        raise ValueError("window command header has an invalid fixed width")
    values = header.detach().to(device="cpu").tolist()
    if values[0] != _PROTOCOL_VERSION:
        raise ValueError("unsupported window command protocol version")
    try:
        code = SPWindowCommandCode(values[1])
    except ValueError as error:
        raise ValueError("window command code is invalid") from error
    return _validate_command(
        SPWindowCommand(
            code=code,
            object_index=values[2],
            window_index=values[3],
            scene_index=values[4],
            reason_code=values[5],
        )
    )


class WindowCommitFatalError(RuntimeError):
    """Normalize a non-Exception failure before the executor stage boundary."""

    def __init__(self, *, phase: str, original: BaseException) -> None:
        self.phase = phase
        self.original_type = type(original).__name__
        super().__init__(
            f"Window commit {phase} raised non-Exception "
            f"{self.original_type}: {original}"
        )


def _normalize_stage_error(
    error: BaseException | None,
    *,
    phase: str,
) -> Exception | None:
    if error is None or isinstance(error, Exception):
        return error
    try:
        raise WindowCommitFatalError(phase=phase, original=error) from error
    except WindowCommitFatalError as normalized_error:
        return normalized_error


def synchronize_window_runtime_boundary(
    error: BaseException | None,
    server_args: ServerArgs,
) -> None:
    """Keep distributed ranks aligned after writer-only window lifecycle work."""
    normalized_error = _normalize_stage_error(error, phase="runtime boundary")
    parallel_context = getattr(server_args, "parallel_context", None)
    synchronize_stage_error(
        normalized_error,
        parallel_context if isinstance(parallel_context, ParallelContext) else None,
    )


def _error_command(error: BaseException) -> SPWindowCommand:
    reason = (
        _WRITER_EXCEPTION_REASON
        if isinstance(error, Exception)
        else _WRITER_FATAL_REASON
    )
    return SPWindowCommand(
        code=SPWindowCommandCode.ERROR,
        object_index=-1,
        window_index=-1,
        scene_index=-1,
        reason_code=reason,
    )


def dispatch_sp_window_command(
    server_args: ServerArgs,
    *,
    runtime_mode: str,
    prepare_writer_command: Callable[[], SPWindowCommand] | None,
    phase: str,
) -> SPWindowCommand | None:
    """Broadcast exactly one fixed-width header for an active SP window command."""
    if runtime_mode != "windowed_streaming":
        return None
    active = resolve_active_window_commit_context(server_args)
    if active is None:
        return None
    if active.is_writer and not callable(prepare_writer_command):
        raise RuntimeError("writer rank must prepare the SP window writer command")
    if not active.is_writer and prepare_writer_command is not None:
        raise RuntimeError("peer rank cannot construct the SP window writer command")

    preparation_error: BaseException | None = None
    if active.is_writer:
        try:
            command = prepare_writer_command()
            header = encode_sp_window_command(
                command,
                device=active.control_device,
            )
        except BaseException as error:
            preparation_error = error
            header = encode_sp_window_command(
                _error_command(error),
                device=active.control_device,
            )
    else:
        header = torch.zeros(
            SP_WINDOW_HEADER_SIZE,
            dtype=torch.int64,
            device=active.control_device,
        )

    try:
        coordinator = GroupCoordinator(active.world_data_group)
        coordinator.broadcast(header, src=active.contract.writer_rank)
    except BaseException as error:
        raise SPWindowControlError(
            rank=active.parallel_context.global_rank,
            phase=phase,
            original=error,
        ) from error

    decode_error: BaseException | None = None
    command = None
    try:
        command = decode_sp_window_command(header)
    except BaseException as error:
        decode_error = error
    if decode_error is not None:
        synchronize_window_runtime_boundary(decode_error, server_args)
        raise decode_error
    assert command is not None

    if command.code is SPWindowCommandCode.ERROR:
        synchronize_window_runtime_boundary(preparation_error, server_args)
        raise SPWindowControlError(
            rank=active.parallel_context.global_rank,
            phase=phase,
            original=RuntimeError(
                f"writer command preparation failed with reason {command.reason_code}"
            ),
        )
    return command


__all__ = (
    "ActiveSPWindowContext",
    "SPWindowCommand",
    "SPWindowCommandCode",
    "SPWindowControlError",
    "SP_WINDOW_HEADER_SIZE",
    "decode_sp_window_command",
    "dispatch_sp_window_command",
    "encode_sp_window_command",
    "resolve_active_window_commit_context",
    "synchronize_window_runtime_boundary",
)

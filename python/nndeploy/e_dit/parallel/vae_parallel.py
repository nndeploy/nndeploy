"""Model-agnostic contracts for owner-only VAE parallel execution."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nndeploy.e_dit.distributed.parallel_state import RuntimeGroup


VAE_PLAN_SCHEMA_VERSION = 1


class VAEOperation(str, Enum):
    ENCODE = "encode"
    DECODE = "decode"


@dataclass(frozen=True)
class VAETaskSpec:
    task_id: int
    assigned_rank: int
    input_slices: tuple[tuple[int, int], ...]
    valid_output_slices: tuple[tuple[int, int], ...]
    padded_input_shape: tuple[int, ...]
    padded_output_shape: tuple[int, ...]
    estimated_cost: int


@dataclass(frozen=True)
class VAETaskPlan:
    operation: VAEOperation
    owner_rank: int
    global_input_shape: tuple[int, ...]
    global_output_shape: tuple[int, ...]
    tasks: tuple[VAETaskSpec, ...]
    rounds: int


@dataclass(frozen=True)
class VAEParallelBinding:
    global_rank: int
    owner_rank: int
    vae_group: RuntimeGroup
    world_control_group: RuntimeGroup
    data_coordinator: Any
    control_coordinator: Any

    def __post_init__(self) -> None:
        from nndeploy.e_dit.distributed.parallel_state import RuntimeGroup

        _require_non_negative_int("global_rank", self.global_rank)
        _require_non_negative_int("owner_rank", self.owner_rank)
        for name in ("vae_group", "world_control_group"):
            group = getattr(self, name)
            if not isinstance(group, RuntimeGroup):
                raise TypeError(f"{name} must be a RuntimeGroup")
            if not group.is_member or group.global_rank != self.global_rank:
                raise ValueError(f"{name} must contain global_rank")
        if self.owner_rank not in self.vae_group.spec.ranks:
            raise ValueError("owner_rank must belong to vae_group")
        _validate_coordinator_group(
            "data_coordinator",
            self.data_coordinator,
            expected_group=self.vae_group,
        )
        _validate_coordinator_group(
            "control_coordinator",
            self.control_coordinator,
            expected_group=self.world_control_group,
        )

    @property
    def is_owner(self) -> bool:
        return self.global_rank == self.owner_rank


class VAEParallelPeerError(RuntimeError):
    """Report failures observed at a fixed VAE-parallel phase."""

    def __init__(self, peer_ranks: tuple[int, ...], *, phase: str) -> None:
        self.peer_ranks = peer_ranks
        self.phase = phase
        ranks = ", ".join(str(rank) for rank in peer_ranks)
        super().__init__(f"VAE parallel {phase} failed on rank(s): {ranks}")


@dataclass(frozen=True)
class VAEParallelMetrics:
    plan_hash: str
    operation: str
    rank: int
    owner_rank: int
    global_input_shape: tuple[int, ...]
    global_output_shape: tuple[int, ...]
    global_input_bytes: int
    global_output_bytes: int
    task_count: int
    local_task_ids: tuple[int, ...]
    local_task_estimated_costs: tuple[tuple[int, int], ...]
    active_rounds: int
    input_envelope_shapes: tuple[tuple[int, ...], ...]
    output_envelope_shapes: tuple[tuple[int, ...], ...]
    valid_input_bytes: int
    padded_input_bytes: int
    valid_output_bytes: int
    padded_output_bytes: int
    phase_seconds: tuple[tuple[str, float], ...]
    collective_counts: tuple[tuple[str, int], ...]
    memory_snapshots: tuple[tuple[str, int, int], ...]
    peak_memory_allocated_bytes: int
    peak_memory_reserved_bytes: int
    owns_full_input: bool
    owns_full_output: bool
    published_video: bool
    execution_signature: tuple[tuple[int, ...], ...]
    error_phase: str | None
    teardown_status: str
    resolved_degree: int = 1
    effective_degree: int = 1
    fallback_reason: str | None = None
    max_inflight_tiles: int = 1
    max_owner_input_tiles: int = 0
    max_owner_return_tiles: int = 0
    max_worker_output_tiles: int = 0
    max_merge_band_bytes: int = 0
    merge_order: tuple[int, ...] = ()
    p2p_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class VAEParallelResult:
    output: torch.Tensor | None
    metrics: VAEParallelMetrics


def tasks_for_round(
    plan: VAETaskPlan,
    *,
    group_ranks: tuple[int, ...],
    round_index: int,
) -> tuple[VAETaskSpec | None, ...]:
    validate_vae_task_plan(plan)
    if type(group_ranks) is not tuple or not group_ranks:
        raise ValueError("group_ranks must be a non-empty tuple")
    if any(type(rank) is not int or rank < 0 for rank in group_ranks):
        raise ValueError("group_ranks must contain non-negative plain ints")
    if len(set(group_ranks)) != len(group_ranks):
        raise ValueError("group_ranks must be unique")
    if not 0 <= round_index < plan.rounds:
        raise ValueError("round_index is outside the plan")
    slots: list[VAETaskSpec | None] = []
    for rank in group_ranks:
        tasks = tuple(task for task in plan.tasks if task.assigned_rank == rank)
        slots.append(tasks[round_index] if round_index < len(tasks) else None)
    return tuple(slots)


def _vae_transfer_tag(
    round_index: int,
    slot_index: int,
    group_size: int,
    *,
    output: bool,
) -> int:
    return (round_index * group_size * 2) + slot_index + (
        group_size if output else 0
    )


def _synchronize_p2p_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release_owner_input_cache(device: torch.device) -> None:
    """Return dead owner-only input blocks before the local VAE tile runs."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _release_worker_decode_cache(
    device: torch.device,
    *,
    operation: VAEOperation,
    is_owner: bool,
) -> None:
    """Drop stale denoise cache once before a peer starts parallel decode."""
    if (
        device.type == "cuda"
        and operation is VAEOperation.DECODE
        and not is_owner
    ):
        torch.cuda.empty_cache()


class VAEParallelEngine:
    """Execute model-specific VAE tiles through tensor-only collectives."""

    _PHASES = (
        "plan",
        "materialize",
        "scatter",
        "execute",
        "gather",
        "merge",
        "drain",
    )

    def __init__(
        self,
        binding: VAEParallelBinding,
        *,
        control_device: torch.device | str = "cpu",
        max_inflight_tiles: int = 1,
    ) -> None:
        if not isinstance(binding, VAEParallelBinding):
            raise TypeError("binding must be a VAEParallelBinding")
        self.binding = binding
        self.control_device = torch.device(control_device)
        if max_inflight_tiles not in (1, 2):
            raise ValueError("max_inflight_tiles must be 1 or 2")
        self.max_inflight_tiles = max_inflight_tiles
        _require_coordinator_methods(
            "data_coordinator",
            binding.data_coordinator,
            ("broadcast", "isend_tensor", "irecv_tensor", "wait_tensor_transfer"),
        )
        _require_coordinator_methods(
            "control_coordinator",
            binding.control_coordinator,
            ("all_gather_into_tensor",),
        )

    def execute(
        self,
        plan: VAETaskPlan | None,
        *,
        plan_metadata_size: int | None = None,
        full_input: torch.Tensor | None,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        device: torch.device | str,
        materialize_input: Any,
        execute_local: Any,
        merge_round: Any,
    ) -> VAEParallelResult:
        device = torch.device(device)
        phase_seconds = {phase: 0.0 for phase in self._PHASES}
        memory_snapshots = {
            "stage_entry": _memory_snapshot(device),
            "before_local_execute": (0, 0),
            "after_local_execute": (0, 0),
            "stage_exit": (0, 0),
        }
        local_error: BaseException | None = None
        metadata: torch.Tensor | None = None
        started = time.perf_counter()
        try:
            _validate_tensor_factory("input_dtype", input_dtype, device=device)
            _validate_tensor_factory("output_dtype", output_dtype, device=device)
            for name, callback in (
                ("materialize_input", materialize_input),
                ("execute_local", execute_local),
                ("merge_round", merge_round),
            ):
                if not callable(callback):
                    raise TypeError(f"{name} must be callable")
            if plan is None:
                if self.binding.is_owner:
                    raise ValueError("owner must provide the VAE task plan")
                if full_input is not None:
                    raise ValueError("non-owner full_input must be None")
                _require_positive_int("plan_metadata_size", plan_metadata_size)
                metadata = torch.empty(
                    plan_metadata_size,
                    dtype=torch.int64,
                    device=device,
                )
            else:
                validate_vae_task_plan(plan)
                self._validate_plan_membership(plan)
                self._validate_full_input(
                    plan,
                    full_input=full_input,
                    input_dtype=input_dtype,
                    device=device,
                )
                metadata = encode_vae_task_plan(plan).to(device=device)
                if plan_metadata_size is not None:
                    _require_positive_int("plan_metadata_size", plan_metadata_size)
                    if metadata.numel() != plan_metadata_size:
                        raise ValueError(
                            "plan_metadata_size must match encoded VAE plan"
                        )
        except BaseException as error:
            local_error = error
        phase_seconds["plan"] += time.perf_counter() - started
        self.synchronize_phase_error(local_error, phase="plan")
        assert metadata is not None

        self._synchronize_execution_signature(metadata, has_local_plan=plan is not None)
        broadcast_error: BaseException | None = None
        try:
            self.binding.data_coordinator.broadcast(
                metadata,
                src=self.binding.owner_rank,
            )
            broadcast_plan = decode_vae_task_plan(metadata)
            self._validate_plan_membership(broadcast_plan)
            if plan is not None and broadcast_plan != plan:
                raise ValueError("broadcast VAE plan does not match local plan")
            plan = broadcast_plan
        except BaseException as error:
            broadcast_error = error

        self.synchronize_phase_error(broadcast_error, phase="plan")
        plan_hash = stable_vae_plan_hash(metadata)
        assert plan is not None

        group_ranks = self.binding.vae_group.spec.ranks
        execution_signature = tuple(
            tuple(
                task.task_id if task is not None else -1
                for task in tasks_for_round(
                    plan,
                    group_ranks=group_ranks,
                    round_index=round_index,
                )
            )
            for round_index in range(plan.rounds)
        )
        local_slot = self.binding.vae_group.group_rank
        local_task_ids: list[int] = []
        local_task_estimated_costs: list[tuple[int, int]] = []
        input_envelopes: list[tuple[int, ...]] = []
        output_envelopes: list[tuple[int, ...]] = []
        valid_input_bytes = 0
        padded_input_bytes = 0
        valid_output_bytes = 0
        padded_output_bytes = 0
        merged_output: torch.Tensor | None = None
        owns_full_input = self.binding.is_owner and full_input is not None

        for round_index in range(plan.rounds):
            slots = tasks_for_round(
                plan,
                group_ranks=group_ranks,
                round_index=round_index,
            )
            input_envelope = _round_envelope(slots, field="padded_input_shape")
            output_envelope = _round_envelope(slots, field="padded_output_shape")
            input_envelopes.append(input_envelope)
            output_envelopes.append(output_envelope)
            padded_input_bytes += _tensor_bytes(input_envelope, input_dtype)
            padded_output_bytes += _tensor_bytes(output_envelope, output_dtype)
            local_task = slots[local_slot]
            if local_task is not None:
                local_task_ids.append(local_task.task_id)
                local_task_estimated_costs.append(
                    (local_task.task_id, local_task.estimated_cost)
                )
                valid_input_bytes += _slice_bytes(
                    local_task.input_slices,
                    input_dtype,
                )
                valid_output_bytes += _slice_bytes(
                    local_task.valid_output_slices,
                    output_dtype,
                )

            local_input: torch.Tensor | None = None
            for slot_index, task in enumerate(slots):
                if task is None:
                    continue
                prepared: torch.Tensor | None = None
                action_started = time.perf_counter()
                materialize_error: BaseException | None = None
                try:
                    if self.binding.is_owner:
                        assert full_input is not None
                        materialized = materialize_input(full_input, task)
                        prepared = _pad_tensor(
                            materialized,
                            target_shape=input_envelope,
                            maximum_shape=task.padded_input_shape,
                            expected_dtype=input_dtype,
                            expected_device=device,
                            name="materialize_input output",
                        )
                except BaseException as error:
                    materialize_error = error
                phase_seconds["materialize"] += time.perf_counter() - action_started
                self.synchronize_phase_error(
                    materialize_error,
                    phase="materialize",
                )

                transfer_error: BaseException | None = None
                action_started = time.perf_counter()
                try:
                    if task.assigned_rank == self.binding.owner_rank:
                        if self.binding.is_owner:
                            local_input = prepared
                    else:
                        tag = _vae_transfer_tag(
                            round_index,
                            slot_index,
                            len(group_ranks),
                            output=False,
                        )
                        if self.binding.is_owner:
                            assert prepared is not None
                            work = self.binding.data_coordinator.isend_tensor(
                                prepared,
                                dst=task.assigned_rank,
                                tag=tag,
                            )
                            self.binding.data_coordinator.wait_tensor_transfer(work)
                        elif self.binding.global_rank == task.assigned_rank:
                            local_input = torch.empty(
                                input_envelope,
                                dtype=input_dtype,
                                device=device,
                            )
                            work = self.binding.data_coordinator.irecv_tensor(
                                local_input,
                                src=self.binding.owner_rank,
                                tag=tag,
                            )
                            self.binding.data_coordinator.wait_tensor_transfer(work)
                except BaseException as error:
                    transfer_error = error
                phase_seconds["scatter"] += time.perf_counter() - action_started
                self.synchronize_phase_error(transfer_error, phase="scatter")
                prepared = None
                materialized = None

            if local_input is None:
                local_input = torch.zeros(
                    input_envelope,
                    dtype=input_dtype,
                    device=device,
                )
            if round_index + 1 == plan.rounds:
                full_input = None
                if self.binding.is_owner:
                    _release_owner_input_cache(device)
            if round_index == 0:
                _release_worker_decode_cache(
                    device,
                    operation=plan.operation,
                    is_owner=self.binding.is_owner,
                )

            local_output = torch.zeros(
                output_envelope,
                dtype=output_dtype,
                device=device,
            )
            memory_snapshots["before_local_execute"] = _max_memory_snapshot(
                memory_snapshots["before_local_execute"],
                _memory_snapshot(device),
            )
            execute_error: BaseException | None = None
            local_view: torch.Tensor | None = None
            executed: torch.Tensor | None = None
            action_started = time.perf_counter()
            try:
                if local_task is not None:
                    local_view = _origin_crop(
                        local_input,
                        local_task.padded_input_shape,
                    )
                    executed = execute_local(local_view, local_task)
                    local_output = _pad_tensor(
                        executed,
                        target_shape=output_envelope,
                        maximum_shape=local_task.padded_output_shape,
                        expected_dtype=output_dtype,
                        expected_device=device,
                        name="execute_local output",
                    )
            except BaseException as error:
                execute_error = error
            phase_seconds["execute"] += time.perf_counter() - action_started
            memory_snapshots["after_local_execute"] = _max_memory_snapshot(
                memory_snapshots["after_local_execute"],
                _memory_snapshot(device),
            )
            local_input = None
            local_view = None
            executed = None
            self.synchronize_phase_error(execute_error, phase="execute")

            for slot_index, task in sorted(
                (
                    (index, task)
                    for index, task in enumerate(slots)
                    if task is not None
                ),
                key=lambda item: item[1].task_id,
            ):
                received: torch.Tensor | None = None
                gather_error: BaseException | None = None
                action_started = time.perf_counter()
                try:
                    if task.assigned_rank == self.binding.owner_rank:
                        if self.binding.is_owner:
                            received = local_output
                    else:
                        tag = _vae_transfer_tag(
                            round_index,
                            slot_index,
                            len(group_ranks),
                            output=True,
                        )
                        if self.binding.is_owner:
                            received = torch.empty(
                                output_envelope,
                                dtype=output_dtype,
                                device=device,
                            )
                            work = self.binding.data_coordinator.irecv_tensor(
                                received,
                                src=task.assigned_rank,
                                tag=tag,
                            )
                            self.binding.data_coordinator.wait_tensor_transfer(work)
                        elif self.binding.global_rank == task.assigned_rank:
                            work = self.binding.data_coordinator.isend_tensor(
                                local_output,
                                dst=self.binding.owner_rank,
                                tag=tag,
                            )
                            self.binding.data_coordinator.wait_tensor_transfer(work)
                except BaseException as error:
                    gather_error = error
                phase_seconds["gather"] += time.perf_counter() - action_started
                self.synchronize_phase_error(gather_error, phase="gather")

                merge_error: BaseException | None = None
                action_started = time.perf_counter()
                try:
                    if self.binding.is_owner:
                        assert received is not None
                        cropped = _origin_crop(
                            received,
                            task.padded_output_shape,
                        )
                        merged_output = merge_round(
                            merged_output,
                            ((task, cropped),),
                        )
                        if not isinstance(merged_output, torch.Tensor):
                            raise TypeError(
                                "merge_round must return a torch.Tensor on owner"
                            )
                except BaseException as error:
                    merge_error = error
                phase_seconds["merge"] += time.perf_counter() - action_started
                self.synchronize_phase_error(merge_error, phase="merge")
                cropped = None
                received = None
                work = None
            local_output = None

        drain_error: BaseException | None = None
        action_started = time.perf_counter()
        try:
            _synchronize_p2p_device(device)
        except BaseException as error:
            drain_error = error
        phase_seconds["drain"] += time.perf_counter() - action_started
        self.synchronize_phase_error(drain_error, phase="drain")

        memory_snapshots["stage_exit"] = _memory_snapshot(device)
        ordered_memory_snapshots = tuple(
            (name, *memory_snapshots[name])
            for name in (
                "stage_entry",
                "before_local_execute",
                "after_local_execute",
                "stage_exit",
            )
        )
        metrics = VAEParallelMetrics(
            plan_hash=plan_hash,
            operation=plan.operation.value,
            rank=self.binding.global_rank,
            owner_rank=self.binding.owner_rank,
            global_input_shape=plan.global_input_shape,
            global_output_shape=plan.global_output_shape,
            global_input_bytes=_tensor_bytes(plan.global_input_shape, input_dtype),
            global_output_bytes=_tensor_bytes(plan.global_output_shape, output_dtype),
            task_count=len(plan.tasks),
            local_task_ids=tuple(local_task_ids),
            local_task_estimated_costs=tuple(local_task_estimated_costs),
            active_rounds=len(local_task_ids),
            input_envelope_shapes=tuple(input_envelopes),
            output_envelope_shapes=tuple(output_envelopes),
            valid_input_bytes=valid_input_bytes,
            padded_input_bytes=padded_input_bytes,
            valid_output_bytes=valid_output_bytes,
            padded_output_bytes=padded_output_bytes,
            phase_seconds=tuple(
                (phase, phase_seconds[phase]) for phase in self._PHASES
            ),
            collective_counts=(
                ("broadcast", 1),
                ("p2p_input", len(plan.tasks) - sum(
                    task.assigned_rank == self.binding.owner_rank
                    for task in plan.tasks
                )),
                ("p2p_output", len(plan.tasks) - sum(
                    task.assigned_rank == self.binding.owner_rank
                    for task in plan.tasks
                )),
                ("control_all_gather", 4 + (4 * len(plan.tasks)) + plan.rounds),
            ),
            memory_snapshots=ordered_memory_snapshots,
            peak_memory_allocated_bytes=max(
                allocated for _, allocated, _ in ordered_memory_snapshots
            ),
            peak_memory_reserved_bytes=max(
                reserved for _, _, reserved in ordered_memory_snapshots
            ),
            owns_full_input=owns_full_input,
            owns_full_output=self.binding.is_owner and merged_output is not None,
            published_video=False,
            execution_signature=execution_signature,
            error_phase=None,
            teardown_status="complete",
            resolved_degree=len(group_ranks),
            effective_degree=len(group_ranks),
            max_inflight_tiles=self.max_inflight_tiles,
            max_owner_input_tiles=1 if self.binding.is_owner else 0,
            max_owner_return_tiles=1 if self.binding.is_owner else 0,
            max_worker_output_tiles=0 if self.binding.is_owner else 1,
            merge_order=tuple(task.task_id for task in plan.tasks),
            p2p_counts=(
                ("input", len(plan.tasks) - sum(
                    task.assigned_rank == self.binding.owner_rank
                    for task in plan.tasks
                )),
                ("output", len(plan.tasks) - sum(
                    task.assigned_rank == self.binding.owner_rank
                    for task in plan.tasks
                )),
            ),
        )
        return VAEParallelResult(
            output=merged_output if self.binding.is_owner else None,
            metrics=metrics,
        )

    def synchronize_phase_error(
        self,
        local_error: BaseException | None,
        *,
        phase: str,
    ) -> None:
        if phase not in self._PHASES:
            raise ValueError(f"unknown VAE parallel error phase: {phase}")
        world_size = self.binding.world_control_group.world_size
        local_flag = torch.tensor(
            [1 if local_error is not None else 0],
            dtype=torch.int32,
            device=self.control_device,
        )
        gathered_flags = torch.empty(
            world_size,
            dtype=torch.int32,
            device=self.control_device,
        )
        self.binding.control_coordinator.all_gather_into_tensor(
            gathered_flags,
            local_flag,
        )
        failed_slots = gathered_flags.ne(0).nonzero().flatten().tolist()
        if failed_slots:
            failed_ranks = tuple(
                self.binding.world_control_group.spec.ranks[slot]
                for slot in failed_slots
            )
            error = VAEParallelPeerError(failed_ranks, phase=phase)
            if local_error is not None:
                raise error from local_error
            raise error
        if local_error is not None:
            raise RuntimeError(
                "world error flags omitted the local VAE-parallel failure"
            ) from local_error

    def _synchronize_execution_signature(
        self,
        metadata: torch.Tensor,
        *,
        has_local_plan: bool,
    ) -> None:
        digest_value = 0
        if has_local_plan:
            digest = hashlib.sha256(
                metadata.detach().to(device="cpu").contiguous().numpy().tobytes()
            ).digest()
            digest_value = int.from_bytes(
                digest[:8],
                byteorder="little",
                signed=True,
            )
        signature = torch.tensor(
            [
                int(has_local_plan),
                metadata.numel(),
                digest_value,
            ],
            dtype=torch.int64,
            device=self.control_device,
        )
        world_size = self.binding.world_control_group.world_size
        gathered = torch.empty(
            world_size * signature.numel(),
            dtype=torch.int64,
            device=self.control_device,
        )
        self.binding.control_coordinator.all_gather_into_tensor(gathered, signature)
        signatures = gathered.reshape(world_size, signature.numel())
        owner_slot = self.binding.world_control_group.spec.ranks.index(
            self.binding.owner_rank
        )
        owner_signature = signatures[owner_slot]
        same_lengths = signatures[:, 1].eq(owner_signature[1]).all().item()
        supplied_plans_match = all(
            item[0].item() == 0 or item[2].item() == owner_signature[2].item()
            for item in signatures
        )
        if (
            owner_signature[0].item() != 1
            or not same_lengths
            or not supplied_plans_match
        ):
            raise VAEParallelPeerError(
                self.binding.world_control_group.spec.ranks,
                phase="plan",
            )

    def _validate_plan_membership(self, plan: VAETaskPlan) -> None:
        ranks = self.binding.vae_group.spec.ranks
        if plan.owner_rank != self.binding.owner_rank:
            raise ValueError("plan owner_rank must match binding owner_rank")
        invalid_ranks = tuple(
            task.assigned_rank for task in plan.tasks if task.assigned_rank not in ranks
        )
        if invalid_ranks:
            raise ValueError("assigned_rank must belong to the VAE group")

    def _validate_full_input(
        self,
        plan: VAETaskPlan,
        *,
        full_input: torch.Tensor | None,
        input_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if self.binding.is_owner:
            if not isinstance(full_input, torch.Tensor):
                raise ValueError("full_input is required only on the owner")
            if tuple(full_input.shape) != plan.global_input_shape:
                raise ValueError("full_input shape must match global_input_shape")
            if full_input.dtype != input_dtype:
                raise ValueError("full_input dtype must match input_dtype")
            if full_input.device != device:
                raise ValueError("full_input device must match execution device")
        elif full_input is not None:
            raise ValueError("non-owner full_input must be None")


def validate_vae_task_plan(plan: VAETaskPlan) -> VAETaskPlan:
    if not isinstance(plan, VAETaskPlan):
        raise TypeError("plan must be a VAETaskPlan")
    if not isinstance(plan.operation, VAEOperation):
        raise TypeError("operation must be a VAEOperation")
    _require_non_negative_int("owner_rank", plan.owner_rank)
    _validate_shape("global_input_shape", plan.global_input_shape)
    _validate_shape("global_output_shape", plan.global_output_shape)
    if type(plan.tasks) is not tuple or not plan.tasks:
        raise ValueError("tasks must be a non-empty tuple")

    task_ids: list[int] = []
    tasks_per_rank: dict[int, int] = {}
    for task in plan.tasks:
        if not isinstance(task, VAETaskSpec):
            raise TypeError("tasks must contain VAETaskSpec values")
        _require_non_negative_int("task_id", task.task_id)
        _require_non_negative_int("assigned_rank", task.assigned_rank)
        _require_positive_int("estimated_cost", task.estimated_cost)
        _validate_slices(
            "input_slices",
            task.input_slices,
            global_shape=plan.global_input_shape,
            padded_shape=task.padded_input_shape,
        )
        _validate_slices(
            "valid_output_slices",
            task.valid_output_slices,
            global_shape=plan.global_output_shape,
            padded_shape=task.padded_output_shape,
        )
        task_ids.append(task.task_id)
        tasks_per_rank[task.assigned_rank] = (
            tasks_per_rank.get(task.assigned_rank, 0) + 1
        )

    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_id values must be unique")
    if task_ids != list(range(len(task_ids))):
        raise ValueError("task_id values must be in stable row-major order")
    _require_positive_int("rounds", plan.rounds)
    expected_rounds = max(tasks_per_rank.values())
    if plan.rounds != expected_rounds:
        raise ValueError("rounds must equal the maximum assigned task count")
    return plan


def encode_vae_task_plan(plan: VAETaskPlan) -> torch.Tensor:
    validate_vae_task_plan(plan)
    operation_code = 0 if plan.operation is VAEOperation.ENCODE else 1
    values = [
        VAE_PLAN_SCHEMA_VERSION,
        operation_code,
        plan.owner_rank,
        len(plan.global_input_shape),
        len(plan.global_output_shape),
        len(plan.tasks),
        plan.rounds,
        *plan.global_input_shape,
        *plan.global_output_shape,
    ]
    for task in plan.tasks:
        values.extend((task.task_id, task.assigned_rank, task.estimated_cost))
        values.extend(value for interval in task.input_slices for value in interval)
        values.extend(
            value for interval in task.valid_output_slices for value in interval
        )
        values.extend(task.padded_input_shape)
        values.extend(task.padded_output_shape)
    return torch.tensor(values, dtype=torch.int64)


def vae_task_plan_metadata_size(
    input_ndim: int,
    output_ndim: int,
    task_count: int,
) -> int:
    _require_positive_int("input_ndim", input_ndim)
    _require_positive_int("output_ndim", output_ndim)
    _require_positive_int("task_count", task_count)
    return (
        7
        + input_ndim
        + output_ndim
        + task_count * (3 + (3 * input_ndim) + (3 * output_ndim))
    )


def decode_vae_task_plan(metadata: torch.Tensor) -> VAETaskPlan:
    _validate_metadata_tensor(metadata)
    values = metadata.detach().to(device="cpu").tolist()
    if len(values) < 7:
        raise ValueError("VAE plan metadata is shorter than the schema header")
    (
        schema_version,
        operation_code,
        owner_rank,
        input_ndim,
        output_ndim,
        task_count,
        rounds,
    ) = values[:7]
    if schema_version != VAE_PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported VAE plan schema version: {schema_version}")
    if operation_code not in (0, 1):
        raise ValueError("VAE plan operation code is invalid")
    _require_positive_int("input_ndim", input_ndim)
    _require_positive_int("output_ndim", output_ndim)
    _require_positive_int("task_count", task_count)

    expected_length = vae_task_plan_metadata_size(
        input_ndim,
        output_ndim,
        task_count,
    )
    if len(values) < expected_length:
        raise ValueError("VAE plan task_count exceeds available task metadata")
    if len(values) > expected_length:
        raise ValueError("VAE plan metadata contains trailing fields")

    cursor = 7
    global_input_shape = tuple(values[cursor : cursor + input_ndim])
    cursor += input_ndim
    global_output_shape = tuple(values[cursor : cursor + output_ndim])
    cursor += output_ndim
    tasks: list[VAETaskSpec] = []
    for _ in range(task_count):
        task_id, assigned_rank, estimated_cost = values[cursor : cursor + 3]
        cursor += 3
        input_slices, cursor = _decode_slices(values, cursor, input_ndim)
        output_slices, cursor = _decode_slices(values, cursor, output_ndim)
        padded_input_shape = tuple(values[cursor : cursor + input_ndim])
        cursor += input_ndim
        padded_output_shape = tuple(values[cursor : cursor + output_ndim])
        cursor += output_ndim
        tasks.append(
            VAETaskSpec(
                task_id=task_id,
                assigned_rank=assigned_rank,
                input_slices=input_slices,
                valid_output_slices=output_slices,
                padded_input_shape=padded_input_shape,
                padded_output_shape=padded_output_shape,
                estimated_cost=estimated_cost,
            )
        )

    plan = VAETaskPlan(
        operation=(
            VAEOperation.ENCODE if operation_code == 0 else VAEOperation.DECODE
        ),
        owner_rank=owner_rank,
        global_input_shape=global_input_shape,
        global_output_shape=global_output_shape,
        tasks=tuple(tasks),
        rounds=rounds,
    )
    return validate_vae_task_plan(plan)


def stable_vae_plan_hash(metadata: torch.Tensor) -> str:
    if metadata.dtype != torch.int64 or metadata.ndim != 1:
        raise ValueError("VAE plan metadata must be a flat int64 tensor")
    payload = metadata.detach().to(device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _validate_metadata_tensor(metadata: torch.Tensor) -> None:
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("VAE plan metadata must be a torch.Tensor")
    if metadata.dtype != torch.int64:
        raise ValueError("VAE plan metadata must use int64 dtype")
    if metadata.ndim != 1:
        raise ValueError("VAE plan metadata must be flat")


def _validate_shape(name: str, shape: tuple[int, ...]) -> None:
    if type(shape) is not tuple or not shape:
        raise ValueError(f"{name} must be a non-empty tuple")
    for value in shape:
        _require_positive_int(name, value)


def _validate_slices(
    name: str,
    slices: tuple[tuple[int, int], ...],
    *,
    global_shape: tuple[int, ...],
    padded_shape: tuple[int, ...],
) -> None:
    _validate_shape(
        "padded_input_shape" if name == "input_slices" else "padded_output_shape",
        padded_shape,
    )
    if type(slices) is not tuple or len(slices) != len(global_shape):
        raise ValueError(f"{name} must match global shape dimensions")
    if len(padded_shape) != len(global_shape):
        padded_name = (
            "padded_input_shape" if name == "input_slices" else "padded_output_shape"
        )
        raise ValueError(f"{padded_name} must match global shape dimensions")
    for dimension, (interval, global_size, padded_size) in enumerate(
        zip(slices, global_shape, padded_shape, strict=True)
    ):
        if type(interval) is not tuple or len(interval) != 2:
            raise ValueError(f"{name}[{dimension}] must be a start/end tuple")
        start, end = interval
        _require_non_negative_int(name, start)
        _require_non_negative_int(name, end)
        if start >= end or end > global_size:
            raise ValueError(f"{name}[{dimension}] is outside global shape")
        if end - start > padded_size:
            raise ValueError(f"{name}[{dimension}] exceeds padded shape")


def _decode_slices(
    values: list[int],
    cursor: int,
    ndim: int,
) -> tuple[tuple[tuple[int, int], ...], int]:
    slices = tuple(
        (values[cursor + (2 * index)], values[cursor + (2 * index) + 1])
        for index in range(ndim)
    )
    return slices, cursor + (2 * ndim)


def _require_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a plain int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a plain int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_coordinator_group(
    name: str,
    coordinator: Any,
    *,
    expected_group: RuntimeGroup,
) -> None:
    coordinator_group = getattr(coordinator, "group", None)
    if coordinator_group is not expected_group:
        raise ValueError(f"{name} group must match binding group")


def _require_coordinator_methods(
    name: str,
    coordinator: Any,
    methods: tuple[str, ...],
) -> None:
    for method in methods:
        if not callable(getattr(coordinator, method, None)):
            raise TypeError(f"{name} must provide {method}")


def _validate_tensor_factory(
    name: str,
    dtype: torch.dtype,
    *,
    device: torch.device,
) -> None:
    try:
        torch.empty(0, dtype=dtype, device=device)
    except (TypeError, RuntimeError) as error:
        raise TypeError(f"{name} must be a supported torch dtype") from error


def _round_envelope(
    slots: tuple[VAETaskSpec | None, ...],
    *,
    field: str,
) -> tuple[int, ...]:
    shapes = tuple(getattr(task, field) for task in slots if task is not None)
    if not shapes:
        raise ValueError("every VAE round must contain at least one active task")
    ndim = len(shapes[0])
    if any(len(shape) != ndim for shape in shapes):
        raise ValueError(f"{field} dimensions must match within a round")
    return tuple(max(shape[dimension] for shape in shapes) for dimension in range(ndim))


def _pad_tensor(
    tensor: torch.Tensor,
    *,
    target_shape: tuple[int, ...],
    maximum_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
    name: str,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} dtype does not match the transfer envelope")
    if tensor.device != expected_device:
        raise ValueError(f"{name} device does not match the transfer envelope")
    if tensor.ndim != len(target_shape) or tensor.ndim != len(maximum_shape):
        raise ValueError(f"{name} dimensions do not match the transfer envelope")
    if any(
        current > maximum or current > target
        for current, maximum, target in zip(
            tensor.shape,
            maximum_shape,
            target_shape,
            strict=True,
        )
    ):
        raise ValueError(f"{name} exceeds its padded transfer shape")
    padded = torch.zeros(
        target_shape,
        dtype=expected_dtype,
        device=expected_device,
    )
    padded[tuple(slice(0, length) for length in tensor.shape)] = tensor
    return padded.contiguous()


def _origin_crop(
    tensor: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return tensor[tuple(slice(0, length) for length in shape)].contiguous()


def _memory_snapshot(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return (0, 0)
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
    )


def _max_memory_snapshot(
    left: tuple[int, int],
    right: tuple[int, int],
) -> tuple[int, int]:
    return (max(left[0], right[0]), max(left[1], right[1]))


def _tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    return element_count * torch.empty((), dtype=dtype).element_size()


def _slice_bytes(
    slices: tuple[tuple[int, int], ...],
    dtype: torch.dtype,
) -> int:
    shape = tuple(end - start for start, end in slices)
    return _tensor_bytes(shape, dtype)

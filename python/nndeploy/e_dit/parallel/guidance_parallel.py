"""Model-agnostic CFG branch pairing and prediction combination."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Any

import torch

from nndeploy.e_dit.parallel.layouts import (
    ShardMetadata,
    TensorLayout,
    TensorLayoutKind,
)

if TYPE_CHECKING:
    from nndeploy.e_dit.distributed.parallel_state import ParallelContext, RuntimeGroup


class GuidanceBranch(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class GuidanceParallelBinding:
    global_rank: int
    branch: GuidanceBranch
    branch_index: int
    sequence_shard_index: int
    world_data_group: RuntimeGroup
    world_control_group: RuntimeGroup
    sp_group: RuntimeGroup
    cfg_group: RuntimeGroup
    writer_rank: int

    def __post_init__(self) -> None:
        from nndeploy.e_dit.distributed.parallel_state import RuntimeGroup

        if type(self.global_rank) is not int or self.global_rank < 0:
            raise ValueError("global_rank must be a non-negative int")
        if not isinstance(self.branch, GuidanceBranch):
            raise TypeError("branch must be a GuidanceBranch")
        expected_branch_index = (
            0 if self.branch is GuidanceBranch.POSITIVE else 1
        )
        if self.branch_index != expected_branch_index:
            raise ValueError("branch_index must match branch")
        if type(self.sequence_shard_index) is not int or self.sequence_shard_index < 0:
            raise ValueError("sequence_shard_index must be a non-negative int")
        if type(self.writer_rank) is not int or self.writer_rank < 0:
            raise ValueError("writer_rank must be a non-negative int")
        for name in (
            "world_data_group",
            "world_control_group",
            "sp_group",
            "cfg_group",
        ):
            group = getattr(self, name)
            if not isinstance(group, RuntimeGroup):
                raise TypeError(f"{name} must be a RuntimeGroup")
            if not group.is_member or group.global_rank != self.global_rank:
                raise ValueError(f"{name} must contain global_rank")
        if self.cfg_group.world_size != 2:
            raise ValueError("CFG pair group must contain exactly two ranks")
        if self.cfg_group.group_rank != self.branch_index:
            raise ValueError("CFG pair group order must be positive then negative")
        if self.sequence_shard_index >= self.sp_group.world_size:
            raise ValueError("sequence_shard_index must fit SP group")

    @property
    def positive_pair_source(self) -> int:
        return self.cfg_group.spec.ranks[0]

    def branch_layout(self, branch_tensor: torch.Tensor) -> TensorLayout:
        if not isinstance(branch_tensor, torch.Tensor):
            raise TypeError("branch_tensor must be a torch.Tensor")
        tensor_shape = tuple(branch_tensor.shape)
        return TensorLayout(
            kind=TensorLayoutKind.BRANCH_SHARDED,
            global_shape=(2, *tensor_shape),
            local_shape=(1, *tensor_shape),
            shard=ShardMetadata(
                shard_dim=0,
                shard_index=self.branch_index,
                shard_count=2,
                offset=self.branch_index,
                valid_length=1,
                padded_length=1,
            ),
        )


class GuidanceParallelPeerError(RuntimeError):
    """Report failures observed on another rank at a fixed CFG phase."""

    def __init__(self, peer_ranks: tuple[int, ...], *, phase: str) -> None:
        self.peer_ranks = peer_ranks
        self.phase = phase
        ranks = ", ".join(str(rank) for rank in peer_ranks)
        super().__init__(f"guidance parallel {phase} failed on peer rank(s): {ranks}")


def resolve_guidance_parallel_binding(
    context: ParallelContext,
    *,
    writer_rank: int,
) -> GuidanceParallelBinding:
    from nndeploy.e_dit.distributed.parallel_state import ParallelContext

    if not isinstance(context, ParallelContext):
        raise TypeError("context must be a ParallelContext")
    if not context.enabled or context.topology is None:
        raise RuntimeError("guidance parallel requires an active parallel context")
    if context.plan.cfg_degree != 2:
        raise ValueError("guidance parallel requires cfg_degree=2")
    if type(writer_rank) is not int:
        raise TypeError("writer_rank must be an int")
    if writer_rank != context.plan.writer_rank or writer_rank != 0:
        raise ValueError(
            "guidance parallel writer_rank must match plan.writer_rank and equal 0"
        )

    topology = context.topology
    if len(topology.sp_groups) != 2:
        raise ValueError("guidance parallel requires two ordered branch SP groups")
    if len(topology.cfg_groups) != context.plan.sp_degree:
        raise ValueError("guidance parallel CFG group count must match sp_degree")

    branch_index = next(
        (
            index
            for index, spec in enumerate(topology.sp_groups)
            if context.global_rank in spec.ranks
        ),
        None,
    )
    sequence_shard_index = next(
        (
            index
            for index, spec in enumerate(topology.cfg_groups)
            if context.global_rank in spec.ranks
        ),
        None,
    )
    if branch_index is None or sequence_shard_index is None:
        raise RuntimeError("rank is missing from the guidance parallel mesh")

    sp_group = context.current_group("sp_")
    cfg_group = context.current_group("cfg_")
    expected_sp = topology.sp_groups[branch_index]
    expected_cfg = topology.cfg_groups[sequence_shard_index]
    if sp_group.spec.ranks != expected_sp.ranks:
        raise ValueError("SP group does not match deterministic branch mesh")
    if cfg_group.spec.ranks != expected_cfg.ranks:
        raise ValueError("CFG pair does not match deterministic sequence mesh")
    if cfg_group.spec.ranks != (
        topology.sp_groups[0].ranks[sequence_shard_index],
        topology.sp_groups[1].ranks[sequence_shard_index],
    ):
        raise ValueError("CFG pair order must be positive then negative")
    if writer_rank != topology.sp_groups[0].ranks[0]:
        raise ValueError("writer_rank must own positive sequence shard zero")

    return GuidanceParallelBinding(
        global_rank=context.global_rank,
        branch=(
            GuidanceBranch.POSITIVE
            if branch_index == 0
            else GuidanceBranch.NEGATIVE
        ),
        branch_index=branch_index,
        sequence_shard_index=sequence_shard_index,
        world_data_group=context.world_data_group(),
        world_control_group=context.world_control_group(),
        sp_group=sp_group,
        cfg_group=cfg_group,
        writer_rank=writer_rank,
    )


class GuidanceParallelEngine:
    def __init__(
        self,
        binding: GuidanceParallelBinding,
        *,
        cfg_coordinator: Any | None = None,
        world_control_coordinator: Any | None = None,
        control_device: torch.device | str = "cpu",
    ) -> None:
        if not isinstance(binding, GuidanceParallelBinding):
            raise TypeError("binding must be a GuidanceParallelBinding")
        from nndeploy.e_dit.distributed.group_coordinator import GroupCoordinator

        self.binding = binding
        self.cfg_coordinator = cfg_coordinator or GroupCoordinator(binding.cfg_group)
        self.world_control_coordinator = world_control_coordinator or GroupCoordinator(
            binding.world_control_group
        )
        self.control_device = torch.device(control_device)
        self._validate_coordinator(
            self.cfg_coordinator,
            expected_group=binding.cfg_group,
            required_methods=("broadcast", "all_reduce"),
            name="cfg_coordinator",
        )
        self._validate_coordinator(
            self.world_control_coordinator,
            expected_group=binding.world_control_group,
            required_methods=("all_gather_into_tensor",),
            name="world_control_coordinator",
        )

    @staticmethod
    def _validate_coordinator(
        coordinator: Any,
        *,
        expected_group: RuntimeGroup,
        required_methods: tuple[str, ...],
        name: str,
    ) -> None:
        if getattr(coordinator, "rank", None) != expected_group.group_rank:
            raise ValueError(f"{name} rank must match binding group rank")
        if getattr(coordinator, "world_size", None) != expected_group.world_size:
            raise ValueError(f"{name} world_size must match binding group")
        coordinator_ranks = getattr(
            getattr(coordinator, "group", None),
            "spec",
            None,
        )
        if getattr(coordinator_ranks, "ranks", None) != expected_group.spec.ranks:
            raise ValueError(f"{name} group ranks must match binding group")
        for method in required_methods:
            if not callable(getattr(coordinator, method, None)):
                raise TypeError(f"{name} must provide {method}")

    @staticmethod
    def _validate_reference(reference_tensor: torch.Tensor) -> None:
        if not isinstance(reference_tensor, torch.Tensor):
            raise TypeError("reference_tensor must be a torch.Tensor")
        if not reference_tensor.is_floating_point():
            raise TypeError("reference_tensor must have floating dtype")

    def _validate_branch_prediction(
        self,
        branch_prediction: torch.Tensor | None,
        *,
        reference_tensor: torch.Tensor,
        required: bool,
    ) -> None:
        if branch_prediction is None:
            if required:
                raise ValueError("branch_prediction is required on the active branch")
            return
        if not isinstance(branch_prediction, torch.Tensor):
            raise TypeError("branch_prediction must be a torch.Tensor or None")
        if not branch_prediction.is_floating_point():
            raise TypeError("branch_prediction must have floating dtype")
        if branch_prediction.shape != reference_tensor.shape:
            raise ValueError("branch_prediction shape must match reference_tensor")
        if branch_prediction.device != reference_tensor.device:
            raise ValueError("branch_prediction device must match reference_tensor")

    def replicate_from_positive(
        self,
        positive_tensor: torch.Tensor | None,
        *,
        receive_template: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(receive_template, torch.Tensor):
            raise TypeError("receive_template must be a torch.Tensor")
        if self.binding.branch is GuidanceBranch.POSITIVE:
            if not isinstance(positive_tensor, torch.Tensor):
                raise ValueError("positive_tensor is required on the positive branch")
            if positive_tensor.shape != receive_template.shape:
                raise ValueError("positive_tensor shape must match receive_template")
            if positive_tensor.dtype != receive_template.dtype:
                raise ValueError("positive_tensor dtype must match receive_template")
            if positive_tensor.device != receive_template.device:
                raise ValueError("positive_tensor device must match receive_template")
            result = positive_tensor.contiguous()
        else:
            if positive_tensor is not None:
                raise ValueError("positive_tensor must be None on the negative branch")
            result = torch.empty_like(receive_template)
        return self.cfg_coordinator.broadcast(
            result,
            src=self.binding.positive_pair_source,
        )

    def merge_predictions(
        self,
        branch_prediction: torch.Tensor | None,
        *,
        reference_tensor: torch.Tensor,
        guidance_scale: float | torch.Tensor,
        do_cfg: bool,
    ) -> torch.Tensor:
        self._validate_reference(reference_tensor)
        if type(do_cfg) is not bool:
            raise TypeError("do_cfg must be a bool")
        prediction_required = do_cfg or self.binding.branch is GuidanceBranch.POSITIVE
        self._validate_branch_prediction(
            branch_prediction,
            reference_tensor=reference_tensor,
            required=prediction_required,
        )

        if do_cfg:
            guidance = self._guidance_tensor(
                guidance_scale,
                reference_tensor=reference_tensor,
            )
            prediction_fp32 = branch_prediction.to(torch.float32)
            try:
                broadcast_shape = (guidance * prediction_fp32).shape
            except RuntimeError as error:
                raise ValueError(
                    "guidance_scale tensor must broadcast to branch prediction"
                ) from error
            if broadcast_shape != reference_tensor.shape:
                raise ValueError(
                    "guidance_scale tensor must broadcast without changing prediction shape"
                )

            # Keep one pair collective per CFG-on step, but exchange the raw
            # FP32 branch predictions in separate slots.  Combining the slots
            # afterward preserves the historical single-rank operation order;
            # reducing pre-weighted terms is algebraically equivalent but
            # accumulates observable rounding drift across denoising steps.
            branch_slots = torch.zeros(
                (2, *reference_tensor.shape),
                dtype=torch.float32,
                device=reference_tensor.device,
            )
            branch_slots[self.binding.branch_index].copy_(prediction_fp32)
            paired_predictions = self.cfg_coordinator.all_reduce(
                branch_slots.contiguous()
            )
            positive_prediction = paired_predictions[0]
            negative_prediction = paired_predictions[1]
            return negative_prediction + guidance * (
                positive_prediction - negative_prediction
            )

        if self.binding.branch is GuidanceBranch.POSITIVE:
            result = branch_prediction.to(torch.float32).contiguous()
        else:
            result = torch.empty_like(reference_tensor, dtype=torch.float32)
        return self.cfg_coordinator.broadcast(
            result,
            src=self.binding.positive_pair_source,
        )

    @staticmethod
    def _guidance_tensor(
        guidance_scale: float | torch.Tensor,
        *,
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(guidance_scale, torch.Tensor):
            if not guidance_scale.is_floating_point():
                raise TypeError("guidance_scale tensor must have floating dtype")
            if guidance_scale.device != reference_tensor.device:
                raise ValueError("guidance_scale device must match reference_tensor")
            return guidance_scale.to(torch.float32)
        if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, Real):
            raise TypeError("guidance_scale must be a real scalar or tensor")
        return torch.tensor(
            float(guidance_scale),
            dtype=torch.float32,
            device=reference_tensor.device,
        )

    def synchronize_phase_error(
        self,
        local_error: BaseException | None,
        *,
        phase: str,
    ) -> None:
        if not isinstance(phase, str):
            raise TypeError("phase must be a string")
        if not phase:
            raise ValueError("phase must be non-empty")
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
        self.world_control_coordinator.all_gather_into_tensor(
            gathered_flags,
            local_flag,
        )
        failed_slots = gathered_flags.ne(0).nonzero().flatten().tolist()
        if failed_slots:
            failed_ranks = tuple(
                self.binding.world_control_group.spec.ranks[slot]
                for slot in failed_slots
            )
            error = GuidanceParallelPeerError(failed_ranks, phase=phase)
            if local_error is not None:
                raise error from local_error
            raise error
        if local_error is not None:
            raise RuntimeError(
                "world error flags omitted the local guidance-parallel failure"
            ) from local_error


__all__ = (
    "GuidanceBranch",
    "GuidanceParallelBinding",
    "GuidanceParallelEngine",
    "GuidanceParallelPeerError",
    "resolve_guidance_parallel_binding",
)

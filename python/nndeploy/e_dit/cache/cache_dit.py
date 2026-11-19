"""Request/window-scoped Cache-DiT DBCache controller for diffusion inference."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from nndeploy.e_dit.config.cache_dit import (
    CacheDitParams,
)
from nndeploy.e_dit.config.transformer_cache import (
    TransformerCacheMode,
)
from nndeploy.e_dit.cache.base import CacheBranch, CacheExecutionContext
from nndeploy.e_dit.cache.consensus import CacheConsensusStats, CacheDecisionConsensus

if TYPE_CHECKING:
    from nndeploy.e_dit.layers.attention.sequence_parallel import SequenceParallelMetadata
    from nndeploy.e_dit.distributed.group_coordinator import GroupCoordinator


@dataclass(frozen=True)
class CacheDitDecision:
    should_reuse_middle: bool
    reason: str
    residual_diff: float | None = None


@dataclass
class _CacheDitBranchState:
    previous_computed_step: int | None = None
    previous_observed_step: int | None = None
    previous_front_residual: torch.Tensor | None = None
    cached_middle_residual: torch.Tensor | None = None
    previous_layout_signature: str | None = None
    continuous_cached_steps: int = 0
    computed_middle_steps: int = 0
    cached_middle_steps: int = 0
    completed_steps: int = 0
    front_steps: int = 0
    back_steps: int = 0
    residual_diffs: list[float] = field(default_factory=list)
    forced_compute_reasons: dict[str, int] = field(default_factory=dict)
    pending: CacheDitDecision | None = None
    pending_front_residual: torch.Tensor | None = None
    pending_layout_signature: str | None = None
    pending_step: int | None = None

    def clear_tensors(self) -> None:
        self.previous_front_residual = None
        self.cached_middle_residual = None
        self.pending = None
        self.pending_front_residual = None
        self.pending_layout_signature = None
        self.pending_step = None


_DECISION_CODES = {
    "cache_hit": 1,
    "warmup": 2,
    "end_guard": 3,
    "first_branch_step": 4,
    "non_contiguous_branch_step": 5,
    "missing_cached_state": 6,
    "max_consecutive_cached_steps": 7,
    "threshold_exceeded": 8,
    "non_finite_diff": 9,
    "layout_changed": 10,
}


class CacheDitController:
    mode = TransformerCacheMode.CACHE_DIT

    def __init__(
        self,
        params: CacheDitParams,
        *,
        request_id: str,
        object_index: int,
        window_index: int,
        total_steps: int,
        num_transformer_blocks: int,
        sp_degree: int,
        sp_rank: int,
        cfg_degree: int,
        cfg_rank: int,
        coordinator: GroupCoordinator | None,
        sp_group_identity: str,
        cfg_group_identity: str,
        model_identity: str,
    ) -> None:
        if not params.enabled:
            raise ValueError("disabled Cache-DiT must not construct a controller")
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError("Cache-DiT total_steps must be a positive non-bool int")
        params.validate_block_count(num_transformer_blocks)
        self.params = params
        self.model_identity = model_identity
        self.request_id = request_id
        self.object_index = object_index
        self.window_index = window_index
        self.total_steps = total_steps
        self.num_transformer_blocks = num_transformer_blocks
        self.sp_degree = sp_degree
        self.sp_rank = sp_rank
        self.cfg_degree = cfg_degree
        self.cfg_rank = cfg_rank
        self.sp_group_identity = sp_group_identity
        self.cfg_group_identity = cfg_group_identity
        self._states = {branch: _CacheDitBranchState() for branch in CacheBranch}
        self._consensus_stats = CacheConsensusStats()
        self._consensus = CacheDecisionConsensus(
            coordinator,
            stats=self._consensus_stats,
        )
        self._closed = False
        self._final_stats: dict[str, Any] | None = None
        self._abort_reason: str | None = None

    @property
    def front_end(self) -> int:
        return self.params.front_blocks

    @property
    def middle_end(self) -> int:
        return self.num_transformer_blocks - self.params.back_blocks

    @property
    def back_start(self) -> int:
        return self.middle_end

    def adapter(self, branch: CacheBranch) -> "CacheDitBranchAdapter":
        self._require_open()
        return CacheDitBranchAdapter(self, branch)

    def check(
        self,
        *,
        branch: CacheBranch,
        step: int,
        input_hidden_states: torch.Tensor,
        front_output_hidden_states: torch.Tensor,
        global_sequence_length: int,
        local_sequence_length: int,
        valid_local_sequence_length: int,
        layout_signature: str,
        dtype: str,
        device: str,
        hidden_width: int,
    ) -> CacheDitDecision:
        decision_started = time.perf_counter()
        self._require_open()
        state = self._states[branch]
        context = None
        front_residual = None
        validation_error: BaseException | None = None
        try:
            if not isinstance(input_hidden_states, torch.Tensor) or not isinstance(
                front_output_hidden_states, torch.Tensor
            ):
                raise TypeError("Cache-DiT hidden-state inputs must be tensors")
            if input_hidden_states.shape != front_output_hidden_states.shape:
                raise ValueError("Cache-DiT front block output shape changed")
            context = self._context(
                branch=branch,
                step=step,
                global_sequence_length=global_sequence_length,
                local_sequence_length=local_sequence_length,
                valid_local_sequence_length=valid_local_sequence_length,
                layout_signature=layout_signature,
                dtype=dtype,
                device=device,
                hidden_width=hidden_width,
            )
            # Use the same arithmetic policy for the decision probe and the
            # cached middle. Model adapters may override this to subtract in FP32.
            front_residual = self._compute_residual(input_hidden_states, front_output_hidden_states)
        except BaseException as error:
            validation_error = error
        validation_device = (
            input_hidden_states.device
            if isinstance(input_hidden_states, torch.Tensor)
            else torch.device(device)
        )
        self._consensus.synchronize_validation(
            validation_error,
            device=validation_device,
            phase="Cache-DiT execution context",
        )
        assert context is not None
        assert front_residual is not None

        state_error: BaseException | None = None
        reason = None
        try:
            if state.pending is not None:
                raise RuntimeError("Cache-DiT previous step was not completed")
            reason = self._forced_compute_reason(state, step, layout_signature)
        except BaseException as error:
            state_error = error
        self._consensus.synchronize_validation(
            state_error,
            device=validation_device,
            phase="Cache-DiT state preparation",
        )

        if reason is not None:
            decision = CacheDitDecision(False, reason)
        else:
            assert state.previous_front_residual is not None
            diff = self._consensus.relative_l1(
                front_residual,
                state.previous_front_residual,
                context=context,
            )
            if not math.isfinite(diff):
                decision = CacheDitDecision(False, "non_finite_diff", diff)
            elif diff < self.params.residual_diff_threshold:
                decision = CacheDitDecision(True, "cache_hit", diff)
            else:
                decision = CacheDitDecision(False, "threshold_exceeded", diff)
            state.residual_diffs.append(float(diff))

        self._consensus.assert_decision(
            _DECISION_CODES[decision.reason],
            device=front_residual.device,
        )
        if not decision.should_reuse_middle:
            state.forced_compute_reasons[decision.reason] = (
                state.forced_compute_reasons.get(decision.reason, 0) + 1
            )
        state.pending = decision
        state.pending_front_residual = front_residual.detach().clone()
        state.pending_layout_signature = layout_signature
        state.pending_step = step
        self._consensus_stats.decision_count += 1
        self._consensus_stats.decision_seconds += (
            time.perf_counter() - decision_started
        )
        return decision

    def apply_cached_middle(
        self,
        *,
        branch: CacheBranch,
        front_output_hidden_states: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._require_pending(branch, step, should_reuse=True)
        state = self._states[branch]
        if state.cached_middle_residual is None:
            raise RuntimeError("Cache-DiT cached middle residual is missing")
        return front_output_hidden_states + state.cached_middle_residual

    def record_middle_compute(
        self,
        *,
        branch: CacheBranch,
        front_output_hidden_states: torch.Tensor,
        middle_output_hidden_states: torch.Tensor,
        step: int,
    ) -> None:
        self._require_pending(branch, step, should_reuse=False)
        state = self._states[branch]
        if front_output_hidden_states.shape != middle_output_hidden_states.shape:
            raise ValueError("Cache-DiT middle block output shape changed")
        assert state.pending_front_residual is not None
        state.cached_middle_residual = self._compute_residual(front_output_hidden_states, middle_output_hidden_states)
        state.previous_front_residual = state.pending_front_residual
        state.previous_layout_signature = state.pending_layout_signature
        state.previous_computed_step = step

    @staticmethod
    def _compute_residual(input_hidden_states, output_hidden_states):
        return (output_hidden_states - input_hidden_states).detach().clone()

    def complete_step(self, *, branch: CacheBranch, step: int) -> None:
        self._require_open()
        state = self._states[branch]
        if state.pending is None or state.pending_step != step:
            raise RuntimeError("Cache-DiT completion has no matching pending decision")
        # A recorded compute promotes this decision's probe to the reference.
        if (
            not state.pending.should_reuse_middle
            and state.previous_front_residual is not state.pending_front_residual
        ):
            raise RuntimeError("Cache-DiT compute result was not recorded")
        state.front_steps += 1
        if state.pending.should_reuse_middle:
            state.cached_middle_steps += 1
            state.continuous_cached_steps += 1
        else:
            state.computed_middle_steps += 1
            state.continuous_cached_steps = 0
        if self.params.back_blocks:
            state.back_steps += 1
        state.completed_steps += 1
        state.previous_observed_step = step
        state.pending = None
        state.pending_front_residual = None
        state.pending_layout_signature = None
        state.pending_step = None

    def finish_window(self) -> dict[str, Any]:
        return self._close(abort_reason=None)

    def abort_window(self, reason: str) -> dict[str, Any]:
        return self._close(abort_reason=str(reason))

    def _close(self, *, abort_reason: str | None) -> dict[str, Any]:
        if self._closed:
            assert self._final_stats is not None
            return self._final_stats
        self._abort_reason = abort_reason
        self._closed = True
        self._final_stats = self.stats()
        for state in self._states.values():
            state.clear_tensors()
        return self._final_stats

    def stats(self) -> dict[str, Any]:
        middle_blocks = self.middle_end - self.front_end
        branches: dict[str, dict[str, Any]] = {}
        for branch, state in self._states.items():
            executed_blocks = (
                state.front_steps * self.params.front_blocks
                + state.computed_middle_steps * middle_blocks
                + state.back_steps * self.params.back_blocks
            )
            possible_blocks = state.completed_steps * self.num_transformer_blocks
            branches[branch.value] = {
                "branch": branch.value,
                "front_steps": state.front_steps,
                "computed_middle_steps": state.computed_middle_steps,
                "cached_middle_steps": state.cached_middle_steps,
                "back_steps": state.back_steps,
                "completed_steps": state.completed_steps,
                "middle_skip_ratio": _ratio(
                    state.cached_middle_steps, state.completed_steps
                ),
                "effective_blocks_executed": executed_blocks,
                "estimated_block_ratio": _ratio(
                    executed_blocks, possible_blocks
                ),
                "continuous_cached_steps": state.continuous_cached_steps,
                "previous_computed_step": state.previous_computed_step,
                "previous_observed_step": state.previous_observed_step,
                "residual_diffs": list(state.residual_diffs),
                "forced_compute_reasons": dict(state.forced_compute_reasons),
            }
        computed = sum(
            item["computed_middle_steps"] for item in branches.values()
        )
        cached = sum(item["cached_middle_steps"] for item in branches.values())
        completed = sum(item["completed_steps"] for item in branches.values())
        executed = sum(
            item["effective_blocks_executed"] for item in branches.values()
        )
        possible = completed * self.num_transformer_blocks
        return {
            "mode": "cache_dit",
            "enabled": True,
            "model_policy": self.model_identity,
            "closed": self._closed,
            "aborted": self._abort_reason is not None,
            "abort_reason": self._abort_reason,
            "request_id": self.request_id,
            "object_index": self.object_index,
            "window_index": self.window_index,
            "front_blocks": self.params.front_blocks,
            "middle_blocks": middle_blocks,
            "back_blocks": self.params.back_blocks,
            "warmup_steps": self.params.warmup_steps,
            "residual_diff_threshold": self.params.residual_diff_threshold,
            "max_consecutive_cached_steps": (
                self.params.max_consecutive_cached_steps
            ),
            "end_guard_steps": self.params.end_guard_steps,
            "resolved_params": asdict(self.params),
            "topology": {
                "sp_degree": self.sp_degree,
                "sp_rank": self.sp_rank,
                "cfg_degree": self.cfg_degree,
                "cfg_rank": self.cfg_rank,
            },
            "communication": asdict(self._consensus_stats),
            "branches": branches,
            "total": {
                "computed_middle_steps": computed,
                "cached_middle_steps": cached,
                "middle_skip_ratio": _ratio(cached, completed),
                "effective_blocks_executed": executed,
                "estimated_block_ratio": _ratio(executed, possible),
                "completed_steps": completed,
            },
        }

    def _forced_compute_reason(
        self,
        state: _CacheDitBranchState,
        step: int,
        layout_signature: str,
    ) -> str | None:
        if step < self.params.warmup_steps:
            return "warmup"
        if step >= self.total_steps - self.params.end_guard_steps:
            return "end_guard"
        if state.previous_observed_step is None:
            return "first_branch_step"
        if state.previous_layout_signature != layout_signature:
            state.previous_front_residual = None
            state.cached_middle_residual = None
            state.continuous_cached_steps = 0
            return "layout_changed"
        if state.previous_observed_step + 1 != step:
            return "non_contiguous_branch_step"
        if (
            state.previous_front_residual is None
            or state.cached_middle_residual is None
        ):
            return "missing_cached_state"
        if (
            state.continuous_cached_steps
            >= self.params.max_consecutive_cached_steps
        ):
            return "max_consecutive_cached_steps"
        return None

    def _require_pending(
        self,
        branch: CacheBranch,
        step: int,
        *,
        should_reuse: bool,
    ) -> None:
        self._require_open()
        state = self._states[branch]
        if state.pending is None or state.pending_step != step:
            raise RuntimeError("Cache-DiT operation has no matching pending decision")
        if state.pending.should_reuse_middle is not should_reuse:
            raise RuntimeError("Cache-DiT operation contradicts pending decision")

    def _context(
        self,
        *,
        branch: CacheBranch,
        step: int,
        global_sequence_length: int,
        local_sequence_length: int,
        valid_local_sequence_length: int,
        layout_signature: str,
        dtype: str,
        device: str,
        hidden_width: int,
    ) -> CacheExecutionContext:
        return CacheExecutionContext(
            request_id=self.request_id,
            object_index=self.object_index,
            window_index=self.window_index,
            branch=branch,
            step_index=step,
            total_steps=self.total_steps,
            global_sequence_length=global_sequence_length,
            local_sequence_length=local_sequence_length,
            valid_local_sequence_length=valid_local_sequence_length,
            sp_degree=self.sp_degree,
            sp_rank=self.sp_rank,
            cfg_degree=self.cfg_degree,
            cfg_rank=self.cfg_rank,
            model_identity=self.model_identity,
            layout_signature=layout_signature,
            sp_group_identity=self.sp_group_identity,
            cfg_group_identity=self.cfg_group_identity,
            dtype=dtype,
            device=device,
            hidden_width=hidden_width,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Cache-DiT window is already closed")


class CacheDitBranchAdapter:
    def __init__(self, controller: CacheDitController, branch: CacheBranch) -> None:
        self._controller = controller
        self.branch = branch

    @property
    def front_end(self) -> int:
        return self._controller.front_end

    @property
    def middle_end(self) -> int:
        return self._controller.middle_end

    @property
    def back_start(self) -> int:
        return self._controller.back_start

    @property
    def num_transformer_blocks(self) -> int:
        return self._controller.num_transformer_blocks

    def check(
        self,
        *,
        step: int,
        input_hidden_states: torch.Tensor,
        front_output_hidden_states: torch.Tensor,
        sequence_parallel_metadata: SequenceParallelMetadata | None = None,
    ) -> bool:
        sequence_length = front_output_hidden_states.size(1)
        metadata = sequence_parallel_metadata
        global_length = (
            metadata.global_length if metadata is not None else sequence_length
        )
        valid_local_length = (
            metadata.valid_local_length if metadata is not None else sequence_length
        )
        shard_signature = (
            f"sp:{metadata.sp_degree}:{metadata.rank}:"
            f"{metadata.global_length}:{sequence_length}:"
            f"{metadata.valid_local_length}:{metadata.local_start}:"
            f"{metadata.local_end}"
            if metadata is not None
            else f"sp:1:0:{sequence_length}:{sequence_length}:"
            f"{sequence_length}:0:{sequence_length}"
        )
        layout_signature = (
            f"{shard_signature}:cfg:{self._controller.cfg_degree}:"
            f"{self._controller.cfg_rank}:{self.branch.value}:"
            f"hidden:{front_output_hidden_states.size(-1)}:"
            f"dtype:{front_output_hidden_states.dtype}:"
            f"device:{front_output_hidden_states.device}"
        )
        return self._controller.check(
            branch=self.branch,
            step=step,
            input_hidden_states=input_hidden_states,
            front_output_hidden_states=front_output_hidden_states,
            global_sequence_length=global_length,
            local_sequence_length=sequence_length,
            valid_local_sequence_length=valid_local_length,
            layout_signature=layout_signature,
            dtype=str(front_output_hidden_states.dtype),
            device=str(front_output_hidden_states.device),
            hidden_width=front_output_hidden_states.size(-1),
        ).should_reuse_middle

    def update(
        self,
        front_output_hidden_states: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        return self._controller.apply_cached_middle(
            branch=self.branch,
            front_output_hidden_states=front_output_hidden_states,
            step=step,
        )

    def store_truth(
        self,
        *,
        step: int,
        front_output_hidden_states: torch.Tensor,
        middle_output_hidden_states: torch.Tensor,
    ) -> None:
        self._controller.record_middle_compute(
            branch=self.branch,
            front_output_hidden_states=front_output_hidden_states,
            middle_output_hidden_states=middle_output_hidden_states,
            step=step,
        )

    def complete(self, *, step: int) -> None:
        self._controller.complete_step(branch=self.branch, step=step)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0



__all__ = (
    "CacheDitController",
    "CacheDitDecision",
    "CacheDitBranchAdapter",
)

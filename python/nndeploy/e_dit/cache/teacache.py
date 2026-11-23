"""Request/window-scoped TeaCache controller for diffusion inference."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING, Callable

import torch

from nndeploy.e_dit.config.teacache import (
    TeaCacheCoefficientSelection,
    TeaCacheParams,
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
class TeaCacheDecision:
    should_skip: bool
    reason: str
    estimated_distance: float = 0.0
    accumulated_distance: float = 0.0


@dataclass
class _TeaCacheBranchState:
    previous_step: int | None = None
    previous_modulated_input: torch.Tensor | None = None
    previous_residual: torch.Tensor | None = None
    accumulated_distance: float = 0.0
    consecutive_skips: int = 0
    calc_steps: int = 0
    skip_steps: int = 0
    forced_compute_reasons: dict[str, int] = field(default_factory=dict)
    pending: TeaCacheDecision | None = None
    previous_layout_signature: str | None = None
    pending_layout_signature: str | None = None
    pending_step: int | None = None

    def clear_tensors(self) -> None:
        self.previous_modulated_input = None
        self.previous_residual = None
        self.pending = None
        self.pending_layout_signature = None
        self.pending_step = None


_DECISION_CODES = {
    "cache_hit": 1,
    "before_min_skip_step": 2,
    "after_max_skip_step": 3,
    "first_branch_step": 4,
    "non_contiguous_branch_step": 5,
    "missing_cached_state": 6,
    "max_consecutive_skip": 7,
    "threshold_exceeded": 8,
    "non_finite_distance": 9,
    "calibrate": 10,
    "layout_changed": 11,
}


class TeaCacheController:
    mode = TransformerCacheMode.TEACACHE

    def __init__(
        self,
        params: TeaCacheParams,
        *,
        request_id: str,
        object_index: int,
        window_index: int,
        total_steps: int,
        sp_degree: int,
        sp_rank: int,
        cfg_degree: int,
        cfg_rank: int,
        coordinator: GroupCoordinator | None,
        sp_group_identity: str,
        cfg_group_identity: str,
        model_identity: str,
        coefficient_selector: Callable[[int], TeaCacheCoefficientSelection],
    ) -> None:
        if not params.enabled:
            raise ValueError("disabled TeaCache must not construct a controller")
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError("TeaCache total_steps must be a positive non-bool int")
        self.params = params
        self.model_identity = model_identity
        self._coefficient_selector = coefficient_selector
        self.request_id = request_id
        self.object_index = object_index
        self.window_index = window_index
        self.total_steps = total_steps
        self.sp_degree = sp_degree
        self.sp_rank = sp_rank
        self.cfg_degree = cfg_degree
        self.cfg_rank = cfg_rank
        self.sp_group_identity = sp_group_identity
        self.cfg_group_identity = cfg_group_identity
        self._states = {branch: _TeaCacheBranchState() for branch in CacheBranch}
        self._selection: TeaCacheCoefficientSelection | None = None
        self._consensus_stats = CacheConsensusStats()
        self._consensus = CacheDecisionConsensus(
            coordinator,
            stats=self._consensus_stats,
        )
        self._closed = False
        self._final_stats: dict[str, Any] | None = None
        self._abort_reason: str | None = None

    def adapter(self, branch: CacheBranch) -> "TeaCacheBranchAdapter":
        self._require_open()
        return TeaCacheBranchAdapter(self, branch)

    def check(
        self,
        *,
        branch: CacheBranch,
        step: int,
        modulated_input: torch.Tensor,
        global_sequence_length: int,
        local_sequence_length: int,
        valid_local_sequence_length: int,
        layout_signature: str,
        dtype: str,
        device: str,
        hidden_width: int,
    ) -> TeaCacheDecision:
        decision_started = time.perf_counter()
        self._require_open()
        state = self._states[branch]
        context = None
        validation_error: BaseException | None = None
        try:
            if not isinstance(modulated_input, torch.Tensor):
                raise TypeError("TeaCache modulation input must be a tensor")
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
        except BaseException as error:
            validation_error = error
        validation_device = (
            modulated_input.device
            if isinstance(modulated_input, torch.Tensor)
            else torch.device(device)
        )
        self._consensus.synchronize_validation(
            validation_error,
            device=validation_device,
            phase="execution context",
        )
        assert context is not None
        state_error: BaseException | None = None
        selection = None
        reason = None
        try:
            if state.pending is not None:
                raise RuntimeError("TeaCache previous step was not completed")
            selection = self._selection_for(global_sequence_length)
            reason = self._forced_compute_reason(state, step, layout_signature)
        except BaseException as error:
            state_error = error
        self._consensus.synchronize_validation(
            state_error,
            device=validation_device,
            phase="state preparation",
        )
        assert selection is not None
        if reason is not None:
            decision = TeaCacheDecision(False, reason)
        elif self.params.calibrate:
            decision = TeaCacheDecision(False, "calibrate")
        else:
            assert state.previous_modulated_input is not None
            raw_distance = self._consensus.relative_l1(
                modulated_input,
                state.previous_modulated_input,
                context=context,
            )
            estimated = _evaluate_polynomial(selection.coefficients, raw_distance)
            if not math.isfinite(estimated):
                decision = TeaCacheDecision(False, "non_finite_distance")
            else:
                estimated = max(0.0, estimated)
                accumulated = state.accumulated_distance + estimated
                decision = TeaCacheDecision(
                    accumulated < self.params.threshold,
                    "cache_hit"
                    if accumulated < self.params.threshold
                    else "threshold_exceeded",
                    estimated,
                    accumulated,
                )
        self._consensus.assert_decision(
            _DECISION_CODES[decision.reason],
            device=modulated_input.device,
        )
        if not decision.should_skip:
            state.forced_compute_reasons[decision.reason] = (
                state.forced_compute_reasons.get(decision.reason, 0) + 1
            )
        state.pending = decision
        state.pending_step = step
        state.pending_layout_signature = layout_signature
        self._consensus_stats.decision_count += 1
        self._consensus_stats.decision_seconds += (
            time.perf_counter() - decision_started
        )
        return decision

    def apply_cached_update(
        self,
        *,
        branch: CacheBranch,
        modulated_input: torch.Tensor,
        input_hidden_states: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        self._require_open()
        state = self._states[branch]
        if state.pending is None or not state.pending.should_skip or state.pending_step != step:
            raise RuntimeError("TeaCache reuse requires a pending cache-hit decision")
        if state.previous_residual is None:
            raise RuntimeError("TeaCache cached residual is missing")
        output = input_hidden_states + state.previous_residual
        state.previous_step = step
        state.previous_modulated_input = modulated_input.detach().clone()
        state.previous_layout_signature = state.pending_layout_signature
        state.accumulated_distance = state.pending.accumulated_distance
        state.consecutive_skips += 1
        state.skip_steps += 1
        state.pending = None
        state.pending_step = None
        return output

    def record_compute(
        self,
        *,
        branch: CacheBranch,
        step: int,
        modulated_input: torch.Tensor,
        input_hidden_states: torch.Tensor,
        output_hidden_states: torch.Tensor,
    ) -> None:
        self._require_open()
        state = self._states[branch]
        if state.pending is None or state.pending.should_skip or state.pending_step != step:
            raise RuntimeError("TeaCache compute result has no pending compute decision")
        state.previous_step = step
        state.previous_modulated_input = modulated_input.detach().clone()
        state.previous_layout_signature = state.pending_layout_signature
        state.previous_residual = self._compute_residual(input_hidden_states, output_hidden_states)
        state.accumulated_distance = 0.0
        state.consecutive_skips = 0
        state.calc_steps += 1
        state.pending = None
        state.pending_step = None

    @staticmethod
    def _compute_residual(input_hidden_states, output_hidden_states):
        return (output_hidden_states - input_hidden_states).detach().clone()

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
        branches = {
            branch.value: {
                "calc_steps": state.calc_steps,
                "compute_steps": state.calc_steps,
                "skip_steps": state.skip_steps,
                "cache_hit_rate": _ratio(
                    state.skip_steps, state.calc_steps + state.skip_steps
                ),
                "hit_rate": _ratio(
                    state.skip_steps, state.calc_steps + state.skip_steps
                ),
                "forced_compute_reasons": dict(state.forced_compute_reasons),
                "previous_step": state.previous_step,
                "branch": branch.value,
            }
            for branch, state in self._states.items()
        }
        calc_steps = sum(item["calc_steps"] for item in branches.values())
        skip_steps = sum(item["skip_steps"] for item in branches.values())
        return {
            "mode": "teacache",
            "enabled": True,
            "model_policy": self.model_identity,
            "closed": self._closed,
            "aborted": self._abort_reason is not None,
            "abort_reason": self._abort_reason,
            "request_id": self.request_id,
            "object_index": self.object_index,
            "window_index": self.window_index,
            "threshold": self.params.threshold,
            "max_consecutive_skip": self.params.max_consecutive_skip,
            "min_skip_step": self.params.min_skip_step,
            "end_guard_steps": self.params.end_guard_steps,
            "calibrate": self.params.calibrate,
            "resolved_params": asdict(self.params),
            "global_sequence_length": (
                self._selection.requested_global_sequence_length
                if self._selection is not None
                else None
            ),
            "fit_length": (
                self._selection.selected_fit_length
                if self._selection is not None
                else None
            ),
            "coefficient_selection": (
                asdict(self._selection) if self._selection is not None else None
            ),
            "topology": {
                "sp_degree": self.sp_degree,
                "sp_rank": self.sp_rank,
                "cfg_degree": self.cfg_degree,
                "cfg_rank": self.cfg_rank,
            },
            "communication": asdict(self._consensus_stats),
            "branches": branches,
            "total": {
                "calc_steps": calc_steps,
                "compute_steps": calc_steps,
                "skip_steps": skip_steps,
                "cache_hit_rate": _ratio(skip_steps, calc_steps + skip_steps),
                "hit_rate": _ratio(skip_steps, calc_steps + skip_steps),
                "estimated_block_speedup": _ratio(
                    calc_steps + skip_steps,
                    calc_steps,
                ),
            },
        }

    def _forced_compute_reason(
        self,
        state: _TeaCacheBranchState,
        step: int,
        layout_signature: str,
    ) -> str | None:
        if step < self.params.min_skip_step:
            return "before_min_skip_step"
        if step >= self.total_steps - self.params.end_guard_steps:
            return "after_max_skip_step"
        if state.previous_step is None:
            return "first_branch_step"
        if state.previous_layout_signature != layout_signature:
            state.previous_modulated_input = None
            state.previous_residual = None
            return "layout_changed"
        if state.previous_step + 1 != step:
            return "non_contiguous_branch_step"
        if (
            state.previous_modulated_input is None
            or state.previous_residual is None
        ):
            return "missing_cached_state"
        if state.consecutive_skips >= self.params.max_consecutive_skip:
            return "max_consecutive_skip"
        return None

    def _selection_for(self, global_sequence_length: int):
        if self._selection is None:
            self._selection = self._coefficient_selector(global_sequence_length)
        elif (
            self._selection.requested_global_sequence_length
            != global_sequence_length
        ):
            raise RuntimeError("TeaCache global sequence length changed within a window")
        return self._selection

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
            raise RuntimeError("TeaCache window is already closed")


class TeaCacheBranchAdapter:
    def __init__(self, controller: TeaCacheController, branch: CacheBranch) -> None:
        self._controller = controller
        self.branch = branch

    def check(
        self,
        *,
        step: int,
        t_mod: torch.Tensor,
        sequence_length: int,
        sequence_parallel_metadata: SequenceParallelMetadata | None = None,
        hidden_width: int,
        hidden_dtype: torch.dtype,
        hidden_device: torch.device,
    ) -> bool:
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
            f"hidden:{hidden_width}:dtype:{hidden_dtype}:device:{hidden_device}"
        )
        return self._controller.check(
            branch=self.branch,
            step=step,
            modulated_input=t_mod,
            global_sequence_length=global_length,
            local_sequence_length=sequence_length,
            valid_local_sequence_length=valid_local_length,
            layout_signature=layout_signature,
            dtype=str(hidden_dtype),
            device=str(hidden_device),
            hidden_width=hidden_width,
        ).should_skip

    def update(
        self,
        t_mod: torch.Tensor,
        input_latent: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        return self._controller.apply_cached_update(
            branch=self.branch,
            modulated_input=t_mod,
            input_hidden_states=input_latent,
            step=step,
        )

    def store_truth(
        self,
        *,
        step: int,
        t_mod: torch.Tensor,
        input_latent: torch.Tensor,
        output_latent: torch.Tensor,
        sequence_length: int | None = None,
    ) -> None:
        del sequence_length
        self._controller.record_compute(
            branch=self.branch,
            step=step,
            modulated_input=t_mod,
            input_hidden_states=input_latent,
            output_hidden_states=output_latent,
        )


def _evaluate_polynomial(coefficients: tuple[float, ...], value: float) -> float:
    result = 0.0
    for coefficient in coefficients:
        result = result * value + coefficient
    return float(result)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0



__all__ = (
    "TeaCacheBranchAdapter",
    "TeaCacheController",
    "TeaCacheDecision",
)

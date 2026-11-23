"""Native inference graph: isolated request state, prewarm and fusion scopes."""

from __future__ import annotations

import os
import time
from copy import deepcopy

import torch

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.config.server_args import ServerArgs, set_global_server_args
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.utils.cpu_resources import (
    configure_process_cpu_resources,
    process_cpu_resources_snapshot,
)
from nndeploy.e_dit.utils.determinism import enable_deterministic_mode
from nndeploy.e_dit.parallel.runtime import (
    barrier_if_distributed,
    initialize_runtime_distributed,
)
from nndeploy.e_dit.algorithms.eraserdit.timing import diagnostic_timing_enabled


def _build_request_generator(
    server_args: ServerArgs,
    sampling_params: EraserDiTEraseSamplingParams,
) -> torch.Generator | None:
    seed = getattr(sampling_params, "seed", None)
    if seed is None:
        return None
    return torch.Generator(device=server_args.device).manual_seed(int(seed))


from nndeploy.e_dit.nodes.graph import StateGraph, register_type
from nndeploy.e_dit.algorithms.eraserdit.nodes.request import RequestGraph

@register_type
class InferenceGraph(StateGraph):
    def __init__(self, name='inference', inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        self.compose([RequestGraph()])

    def bind(self, resources):
        super().bind(resources)
        self.server_args = resources.config
        self.distributed_context = resources.distributed_context
        self._closed = False
        self._cpu_resources = resources._cpu_resources
        self.distributed_setup_s = resources.distributed_setup_s
        self.pipeline_initialize_s = 0

    def build_request(
        self,
        sampling_params: EraserDiTEraseSamplingParams,
        *,
        request_extra: dict[str, object] | None = None,
    ) -> Req:
        if self._closed:
            raise RuntimeError("EraserDiT inference graph is closed")
        req = Req(
            sampling_params=sampling_params,
            generator=_build_request_generator(self.server_args, sampling_params),
        )
        req.extra["runtime_distributed_metadata"] = self.distributed_context.as_dict()
        if request_extra:
            req.extra.update(request_extra)
        req.suppress_logs = bool(req.suppress_logs) or not (
            self.distributed_context.is_main_process
            or self.distributed_context.is_progress_rank
        )
        return req

    def _forward_with_operator_fusion(self, req: Req) -> Req:
        from nndeploy.e_dit.layers.operator_fusion.runtime import (
            operator_fusion_request_scope,
        )

        decision = self.server_args.operator_fusion_decision
        with operator_fusion_request_scope(decision) as fusion_stats:
            result = self.invoke(req)
        result.extra["operator_fusion"] = fusion_stats.snapshot()
        return result

    def infer(
        self,
        sampling_params: EraserDiTEraseSamplingParams,
        *,
        warmup_steps: int | None = None,
        request_extra: dict[str, object] | None = None,
    ) -> Req:
        request_start = time.perf_counter()
        req = self.build_request(sampling_params, request_extra=request_extra)
        request_prepare_s = time.perf_counter() - request_start
        warmup_summary: dict[str, object] | None = None
        if warmup_steps is not None:
            warmup_extra = dict(request_extra or {})
            # Keep cancellation active, but reserve request progress for the
            # real inference so warmup cannot report an early 100% completion.
            warmup_extra.pop('service_progress_state', None)
            warmup_req = self.build_request(deepcopy(sampling_params), request_extra=warmup_extra)
            warmup_req.set_as_warmup(int(warmup_steps))
            warmup_start = time.perf_counter()
            warmup_result = self._forward_with_operator_fusion(warmup_req)
            warmup_duration_s = time.perf_counter() - warmup_start
            warmup_metrics = getattr(warmup_result, "metrics", None)
            warmup_summary = {
                "mode": (
                    "compiled"
                    if bool(getattr(self.server_args, "enable_torch_compile", False))
                    else "eager"
                ),
                "steps": int(warmup_steps),
                "duration_seconds": warmup_duration_s,
                "pipeline_seconds": warmup_duration_s,
                "stage_breakdown_ms": (
                    dict(warmup_metrics.stages) if warmup_metrics is not None else {}
                ),
                "torch_compile": warmup_result.extra.get("torch_compile"),
                "operator_fusion": warmup_result.extra.get("operator_fusion"),
            }
            # Only the small summary is consumed below; do not retain a full
            # warmup video while the real request allocates its own window.
            del warmup_result, warmup_req
        pipeline_start = time.perf_counter()
        result = self._forward_with_operator_fusion(req)
        pipeline_duration_s = time.perf_counter() - pipeline_start
        if warmup_summary is not None:
            result.extra["warmup"] = warmup_summary
        if result.metrics is not None:
            result.metrics.total_duration_ms = pipeline_duration_s * 1000.0
        if self._cpu_resources is not None:
            result.extra["cpu_resources"] = process_cpu_resources_snapshot(
                self._cpu_resources
            )
        barrier_start = time.perf_counter()
        barrier_if_distributed()
        post_pipeline_barrier_s = time.perf_counter() - barrier_start
        if diagnostic_timing_enabled():
            result.extra["diagnostic_runner_phase_seconds"] = {
                "distributed_setup": self.distributed_setup_s,
                "pipeline_initialize": self.pipeline_initialize_s,
                "request_prepare": request_prepare_s,
                "pipeline_forward": pipeline_duration_s,
                "post_pipeline_barrier": post_pipeline_barrier_s,
                "through_post_pipeline_barrier": time.perf_counter() - request_start,
            }
        return result


"""Native request preparation, window loop and output finalization graph."""

from __future__ import annotations

import time
from functools import partial

import torch

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams, EraserDiTPipelineConfig
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.base import StateNode, operation
from nndeploy.e_dit.nodes.graph import StateGraph, IteratorLoop, register_type
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.algorithms.eraserdit.nodes import (
    EraserDiTEraseConditionEncodingNode,
    EraserDiTEraseDecodingNode,
    EraserDiTEraseDenoisingNode,
    EraserDiTEraseLatentPreparationNode,
    EraserDiTErasePreprocessNode,
    EraserDiTEraseTextEncodingNode,
    EraserDiTEraseTimestepPreparationNode,
    EraserDiTEraseWindowCommitSyncNode,
    EraserDiTEraseWindowPostprocessNode,
    EraserDiTEraseWindowValidationNode,
)
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import (
    TASK_STATE_KEY,
    EraserDiTTaskState,
)
from nndeploy.e_dit.algorithms.eraserdit.timing import record_diagnostic_stage
from nndeploy.e_dit.utils.logging_utils import init_logger
from nndeploy.e_dit.video.io import (
    read_mask_rgb_array as _read_mask_rgb_array,
)
from nndeploy.e_dit.algorithms.eraserdit.video.context import prepare_runtime_context
from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    _is_windowed_runtime_mode,
)
from nndeploy.e_dit.algorithms.eraserdit.video.windowed import run_windowed_runtime
from nndeploy.e_dit.algorithms.eraserdit.video.output import (
    close_runtime_resources,
    finalize_output,
)

logger = init_logger(__name__)


def _as_eraserdit_params(batch: Req) -> EraserDiTEraseSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, EraserDiTEraseSamplingParams):
        raise TypeError(
            "RequestGraph requires EraserDiTEraseSamplingParams, "
            f"got {type(params).__name__}"
        )
    return params


@register_type
class PrepareRequestNode(StateNode):
    def _prepare_global_context(
        self, batch: Req, server_args: ServerArgs
    ) -> EraseRuntimeContext:
        params = _as_eraserdit_params(batch)
        return prepare_runtime_context(
            batch=batch,
            params=params,
            server_args=server_args,
            resource_policy=server_args.resolve_resource_policy(),
            # The baseline thresholds each RGB channel of the raw mask stream at
            # ``255/2 * mask_threshold`` (utils/pre.py:257).  The shared readers
            # default to 0.3*max and to a luma decode, either of which flips a
            # small number of pixels.
            read_mask_array=partial(
                _read_mask_rgb_array,
                threshold_ratio=float(params.mask_threshold) / 2.0,
            ),
            memory_adapter=self.resources._memory_adapter,
        )

    def _install_task_state(
        self, batch: Req, server_args: ServerArgs
    ) -> EraserDiTTaskState:
        params = _as_eraserdit_params(batch)
        device = torch.device(server_args.device)
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(params.seed))
        state = EraserDiTTaskState(generator=generator)
        batch.extra[TASK_STATE_KEY] = state
        batch.generator = generator
        return state

    @operation
    def run(self, batch, server_args):
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.quantization import validate_quantization, runtime_report
        validate_quantization(server_args, batch)
        from nndeploy.e_dit.algorithms.eraserdit.parallel.cfg import validate_cfg_parallel
        validate_cfg_parallel(server_args, batch)
        from nndeploy.e_dit.algorithms.eraserdit.parallel.mesh import resolve_mesh
        resolve_mesh(server_args, batch)
        params = _as_eraserdit_params(batch)
        batch.modules = self.resources.modules if not batch.modules else batch.modules
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.memory_validation import validate_memory_config
        validate_memory_config(server_args, batch)
        from nndeploy.e_dit.algorithms.eraserdit.cache.config import resolve_eraserdit_cache_params
        resolve_eraserdit_cache_params(
            params, enable_torch_compile=server_args.enable_torch_compile,
            num_blocks=len(batch.modules["transformer"].transformer_blocks),
        )
        batch.extra["memory_registration_summary"] = dict(
            getattr(self.resources, "memory_registration_summary", {})
        )
        self._install_task_state(batch, server_args)

        context_prepare_start = time.perf_counter()
        context = self._prepare_global_context(batch, server_args)
        record_diagnostic_stage(
            batch.metrics,
            "diagnostic.pipeline.context_prepare",
            time.perf_counter() - context_prepare_start,
        )
        batch.extra['request_context'] = context
        return batch


@register_type
class FinalizeOutputNode(StateNode):
    @operation
    def run(self, batch, server_args):
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.quantization import runtime_report
        finalize_output(batch=batch, context=batch.extra['request_context'], params=batch.sampling_params)
        batch.extra['quantization'] = runtime_report(batch.modules['transformer'])
        manager = getattr(self.resources, '_component_compile', None)
        batch.extra['component_compile'] = manager.snapshot() if manager else {}
        return batch


@register_type
class WindowLoop(IteratorLoop):
    def __init__(self, name='windows', inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        self.compose([
            EraserDiTEraseWindowValidationNode(), EraserDiTEraseTextEncodingNode(),
            EraserDiTErasePreprocessNode(), EraserDiTEraseConditionEncodingNode(),
            EraserDiTEraseLatentPreparationNode(), EraserDiTEraseTimestepPreparationNode(),
            EraserDiTEraseDenoisingNode(), EraserDiTEraseDecodingNode(),
            EraserDiTEraseWindowPostprocessNode(), EraserDiTEraseWindowCommitSyncNode(),
        ])

    def iterations(self, batch):
        context = batch.extra['request_context']
        yield from run_windowed_runtime(batch=batch, context=context, params=batch.sampling_params,
                                       server_args=self._resources.config, logger=logger)


@register_type
class RequestGraph(StateGraph):
    def __init__(self, name='request', inputs=None, outputs=None):
        super().__init__(name, inputs, outputs)
        self.compose([PrepareRequestNode(), WindowLoop(), FinalizeOutputNode()])

    def run(self):
        batch = self.get_input(0).get(self).state
        try:
            with torch.no_grad():
                return super().run()
        finally:
            context = batch.extra.pop('request_context', None)
            if context is not None:
                close_runtime_resources(context)
            batch.extra[TASK_STATE_KEY] = None
            if self._resources is not None:
                batch.extra['memory_runtime'] = self._resources._memory_adapter.snapshot()

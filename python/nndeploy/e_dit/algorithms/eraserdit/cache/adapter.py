"""Window-owned EraserDiT caches with independent CFG branch state."""
from dataclasses import replace
from uuid import uuid4

import torch

from nndeploy.e_dit.cache.base import CacheBranch
from nndeploy.e_dit.algorithms.eraserdit.cache.text import EraserDiTTextCache
from nndeploy.e_dit.cache.teacache import TeaCacheController
from nndeploy.e_dit.algorithms.eraserdit.cache.prediction import EraserDiTTeaCacheController, EraserDiTCacheDitController
from nndeploy.e_dit.algorithms.eraserdit.cache.config import (
    MODEL_IDENTITY, resolve_eraserdit_cache_params, select_eraserdit_coefficients,
)


class EraserDiTCacheWindow:
    def __init__(self, batch, *, total_steps, num_blocks, enable_torch_compile=False,
                 sp_degree=1, sp_rank=0, cfg_degree=1, cfg_rank=0, coordinator=None):
        mode, tea, dbc, self.force_compute = resolve_eraserdit_cache_params(
            batch, enable_torch_compile=enable_torch_compile, num_blocks=num_blocks,
        )
        self.batch = batch
        self.mode = mode.value
        self.controller = None
        self._peak_retained_bytes = 0
        text_cache_setting = getattr(batch, 'cache_text_projections', None)
        enable_text_cache = self.mode != 'off' if text_cache_setting is None else text_cache_setting
        self.text_caches = (
            {branch: EraserDiTTextCache(self._record_retained_bytes) for branch in CacheBranch}
            if enable_text_cache else {}
        )
        if self.mode == 'off':
            return
        extra = batch.extra
        common = dict(
            request_id=str(getattr(batch, 'request_id', None) or getattr(batch, 'rid', None)
                           or extra.get('request_id') or uuid4()),
            object_index=int(extra.get('object_index', 0)),
            window_index=int(extra.get('window_index', 0)), total_steps=total_steps,
            sp_degree=sp_degree, sp_rank=sp_rank, cfg_degree=cfg_degree, cfg_rank=cfg_rank, coordinator=coordinator,
            sp_group_identity=f'eraserdit_peer_sp_cfg_{cfg_rank}' if sp_degree > 1 else 'eraserdit_serial',
            cfg_group_identity='eraserdit_peer_cfg' if cfg_degree > 1 else 'eraserdit_serial',
            model_identity=MODEL_IDENTITY,
        )
        if self.mode == 'teacache':
            self.controller = EraserDiTTeaCacheController(
                tea, residual_predictor=getattr(batch, 'cache_residual_predictor', 'none'),
                coefficient_selector=select_eraserdit_coefficients, **common,
            )
        else:
            if self.force_compute:
                dbc = replace(dbc, warmup_steps=max(total_steps, dbc.warmup_steps))
            self.controller = EraserDiTCacheDitController(
                dbc, residual_predictor=getattr(batch, 'cache_residual_predictor', 'none'),
                num_transformer_blocks=num_blocks, **common,
            )
        self.controller.probe_metric = getattr(batch, "cache_probe_metric", "global")
        self.controller._eraserdit_observe_retained = self._record_retained_bytes

    def _record_retained_bytes(self):
        storages = {}
        for cache in self.text_caches.values():
            storages.update(cache.retained_storages)
        if self.controller is not None:
            for state in (*self.controller._states.values(), *self.controller._forecasts.values()):
                for t in vars(state).values():
                    if isinstance(t, torch.Tensor):
                        storages[(str(t.device), t.untyped_storage().data_ptr())] = t.untyped_storage().nbytes()
            regions = self.controller._consensus.probe_regions
            if regions is not None:
                for t in (regions.weights, regions.frame_ids):
                    storages[(str(t.device), t.untyped_storage().data_ptr())] = t.untyped_storage().nbytes()
        self._peak_retained_bytes = max(self._peak_retained_bytes, sum(storages.values()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._record_retained_bytes()
        if self.controller is None:
            report = {'mode': 'off'}
        elif exc_type is None:
            report = self.controller.finish_window()
        else:
            report = self.controller.abort_window(exc_type.__name__)
        report['text_cache'] = {branch.value: cache.stats() for branch, cache in self.text_caches.items()}
        for cache in self.text_caches.values():
            cache.clear()
            cache.on_update = None
        if self.controller is not None:
            self.controller._eraserdit_observe_retained = None
        report['peak_retained_tensor_bytes'] = self._peak_retained_bytes
        report['cache_text_projections'] = bool(self.text_caches)
        if self.text_caches:
            report['closed'] = True
            report['aborted'] = exc_type is not None
        report['cache_probe_metric'] = getattr(self.batch, 'cache_probe_metric', 'global')
        if self.controller is not None:
            self.controller._consensus.probe_regions = None
        report['force_compute'] = self.force_compute
        report['experimental'] = self.mode != 'off'
        if self.mode == 'teacache':
            report['coefficient_calibrated'] = False
        self.batch.extra['transformer_cache'] = report
        return False

    def kwargs(self, branch, step):
        branch = CacheBranch(branch)
        kwargs = {'text_cache': self.text_caches[branch]} if self.text_caches else {}
        if self.controller is not None:
            kwargs['cache_adapter'] = EraserDiTCacheBranch(self.controller, branch, step)
        return kwargs


class EraserDiTCacheBranch:
    def __init__(self, controller, branch, step):
        self.controller, self.branch, self.step = controller, branch, step
        self.global_sequence_length = None

    def configure_probe_regions(self, mask, *, shape, shard=slice(None), patch_size=1, patch_size_t=1):
        if self.controller.probe_metric == 'global':
            return
        if patch_size != 1 or patch_size_t != 1:
            raise ValueError('mask_frame_max currently requires unit spatial/temporal patches')
        from nndeploy.e_dit.cache.regions import ProbeRegions
        self.controller._consensus.probe_regions = ProbeRegions(mask, shape=shape, shard=shard)

    def run(self, hidden_states, modulated_input, run_blocks, *, num_blocks, layout):
        try:
            return self._run(hidden_states, modulated_input, run_blocks, num_blocks=num_blocks, layout=layout)
        finally:
            # Include exact text projections and borrowed conditioning storage,
            # deduplicating storage across CFG branches and tensor views.
            observer = self.controller._eraserdit_observe_retained
            if observer is not None:
                observer()

    def _run(self, hidden_states, modulated_input, run_blocks, *, num_blocks, layout):
        c = self.controller
        common = dict(
            branch=self.branch, step=self.step,
            global_sequence_length=self.global_sequence_length or hidden_states.shape[1],
            local_sequence_length=hidden_states.shape[1],
            valid_local_sequence_length=hidden_states.shape[1],
            layout_signature=repr((layout, tuple(hidden_states.shape), str(hidden_states.dtype), str(hidden_states.device))),
            dtype=str(hidden_states.dtype), device=str(hidden_states.device),
            hidden_width=hidden_states.shape[-1],
        )
        if isinstance(c, TeaCacheController):
            decision = c.check(modulated_input=modulated_input, **common)
            if decision.should_skip:
                return c.apply_cached_update(
                    branch=self.branch, step=self.step, modulated_input=modulated_input,
                    input_hidden_states=hidden_states,
                )
            output = run_blocks(hidden_states, 0, num_blocks)
            c.record_compute(
                branch=self.branch, step=self.step, modulated_input=modulated_input,
                input_hidden_states=hidden_states, output_hidden_states=output,
            )
            return output
        front = run_blocks(hidden_states, 0, c.front_end)
        decision = c.check(input_hidden_states=hidden_states, front_output_hidden_states=front, **common)
        if decision.should_reuse_middle:
            middle = c.apply_cached_middle(branch=self.branch, step=self.step, front_output_hidden_states=front)
        else:
            middle = run_blocks(front, c.front_end, c.middle_end)
            c.record_middle_compute(
                branch=self.branch, step=self.step,
                front_output_hidden_states=front, middle_output_hidden_states=middle,
            )
        output = run_blocks(middle, c.back_start, num_blocks)
        c.complete_step(branch=self.branch, step=self.step)
        return output


def aggregate_rank_cache_reports(reports, *, sp_degree):
    """Count logical CFG work once per SP group; retain physical-rank details."""
    leaders = reports[::sp_degree]
    result = dict(reports[0], ranks=reports, summary_scope='logical_cfg_branches_sp_leaders')
    result['peak_retained_tensor_bytes'] = sum(r['peak_retained_tensor_bytes'] for r in reports)
    result['retained_peak_scope'] = 'sum_of_rank_peaks_upper_bound'
    result['text_cache'] = {branch: value for r in leaders for branch, value in r['text_cache'].items()
                            if value.get('peak_retained_tensor_bytes', 0)}
    # Controllers also report zero-valued states for the unassigned CFG branch.
    result['branches'] = {branch: value for r in leaders for branch, value in r.get('branches', {}).items()
                          if value.get('calc_steps', 0) + value.get('skip_steps', 0) + value.get('completed_steps', 0)}
    result['communication_scope'] = 'representative_rank_0; see ranks for all counters'
    if result['mode'] == 'teacache':
        calc = sum(r['total']['calc_steps'] for r in leaders)
        skip = sum(r['total']['skip_steps'] for r in leaders)
        result['total'] = dict(calc_steps=calc, compute_steps=calc, skip_steps=skip,
            cache_hit_rate=skip/max(1, calc+skip), hit_rate=skip/max(1, calc+skip),
            estimated_block_speedup=(calc+skip)/max(1, calc))
    elif result['mode'] == 'cache_dit':
        keys = ('computed_middle_steps', 'cached_middle_steps', 'effective_blocks_executed', 'completed_steps')
        total = {key: sum(r['total'][key] for r in leaders) for key in keys}
        total['middle_skip_ratio'] = total['cached_middle_steps']/max(1, total['completed_steps'])
        # Same block count and completed-step count in every CFG group.
        completed = total['completed_steps']
        total['estimated_block_ratio'] = sum(r['total']['estimated_block_ratio'] * r['total']['completed_steps']
                                             for r in leaders)/max(1, completed)
        result['total'] = total
    return result

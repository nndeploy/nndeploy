"""EraserDiT residual forecasting, inspired by SGLang's TaylorSeer option.

Only real block evaluations train the first-order predictor. Residual arithmetic
uses float32 so subtracting and restoring bf16 activations does not add another
rounding error. The block interface always retains the original activation dtype.
"""
from nndeploy.e_dit.cache.prediction import ResidualPredictionMixin


from nndeploy.e_dit.cache.cache_dit import CacheDitController
from nndeploy.e_dit.cache.teacache import TeaCacheController


class EraserDiTTeaCacheController(ResidualPredictionMixin, TeaCacheController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._probe_trace = {branch: [] for branch in self._states}

    def check(self, **kwargs):
        decision = super().check(**kwargs)
        self._probe_trace[kwargs['branch']].append({
            'step': kwargs['step'], 'reason': decision.reason,
            'estimated_distance': decision.estimated_distance,
            'accumulated_distance': decision.accumulated_distance,
        })
        return decision

    def stats(self):
        report = super().stats()
        report['probe'] = 'first_block_modulated_input'
        report['probe_trace'] = {branch.value: list(trace) for branch, trace in self._probe_trace.items()}
        return report

    def record_compute(self, *, branch, step, input_hidden_states, output_hidden_states, **kwargs):
        self._require_open()
        state = self._states[branch]
        if state.pending is None or state.pending.should_skip or state.pending_step != step:
            raise RuntimeError("TeaCache compute result has no pending compute decision")
        previous, reason = state.previous_residual, state.pending.reason
        layout = state.pending_layout_signature
        super().record_compute(branch=branch, step=step, input_hidden_states=input_hidden_states,
                               output_hidden_states=output_hidden_states, **kwargs)
        residual = state.previous_residual
        self._learn(branch, step, previous, residual, reason, layout)

    def apply_cached_update(self, *, branch, step, input_hidden_states, **kwargs):
        output = super().apply_cached_update(branch=branch, step=step,
                                             input_hidden_states=input_hidden_states, **kwargs)
        return self._predict(branch, step, output, input_hidden_states.dtype)


class EraserDiTCacheDitController(ResidualPredictionMixin, CacheDitController):
    def record_middle_compute(self, *, branch, step, front_output_hidden_states, middle_output_hidden_states):
        self._require_pending(branch, step, should_reuse=False)
        state = self._states[branch]
        previous, reason = state.cached_middle_residual, state.pending.reason
        layout = state.pending_layout_signature
        super().record_middle_compute(branch=branch, step=step,
                                      front_output_hidden_states=front_output_hidden_states,
                                      middle_output_hidden_states=middle_output_hidden_states)
        residual = state.cached_middle_residual
        self._learn(branch, step, previous, residual, reason, layout)

    def apply_cached_middle(self, *, branch, step, front_output_hidden_states):
        output = super().apply_cached_middle(branch=branch, step=step,
                                            front_output_hidden_states=front_output_hidden_states)
        return self._predict(branch, step, output, front_output_hidden_states.dtype)

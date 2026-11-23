"""First-order residual forecasting for branch-owned cache controllers."""
from dataclasses import dataclass
import torch


@dataclass
class _Forecast:
    step: int | None = None
    layout: str | None = None
    interval: int = 0
    velocity: torch.Tensor | None = None
    predicted_steps: int = 0


class ResidualPredictionMixin:
    def __init__(self, *args, residual_predictor="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_predictor = residual_predictor
        self._forecasts = {branch: _Forecast() for branch in self._states}

    @staticmethod
    def _compute_residual(input_hidden_states, output_hidden_states):
        return (output_hidden_states.float() - input_hidden_states.float()).detach()

    def _learn(self, branch, step, previous, residual, reason, layout):
        state = self._forecasts[branch]
        state.velocity = None
        if (self.residual_predictor == "linear" and previous is not None
                and state.step is not None and step > state.step
                and state.layout == layout and previous.shape == residual.shape
                and reason not in ("layout_changed", "non_contiguous_branch_step")):
            state.interval = step - state.step
            state.velocity = (residual - previous) / state.interval
        state.step = step
        state.layout = layout

    def _predict(self, branch, step, output, dtype):
        state = self._forecasts[branch]
        if state.velocity is not None:
            # Bound extrapolation to one observed interval, including when a
            # caller allows a long run of cache hits. Never train on predictions.
            horizon = min(step - state.step, state.interval)
            output = output + state.velocity * horizon
            state.predicted_steps += 1
        return output.to(dtype)

    def stats(self):
        report = super().stats()
        report["residual_predictor"] = self.residual_predictor
        report["residual_dtype"] = "torch.float32"
        report["predicted_steps"] = sum(s.predicted_steps for s in self._forecasts.values())
        return report

    def _close(self, *, abort_reason):
        try:
            return super()._close(abort_reason=abort_reason)
        finally:
            for state in self._forecasts.values():
                state.velocity = None

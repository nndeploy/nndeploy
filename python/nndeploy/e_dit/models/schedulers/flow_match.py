"""Shared flow-match scheduler wrapper."""

from __future__ import annotations

from typing import Optional

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as _DiffusersFlowMatchEulerDiscreteScheduler,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)

from nndeploy.e_dit.models.schedulers.base import BaseScheduler


class FlowMatchEulerDiscreteScheduler(
    _DiffusersFlowMatchEulerDiscreteScheduler, BaseScheduler
):
    """Official flow-match scheduler with the local runtime compatibility layer."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
            invert_sigmas=invert_sigmas,
            shift_terminal=shift_terminal,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
            time_shift_type=time_shift_type,
            stochastic_sampling=stochastic_sampling,
        )
        self.num_train_timesteps = self.config.num_train_timesteps
        BaseScheduler.__init__(self)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: int | None = None
    ) -> torch.Tensor:
        del timestep
        return sample

    def set_shift(self, shift: float) -> None:
        self._shift = shift


EntryClass = FlowMatchEulerDiscreteScheduler


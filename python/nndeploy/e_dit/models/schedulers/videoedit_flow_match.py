# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch


class VideoEditFlowMatchScheduler:
    """FlowMatch scheduler used by VideoEdit-diffusers."""

    order = 1

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        inverse_timesteps: bool = False,
        extra_one_step: bool = True,
        reverse_sigmas: bool = False,
        **_: Any,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_shift(self, shift: float) -> None:
        self.shift = shift

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
        device: torch.device | str | None = None,
        **_: Any,
    ) -> None:
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * denoising_strength
        steps = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        sigmas = torch.linspace(sigma_start, self.sigma_min, steps, device="cpu")[
            :num_inference_steps
        ]
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.reverse_sigmas:
            sigmas = 1 - sigmas
        # CPU zero-dimensional sigma tensors follow PyTorch's wrapped-scalar
        # promotion rules. Moving them to CUDA changes BF16 update rounding.
        self.sigmas = sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if device is not None:
            self.timesteps = self.timesteps.to(device=device)
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (
                num_inference_steps / y_shifted.sum()
            )

    def _sigma_for_timestep(self, timestep) -> torch.Tensor:
        timestep_cpu = timestep.detach().cpu() if isinstance(timestep, torch.Tensor) else timestep
        timestep_id = torch.argmin((self.timesteps.detach().cpu() - timestep_cpu).abs())
        return self.sigmas[timestep_id].to(
            device=timestep.device if isinstance(timestep, torch.Tensor) else self.sigmas.device
        )

    def step(self, model_output, timestep, sample, to_final: bool = False, **_: Any):
        timestep_cpu = timestep.detach().cpu() if isinstance(timestep, torch.Tensor) else timestep
        timestep_id = torch.argmin((self.timesteps.detach().cpu() - timestep_cpu).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma)

    def add_noise(self, original_samples, noise, timestep):
        sigma = self._sigma_for_timestep(timestep).to(original_samples.device, original_samples.dtype)
        return (1 - sigma) * original_samples + sigma * noise

    def get_timesteps(self, num_inference_steps, timesteps, strength):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        return timesteps[t_start:], num_inference_steps - t_start


EntryClass = VideoEditFlowMatchScheduler

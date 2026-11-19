"""EraserDiT erase – latent preparation stage.

Port of the second half of ``LTXVideoToVideoPipeline.prepare_latents`` plus the
timestep construction from ``__call__`` (``linear_quadratic_schedule`` at
``num_inference_steps`` then trimmed by ``strength``).
"""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

import torch
from diffusers.utils.torch_utils import randn_tensor

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import (
    field_summary,
    get_task_state,
    get_timesteps,
    linear_quadratic_schedule,
    retrieve_timesteps,
)


class EraserDiTEraseLatentPreparationNode(StateNode):
    def setup(self, scheduler):
        self._scheduler = scheduler

    @operation
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        state = get_task_state(batch)
        cond_latents = batch.cond_latents
        if cond_latents is None:
            raise ValueError("latent preparation requires condition_encoding to run first")

        sigmas = linear_quadratic_schedule(int(batch.num_inference_steps))
        timesteps = sigmas * 1000
        timesteps, num_inference_steps = retrieve_timesteps(
            self._scheduler,
            int(batch.num_inference_steps),
            cond_latents.device,
            timesteps=timesteps,
        )
        timesteps, num_inference_steps = get_timesteps(
            self._scheduler, num_inference_steps, float(batch.strength), cond_latents.device
        )
        latent_timestep = timesteps[:1].repeat(1)

        noise = randn_tensor(
            cond_latents.shape,
            generator=state.generator,
            device=cond_latents.device,
            dtype=torch.float32,
        )
        batch.timesteps = timesteps
        batch.latent_timestep = latent_timestep
        with torch.autocast("cuda", dtype=torch.bfloat16):
            batch.latents = self._scheduler.scale_noise(
                sample=cond_latents, timestep=latent_timestep, noise=noise
            )
        batch.extra["effective_inference_steps"] = num_inference_steps
        self.log_info(
            "%s | steps=%d",
            field_summary("latents", batch.latents),
            len(timesteps),
        )
        return batch

"""EraserDiT erase – decoding stage.

Port of the decode tail of ``LTXVideoToVideoPipeline.__call__``.  The decode noise
draw is numerically void (``decode_noise_scale == 0.0``) but still consumes the
request generator, so it must be issued to preserve the per-window draw order.
"""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.algorithms.eraserdit.optimizations.component_offload import offload_component
from nndeploy.e_dit.algorithms.eraserdit.optimizations.memory_phase_controller import MemoryPhase
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import (
    denormalize_latents,
    field_summary,
    get_task_state,
)
from nndeploy.e_dit.memory.tensor_ops import module_device, module_dtype


class EraserDiTEraseDecodingNode(StateNode):
    @operation
    @offload_component("vae", phase=MemoryPhase.VAE_DECODE)
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        vae = batch.modules["vae"]
        state = get_task_state(batch)
        latents = batch.latents
        if latents is None:
            raise ValueError("decoding requires denoising to run first")

        device = module_device(vae)
        vae_dtype = module_dtype(vae)

        latents = denormalize_latents(
            latents,
            vae.latents_mean,
            vae.latents_std,
            scaling_factor=float(vae.config.scaling_factor),
        )
        latents = latents.to(vae_dtype)

        # Drawn but numerically inert at decode_noise_scale == 0.0.
        noise = randn_tensor(
            latents.shape,
            generator=state.generator,
            device=device,
            dtype=latents.dtype,
        )
        decode_timestep = torch.tensor(
            [float(batch.decode_timestep)], device=device, dtype=latents.dtype
        )
        decode_noise_scale = torch.tensor(
            [float(batch.decode_noise_scale)], device=device, dtype=latents.dtype
        )[:, None, None, None, None]
        latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            from nndeploy.e_dit.algorithms.eraserdit.parallel.vae import tiled_vae
            video = tiled_vae(vae, latents, server_args, batch, operation="decode", temb=decode_timestep)

        processor = VideoProcessor(
            vae_scale_factor=int(getattr(vae, "spatial_compression_ratio", 32))
        )
        # [B, C, F, H, W] -> [B, F, C, H, W], denormalised and clamped to [0, 1].
        decoded = processor.postprocess_video(video, output_type="pt")
        batch.decoded_video = decoded.permute(0, 2, 1, 3, 4).contiguous()
        batch.latents = None
        self.log_info("%s", field_summary("decoded_video", batch.decoded_video))
        return batch

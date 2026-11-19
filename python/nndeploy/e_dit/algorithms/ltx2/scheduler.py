# Migrated from SGLang multimodal_gen (Apache-2.0); see ltx2_migration.json.
import os
import math
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.utils.logging_utils import init_logger
logger=init_logger(__name__)
BASE_SHIFT_ANCHOR=1024
MAX_SHIFT_ANCHOR=4096
def _resolve_ltx2_two_stage_component_paths(
    model_path: str, component_paths: dict[str, str]
) -> dict[str, str]:
    resolved = dict(component_paths)
    auto_resolved = []

    if "spatial_upsampler" not in resolved:
        spatial_candidates = [
            os.path.join(model_path, "latent_upsampler"),
            os.path.join(model_path, "ltx-2-spatial-upscaler-x2-1.0.safetensors"),
        ]
        for candidate in spatial_candidates:
            if os.path.exists(candidate):
                resolved["spatial_upsampler"] = candidate
                auto_resolved.append(f"spatial_upsampler={candidate}")
                break

    if "distilled_lora" not in resolved:
        distilled_lora = os.path.join(
            model_path, "ltx-2-19b-distilled-lora-384.safetensors"
        )
        if os.path.exists(distilled_lora):
            resolved["distilled_lora"] = distilled_lora
            auto_resolved.append(f"distilled_lora={distilled_lora}")

    if auto_resolved:
        logger.info(
            "Auto-resolved LTX2 two-stage components: %s", ", ".join(auto_resolved)
        )

    return resolved

def calculate_ltx2_shift(
    image_seq_len: int,
    base_seq_len: int = BASE_SHIFT_ANCHOR,
    max_seq_len: int = MAX_SHIFT_ANCHOR,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    mm = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - mm * base_seq_len
    return image_seq_len * mm + b

def prepare_ltx2_mu(batch: Req, server_args: ServerArgs):
    latent_num_frames = (int(batch.num_frames) - 1) // int(
        server_args.pipeline_config.vae_temporal_compression
    ) + 1
    latent_height = int(batch.height) // int(
        server_args.pipeline_config.vae_scale_factor
    )
    latent_width = int(batch.width) // int(server_args.pipeline_config.vae_scale_factor)
    video_sequence_length = latent_num_frames * latent_height * latent_width
    return "mu", calculate_ltx2_shift(video_sequence_length)

class LTX2FlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Override ``_time_shift_exponential`` to use torch f32 instead of numpy f64."""

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):
        if sigmas is not None and timesteps is None and mu is None:
            sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
            timesteps = sigmas * self.config.num_train_timesteps
            sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
            self.num_inference_steps = len(timesteps)
            self.timesteps = timesteps
            self.sigmas = sigmas
            self._step_index = None
            self._begin_index = None
            return

        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )

    def _time_shift_exponential(self, mu, sigma, t):
        if isinstance(t, np.ndarray):
            t_torch = torch.from_numpy(t).to(torch.float32)
            result = math.exp(mu) / (math.exp(mu) + (1 / t_torch - 1) ** sigma)
            return result.numpy()
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

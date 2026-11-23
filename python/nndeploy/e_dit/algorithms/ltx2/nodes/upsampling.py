# Migrated from SGLang multimodal_gen (Apache-2.0); see ltx2_migration.json.
from nndeploy.e_dit.nodes.base import operation
from nndeploy.e_dit.nodes.graph import register_type
import torch

from nndeploy.e_dit.platforms import get_local_torch_device
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2HalveResolutionNode(StateNode):
    """Halve batch height/width for two-stage Stage 1 (low-res generation)."""

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        original_h, original_w = batch.height, batch.width

        vae_scale_factor = getattr(server_args.pipeline_config, "vae_scale_factor", 32)
        required_alignment = max(64, int(vae_scale_factor) * 2)
        if original_h % required_alignment != 0 or original_w % required_alignment != 0:
            raise ValueError(
                "LTX-2 two-stage requires resolution divisible by "
                f"{required_alignment}, got ({original_h}x{original_w})."
            )

        batch.height = batch.height // 2
        batch.width = batch.width // 2
        logger.info(
            "Halved resolution: %dx%d -> %dx%d",
            original_h,
            original_w,
            batch.height,
            batch.width,
        )
        return batch


class LTX2LoRASwitchNode(StateNode):
    """Switch LoRA configuration for the requested two-stage phase."""

    def setup(self, phase="stage1"):
        pipeline = self.resources._ltx_controller
        self.pipeline = pipeline
        self.phase = phase

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        switch_fn = getattr(self.pipeline, "switch_lora_phase", None)
        if not callable(switch_fn):
            raise ValueError(
                "LTX2LoRASwitchNode requires pipeline.switch_lora_phase()"
            )
        switch_fn(self.phase)
        batch.extra["ltx2_phase"] = self.phase
        return batch


class LTX2UpsampleNode(StateNode):
    """Upsample Stage-1 video latents and prepare Stage-2 inputs."""

    def setup(self, spatial_upsampler, vae, audio_vae=None):
        self.spatial_upsampler = spatial_upsampler
        self.vae = vae
        self.audio_vae = audio_vae

    def _upsample_video_latents(
        self, latents: torch.Tensor, server_args: ServerArgs, device: torch.device
    ) -> torch.Tensor:
        vae_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            device=device, dtype=latents.dtype
        )
        vae_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(
            device=device, dtype=latents.dtype
        )
        latents = latents * vae_std + vae_mean
        self.spatial_upsampler = self.spatial_upsampler.to(
            device=device, dtype=latents.dtype
        )
        latents = self.spatial_upsampler(latents)
        if server_args.vae_cpu_offload:
            self.spatial_upsampler = self.spatial_upsampler.to("cpu")
        latents = (latents - vae_mean) / vae_std
        return latents

    @staticmethod
    def _restore_full_resolution(batch: Req) -> None:
        batch.height *= 2
        batch.width *= 2

    @staticmethod
    def _pack_video_latents(
        batch: Req, latents: torch.Tensor, server_args: ServerArgs
    ) -> None:
        batch_size = latents.shape[0]
        latents = server_args.pipeline_config.maybe_pack_latents(
            latents, batch_size, batch
        )
        batch.latents = latents
        batch.raw_latent_shape = latents.shape

    def _repack_audio_latents(self, batch: Req, server_args: ServerArgs) -> None:
        if batch.audio_latents is None or self.audio_vae is None:
            return
        audio_latents = server_args.pipeline_config.maybe_pack_audio_latents(
            batch.audio_latents, batch.audio_latents.shape[0], batch
        )
        batch.audio_latents = audio_latents
        batch.raw_audio_latent_shape = audio_latents.shape
        logger.info(
            "Re-packed audio latents for Stage 2: %s", list(audio_latents.shape)
        )

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        latents = self._upsample_video_latents(batch.latents, server_args, device)
        logger.info("Upsampled video latents: %s", list(latents.shape))
        self._restore_full_resolution(batch)
        self._pack_video_latents(batch, latents, server_args)
        logger.info(
            "Packed video latents for Stage 2: %s (resolution %dx%d)",
            list(batch.latents.shape),
            batch.height,
            batch.width,
        )
        self._repack_audio_latents(batch, server_args)
        return batch


LTX2HalveResolutionNode.run = operation(LTX2HalveResolutionNode.compute)
register_type(LTX2HalveResolutionNode)

LTX2LoRASwitchNode.run = operation(LTX2LoRASwitchNode.compute)
register_type(LTX2LoRASwitchNode)

LTX2UpsampleNode.run = operation(LTX2UpsampleNode.compute)
register_type(LTX2UpsampleNode)

@register_type
class LTX2RefinementLoRANode(LTX2LoRASwitchNode):
    def setup(self):
        super().setup(phase="stage2")

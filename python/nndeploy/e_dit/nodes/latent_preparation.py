# Migrated from SGLang multimodal_gen (Apache-2.0); see ltx2_migration.json.
from nndeploy.e_dit.nodes.base import operation
from nndeploy.e_dit.nodes.graph import register_type
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""

from diffusers.utils.torch_utils import randn_tensor

from nndeploy.e_dit.platforms import (
    get_local_torch_device,
)
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.nodes.validators import (
    StageValidators as V,
)
from nndeploy.e_dit.nodes.validators import (
    VerificationResult,
)
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LatentPreparationNode(StateNode):
    """
    Stage for preparing initial latent variables for the diffusion process.

    This stage handles the preparation of the initial latent variables that will be
    denoised during the diffusion process.
    """

    def setup(self, scheduler, transformer) -> None:
        self.scheduler = scheduler
        self.transformer = transformer

    def _get_latent_dtype(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        return server_args.pipeline_config.get_latent_dtype(
            batch.prompt_embeds[0].dtype
        )

    def compute(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Prepare initial latent variables for the diffusion process.



        Returns:
            The batch with prepared latent variables.
        """

        # Adjust video length based on VAE version if needed
        latent_num_frames = self.adjust_video_length(batch, server_args)

        batch_size = batch.batch_size

        # Get required parameters
        dtype = self._get_latent_dtype(batch, server_args)
        device = get_local_torch_device()
        generator = batch.generator
        latents = batch.latents
        num_frames = (
            latent_num_frames if latent_num_frames is not None else batch.num_frames
        )
        height = batch.height
        width = batch.width

        # TODO(will): remove this once we add input/output validation for stages
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        # Validate generator if it's a list
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Generate or use provided latents
        if latents is None:
            shape = server_args.pipeline_config.prepare_latent_shape(
                batch, batch_size, num_frames
            )
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )

            latent_ids = server_args.pipeline_config.maybe_prepare_latent_ids(latents)

            if latent_ids is not None:
                batch.latent_ids = latent_ids.to(device=device)

            latents = server_args.pipeline_config.maybe_pack_latents(
                latents, batch_size, batch
            )
        else:
            latents = latents.to(device)

        # Scale the initial noise if needed
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        # Update batch with prepared latents
        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        return batch

    def adjust_video_length(self, batch: Req, server_args: ServerArgs) -> int:
        """
        Adjust video length based on VAE version.
        """

        video_length = batch.num_frames
        latent_num_frames = video_length
        use_temporal_scaling_frames = (
            server_args.pipeline_config.vae_config.use_temporal_scaling_frames
        )
        if use_temporal_scaling_frames:
            temporal_scale_factor = (
                server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
            )
            latent_num_frames = (video_length - 1) // temporal_scale_factor + 1
        return int(latent_num_frames)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify latent preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds),
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors)
        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify latent preparation stage outputs."""
        result = VerificationResult()
        if batch.debug:
            logger.debug(f"{batch.raw_latent_shape=}")
        # disable temporarily for image-generation models
        # result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.is_tuple)
        return result


LatentPreparationNode.run = operation(LatentPreparationNode.compute)
register_type(LatentPreparationNode)

"""LTX-2 denoising bindings; numerical steps are scheduled by native Loop."""
from contextlib import contextmanager
import torch
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.utils.precision import PRECISION_TO_TYPE

class DenoisingNode(StateNode):
    def setup(self, transformer, scheduler, vae=None, pipeline=None):
        self.transformer, self.scheduler, self.vae = transformer, scheduler, vae
        self.pipeline = self.resources._ltx_controller if pipeline is None else pipeline

    def _prepare_denoising_loop(self, batch, args):
        # Schedulers belong to the request; refinement supplies its own copy.
        if batch.extra.get('ltx2_phase') != 'stage2':
            self.scheduler = batch.extra['scheduler']
        self.transformer.to(args.device)
        dtype = args.resolve_component_dtype('transformer') or PRECISION_TO_TYPE[args.pipeline_config.dit_precision]
        batch.did_sp_shard_latents = False
        return dict(target_dtype=dtype, autocast_enabled=dtype != torch.float32 and not args.disable_autocast,
                    timesteps=batch.timesteps, num_inference_steps=batch.num_inference_steps,
                    num_warmup_steps=0, latents=batch.latents, boundary_timestep=None,
                    z=None, reserved_frames_mask=None)

    def _select_and_manage_model(self, *, batch, **kwargs):
        return self.transformer, batch.guidance_scale

    def _build_attn_metadata(self, i, batch, args):
        from nndeploy.e_dit.layers.attention import AttentionMetadata
        return AttentionMetadata(current_timestep=i)

    def _postprocess_sp_latents(self, batch, latents, trajectory):
        return latents, trajectory

    def post_forward_for_ti2v_task(self, batch, args, mask, latents, z):
        # LTX image conditioning is applied by the LTX numerical step itself.
        return latents

    @contextmanager
    def progress_bar(self, total):
        owner = self
        class Progress:
            index = 0
            def update(self):
                progress = owner._active_batch.extra.get('service_progress_state')
                if progress is not None:
                    progress.update_denoise(self.index, total)
                self.index += 1
        yield Progress()

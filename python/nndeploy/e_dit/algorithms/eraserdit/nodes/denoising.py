"""EraserDiT erase – denoising stage.

Port of the ``LTXVideoToVideoPipeline`` denoising loop.  The baseline expands the
CFG batch to two and slices ``[0:1]`` / ``[1:]`` before the forward; because both
halves are identical copies that is exactly two batch-1 forwards, which is what
this stage issues directly.
"""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation, StateNode
from nndeploy.e_dit.nodes.graph import IteratorLoop, register_type

import time
from contextlib import nullcontext

import torch

from nndeploy.e_dit.algorithms.eraserdit.cache.adapter import EraserDiTCacheWindow
from nndeploy.e_dit.algorithms.eraserdit.parallel.cfg import EraserDiTCFGWindow, validate_cfg_parallel
from nndeploy.e_dit.algorithms.eraserdit.parallel.mesh import EraserDiTMeshWindow, resolve_mesh

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.control import service_checkpoint
from nndeploy.e_dit.nodes.compile import DenoisingNode
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import (
    field_summary,
    MODEL_FRAMES_KEY,
    latent_frame_count,
)


def self_attention_backend_report(transformer) -> dict | None:
    """Effective self-attention backend, read off a live block processor."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        return None
    processor = getattr(blocks[0].attn1, "processor", None)
    report = getattr(processor, "attention_backend_report", None)
    return dict(report()) if callable(report) else None


@register_type
class EraserDiTEraseDenoisingNode(IteratorLoop, DenoisingNode):
    """Denoising with selectable FFN/whole-DiT compilation and rank residency."""

    def __init__(self, name=None, inputs=None, outputs=None, **bindings):
        IteratorLoop.__init__(self, name or 'denoise', inputs, outputs)
        self._bindings = bindings
        self._configured = False
        self._enable_logging = True
        self.compose([DiTPredictNode('predict'), SchedulerStepNode('scheduler')])

    def bind(self, resources):
        StateNode.bind(self, resources)
        IteratorLoop.bind(self, resources)

    def iterations(self, batch):
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.memory_phase_controller import MemoryPhase
        controller = batch.extra.get('memory_phase_controller')
        if controller is None and self.server_args.resolve_resource_policy().enabled:
            raise RuntimeError('Model offload requires a request memory controller')
        device = torch.device(self.server_args.device)
        scope = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
        with scope:
            try:
                if controller is not None:
                    controller.enter(MemoryPhase.DENOISE, component_name='transformer',
                                     window_key=(int(batch.extra.get('object_index', -1)),
                                                 int(batch.extra.get('window_index', -1))))
                yield from self.window_steps(batch, self.server_args)
            finally:
                batch.extra.pop('denoise_step', None)
                if controller is not None and controller.active_phase is MemoryPhase.DENOISE:
                    controller.exit(MemoryPhase.DENOISE)

    def setup(self, transformer, scheduler, server_args=None):
        if server_args is None:
            from nndeploy.e_dit.config.server_args import get_global_server_args

            server_args = get_global_server_args()
        validate_cfg_parallel(server_args)
        DenoisingNode.setup(self, transformer, server_args)
        self._scheduler = scheduler
        plan = resolve_mesh(server_args)
        self._replica_pool = None
        if plan is not None and plan.get('backend') == 'nccl':
            from nndeploy.e_dit.algorithms.eraserdit.parallel.workers import DiTProcessPool
            self._replica_pool = DiTProcessPool(transformer, plan, server_args)
        elif plan is not None and plan['sp'] * plan['cfg'] > 1:
            from nndeploy.e_dit.algorithms.eraserdit.parallel.replicas import EraserDiTReplicaPool
            self._replica_pool = EraserDiTReplicaPool(
                transformer, plan, server_args, compiled_transformer=self._compiled_transformer)

    def deinit(self):
        try:
            return IteratorLoop.deinit(self)
        finally:
            try:
                self.close()
            finally:
                self.release_bindings()

    def close(self):
        if getattr(self, "_replica_pool", None) is not None:
            self._replica_pool.close()


    def register_torch_compile(self):
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.block_compile import configure_block_compile
        from nndeploy.e_dit.nodes.compile import resolve_torch_compile_mode
        if self._compile_registration_attempted:
            return
        self._compile_registration_attempted = True
        started = time.perf_counter()
        if self._compile_server_args.torch_compile_scope == "transformer":
            from nndeploy.e_dit.algorithms.eraserdit.optimizations.transformer_compile import configure_transformer_compile
            from nndeploy.e_dit.algorithms.eraserdit.optimizations.compile_config import validate_transformer_compile
            validate_transformer_compile(self._compile_server_args)
            self._compiled_transformer = configure_transformer_compile(
                self._transformer, mode=resolve_torch_compile_mode())
            self._compile_status.applied = True
            self._compile_status.mode = resolve_torch_compile_mode()
            self._compile_status.compile_seconds = time.perf_counter() - started
            return
        report = configure_block_compile(self._transformer, mode=resolve_torch_compile_mode())
        self._compiled_transformer = self._transformer
        self._compile_status.applied = True
        self._compile_status.mode = report['mode']
        self._compile_status.compile_seconds = time.perf_counter() - started

    def fallback_to_eager(self, reason):
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.block_compile import remove_block_compile
        remove_block_compile(self._transformer)
        if self._replica_pool is not None:
            self._replica_pool.compiled_forwards.clear()
            for model in self._replica_pool.models[1:]:
                remove_block_compile(model)
        super().fallback_to_eager(reason)

    def compile_status_snapshot(self):
        report = super().compile_status_snapshot()
        report.update(getattr(self._transformer, '_block_compile_report', {}))
        if self._compile_server_args.torch_compile_scope == 'transformer':
            report.update(getattr(self._compiled_transformer, 'report', {}))
            report['scope'] = 'transformer'
            pool = getattr(self, '_replica_pool', None)
            if pool is not None and pool.compiled_forwards:
                from copy import deepcopy
                report['rank_compile'] = [deepcopy(fn.report) for fn in pool.compiled_forwards]
                report['successful_forwards'] = sum(fn.report['successful_forwards'] for fn in pool.compiled_forwards)
        else:
            report['scope'] = 'block_ffn'
        report['signature_scope'] = 'window_input_before_sp_partition'
        config = self._compile_server_args.pipeline_config
        for signature in report['signatures']:
            signature['sp_degree'] = config.sp_degree
            signature['cfg_degree'] = config.cfg_degree
        report['cudagraphs'] = False
        return report

    def window_steps(self, batch: Req, server_args: ServerArgs):
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.compile_config import validate_transformer_compile
        validate_transformer_compile(server_args, batch)
        transformer = batch.modules.get("transformer") or self._transformer
        scheduler = batch.modules.get("scheduler") or self._scheduler

        latents = batch.latents
        cond_latents = batch.cond_latents
        mask_values = batch.mask_values
        timesteps = batch.timesteps
        if latents is None or cond_latents is None or timesteps is None:
            raise ValueError("denoising requires latent_preparation to run first")

        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        negative_prompt_embeds = batch.negative_prompt_embeds
        negative_attention_mask = batch.negative_attention_mask
        guidance_scale = float(batch.guidance_scale)
        rope_interpolation_scale = batch.rope_interpolation_scale

        model_frames = int(batch.extra[MODEL_FRAMES_KEY])
        temporal_ratio = int(getattr(batch.modules["vae"], "temporal_compression_ratio", 8))
        latent_num_frames = latent_frame_count(model_frames, temporal_ratio)
        latent_height = int(cond_latents.shape[-2])
        latent_width = int(cond_latents.shape[-1])

        model_dtype = prompt_embeds.dtype
        device = latents.device

        # One static shape per window for this model family, so a single compiled
        # graph covers every step; the signature is still recorded for reporting.
        transformer_for_forward = self.select_transformer_for_forward(
            transformer,
            batch=batch,
            local_shape=tuple(latents.shape),
            dynamic_cfg=False,
        )

        parallel_device = validate_cfg_parallel(server_args, batch)
        mesh_plan = resolve_mesh(server_args, batch)
        nccl = mesh_plan is not None and mesh_plan.get('backend') == 'nccl'
        use_mesh = nccl or (mesh_plan is not None and mesh_plan['sp'] * mesh_plan['cfg'] > 1)
        window_type = EraserDiTMeshWindow
        if nccl:
            from nndeploy.e_dit.algorithms.eraserdit.parallel.workers import DiTProcessWindow
            window_type = DiTProcessWindow
        cache_scope = (nullcontext(None) if use_mesh else EraserDiTCacheWindow(
            batch, total_steps=len(timesteps), num_blocks=len(transformer.transformer_blocks),
            enable_torch_compile=server_args.enable_torch_compile))
        with window_type(transformer, mesh_plan, pool=self._replica_pool,
                                batch=batch, total_steps=len(timesteps)) as mesh, \
                EraserDiTCFGWindow(transformer, parallel_device) as cfg_window, \
                cache_scope as cache_window, torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            # Coordinates are constant throughout one window and both CFG
            # branches. Keep this local so the next window/request cannot reuse
            # stale coordinates. Peer paths retain their device-local behavior.
            rotary_kwargs = {}
            if not mesh.active and parallel_device is None:
                from nndeploy.e_dit.algorithms.eraserdit.optimizations.block_compile import prepare_block_compile
                config = transformer.config
                tokens = (latent_num_frames // config.patch_size_t
                          * (latent_height // config.patch_size) * (latent_width // config.patch_size))
                prepare_block_compile(transformer, batch_size=latents.shape[0], sequence_length=tokens,
                                      device=device, dtype=model_dtype)
                rotary_kwargs["image_rotary_emb"] = transformer.rope(
                    latents, latent_num_frames, latent_height, latent_width,
                    rope_interpolation_scale,
                )
            progress = batch.extra.get('runtime_progress_state')
            for step_index, timestep in enumerate(timesteps):
                context = dict(
                    latents=latents,
                    model_dtype=model_dtype,
                    cond_latents=cond_latents,
                    device=device,
                    mask_values=mask_values,
                    timestep=timestep,
                    transformer_for_forward=transformer_for_forward,
                    latent_num_frames=latent_num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    rotary_kwargs=rotary_kwargs,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_attention_mask=negative_attention_mask,
                    cache_window=cache_window,
                    step_index=step_index,
                    mesh=mesh,
                    parallel_device=parallel_device,
                    cfg_window=cfg_window,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    nccl=nccl,
                    guidance_scale=guidance_scale,
                    scheduler=scheduler,
                    progress=progress,
                    timesteps=timesteps,
                )
                batch.extra['denoise_step'] = context
                yield batch
                latents = context['latents']
                noise_pred = context['noise_pred']
            batch.extra.pop('denoise_step', None)

            if mesh.active:
                batch.extra["dit_parallel"] = mesh.report()

        if parallel_device is not None:
            batch.extra["cfg_parallel"] = {
                "enabled": True, "positive_device": str(device),
                "negative_device": str(parallel_device), "steps": cfg_window.steps,
                "replica_setup_seconds": cfg_window.setup_seconds,
                "secondary_peak_allocated_gib": torch.cuda.max_memory_allocated(parallel_device) / 1024**3,
                "transformer_cache_mode": "off",
            }
        if batch.metrics is not None:
            batch.metrics.record_operation("denoise")
        self.record_compile_status(batch)
        # Report the backend the forwards actually resolved to, not the request:
        # `auto` picks one at runtime and the matrix must record which.
        backend_report = self_attention_backend_report(self._transformer)
        if backend_report is not None:
            batch.extra["attention_backend"] = backend_report
        batch.noise_pred = noise_pred
        batch.latents = latents
        self.log_info(
            "%s | steps=%d",
            field_summary("latents", latents),
            len(timesteps),
        )
        return batch


@register_type
class DiTPredictNode(StateNode):
    @operation
    def run(self, batch, server_args):
        context = batch.extra["denoise_step"]
        latents = context["latents"]
        model_dtype = context["model_dtype"]
        cond_latents = context["cond_latents"]
        device = context["device"]
        mask_values = context["mask_values"]
        timestep = context["timestep"]
        transformer_for_forward = context["transformer_for_forward"]
        latent_num_frames = context["latent_num_frames"]
        latent_height = context["latent_height"]
        latent_width = context["latent_width"]
        rope_interpolation_scale = context["rope_interpolation_scale"]
        rotary_kwargs = context["rotary_kwargs"]
        negative_prompt_embeds = context["negative_prompt_embeds"]
        negative_attention_mask = context["negative_attention_mask"]
        cache_window = context["cache_window"]
        step_index = context["step_index"]
        mesh = context["mesh"]
        parallel_device = context["parallel_device"]
        cfg_window = context["cfg_window"]
        prompt_embeds = context["prompt_embeds"]
        prompt_attention_mask = context["prompt_attention_mask"]
        nccl = context["nccl"]
        guidance_scale = context["guidance_scale"]
        service_checkpoint(batch, server_args, phase="denoise_step")
        step_start = time.perf_counter()
        latent_model_input = latents.to(model_dtype)
        cond_input = cond_latents.to(device=device)
        mask_input = mask_values.to(device=device)
        expanded_timestep = timestep.expand(1)

        model_kwargs = dict(
            hidden_states=latent_model_input,
            timestep=expanded_timestep,
            num_frames=latent_num_frames,
            height=latent_height,
            width=latent_width,
            rope_interpolation_scale=rope_interpolation_scale,
            attention_kwargs=None,
            return_dict=False,
            cond_latents=cond_input,
            mask_values=mask_input,
            **rotary_kwargs,
        )
        negative_kwargs = dict(
            model_kwargs,
            encoder_hidden_states=negative_prompt_embeds,
            encoder_attention_mask=negative_attention_mask,
            **(cache_window.kwargs("negative", step_index) if cache_window else {}),
        )
        if not mesh.active:
            if parallel_device is not None:
                negative_future = cfg_window.submit(**negative_kwargs)
            else:
                noise_pred_uncond = transformer_for_forward(**negative_kwargs)[0].float()

        positive_kwargs = dict(
            model_kwargs,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            **(cache_window.kwargs("positive", step_index) if cache_window else {}),
        )
        if nccl:
            noise_pred = mesh.predict_guided(negative_kwargs, positive_kwargs, guidance_scale)
        elif mesh.active:
            noise_pred_uncond, noise_pred_text = mesh.predict(negative_kwargs, positive_kwargs)
        else:
            noise_pred_text = transformer_for_forward(**positive_kwargs)[0].float()

        if parallel_device is not None:
            noise_pred_uncond = negative_future.result().to(device)

        if not nccl:
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        context["noise_pred"] = noise_pred
        context["step_start"] = step_start
        return batch


@register_type
class SchedulerStepNode(StateNode):
    @operation
    def run(self, batch, server_args):
        context = batch.extra["denoise_step"]
        timestep = context["timestep"]
        step_index = context["step_index"]
        scheduler = context["scheduler"]
        step_start = context["step_start"]
        progress = context["progress"]
        timesteps = context["timesteps"]
        noise_pred = context["noise_pred"]
        latents = scheduler.step(noise_pred, timestep, context["latents"], return_dict=False)[0]
        batch.step_index = step_index
        if batch.metrics is not None:
            batch.metrics.record_step(time.perf_counter() - step_start)
        if progress is not None:
            # Report launch progress without reading a CUDA scalar or
            # synchronizing the device just for UI updates.
            progress.update_denoise(step_index, len(timesteps),
                                    timestep_value=None)
        context["latents"] = latents
        return batch

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import math
import os
import time
import weakref
from contextlib import contextmanager, nullcontext
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from nndeploy.e_dit.algorithms.wan.videoedit_sampling import DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT, WanVideoEditSamplingParams
from nndeploy.e_dit.distributed.model_parallel import get_local_torch_device, get_sp_group
from nndeploy.e_dit.execution_context import set_forward_context
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode, operation
from nndeploy.e_dit.nodes.graph import register_type
from nndeploy.e_dit.algorithms.wan.nodes.support import VideoEditDenoiseSupport
from nndeploy.e_dit.algorithms.wan.nodes.support import VideoEditTextEncoder
from nndeploy.e_dit.nodes.validators import V, VerificationResult
from nndeploy.e_dit.platforms import current_platform
from nndeploy.e_dit.algorithms.wan.nodes.support import check_request_timeout
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.utils.logging_utils import init_logger
from nndeploy.e_dit.algorithms.wan.video.readonly_model_residency import StagePrefetch, close_encoder_prefetch, readonly_model_on_device
from nndeploy.e_dit.algorithms.wan.video.videoedit_memory import VideoEditResidencyPlanner, common_weight_budget, decoded_frames_to_images, env_nonnegative_int
from nndeploy.e_dit.algorithms.wan.video.distributed import broadcast_tensor_from_owner
from nndeploy.e_dit.algorithms.wan.video.preprocess import prepare_window_inputs
from nndeploy.e_dit.algorithms.wan.video.progress import build_window_progress_payload, write_videoedit_progress
from nndeploy.e_dit.algorithms.wan.video.vae_spatial import distributed_tiled_decode, distributed_tiled_encode
from nndeploy.e_dit.utils.precision import PRECISION_TO_TYPE
logger = init_logger(__name__)

def _conditioning_group():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return get_sp_group()

def _device_total_bytes() -> int:
    """Optimistic pre-flight headroom: the whole device, before any measurement.

    The window validation stage only proves that a shape cannot possibly fit;
    the denoise stage re-plans against real headroom. Zero on non-CUDA hosts
    keeps the geometry checks running without a memory verdict.
    """
    if current_platform.device_type != 'cuda':
        return 0
    return int(torch.cuda.mem_get_info(get_local_torch_device())[1])

def _stage_owner(name: str, default: int, world_size: int) -> int:
    env_name = f'NNDEPLOY_DIT_WAN_VIDEOEDIT_{name.upper()}_OWNER'
    raw = os.environ.get(env_name, str(default))
    try:
        owner = int(raw)
    except ValueError as exc:
        raise ValueError(f'{env_name} must be an integer, got {raw!r}') from exc
    if not 0 <= owner < world_size:
        raise ValueError(f'{env_name} must be in [0, {world_size}), got {owner}')
    return owner

def _collect_parallel_error(group, local_error: str | None, operation: str) -> None:
    if group is None or group.world_size == 1:
        if local_error is not None:
            raise RuntimeError(f'{operation} failed: {local_error}')
        return
    errors: list[str | None] = [None] * group.world_size
    torch.distributed.all_gather_object(errors, local_error, group=group.cpu_group)
    failures = [f'rank {rank}: {error}' for (rank, error) in enumerate(errors) if error]
    if failures:
        raise RuntimeError(f'{operation} failed; ' + '; '.join(failures))

def _videoedit_params(batch: Req) -> WanVideoEditSamplingParams:
    params = batch.sampling_params
    if not isinstance(params, WanVideoEditSamplingParams):
        raise TypeError(f'VideoEdit stages require WanVideoEditSamplingParams, got {type(params).__name__}')
    return params

def _module_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return default

def _retrieve_latents(encoder_output: Any) -> torch.Tensor:
    if hasattr(encoder_output, 'latent_dist'):
        encoder_output = encoder_output.latent_dist
    if hasattr(encoder_output, 'mode'):
        return encoder_output.mode()
    if hasattr(encoder_output, 'latents'):
        return encoder_output.latents
    if isinstance(encoder_output, torch.Tensor):
        return encoder_output
    raise AttributeError('Could not access latents of VAE encoder output')

def _vae_latent_mean_std(vae: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    vae_config = getattr(vae, 'config', None)
    z_dim = int(getattr(vae_config, 'z_dim', None) or getattr(vae, 'z_dim', 16))
    latents_mean = getattr(vae_config, 'latents_mean', None)
    if latents_mean is None:
        latents_mean = getattr(vae, 'latents_mean')
    latents_std = getattr(vae_config, 'latents_std', None)
    if latents_std is None:
        latents_std = getattr(vae, 'latents_std')
    mean = torch.tensor(latents_mean, device=device, dtype=dtype).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(latents_std, device=device, dtype=dtype).view(1, z_dim, 1, 1, 1)
    return (mean, std)

def _normalize_vae_latents(latents: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    (mean, std) = _vae_latent_mean_std(vae, latents.device, latents.dtype)
    return (latents - mean) / std

def _denormalize_vae_latents(latents: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    (mean, std) = _vae_latent_mean_std(vae, latents.device, latents.dtype)
    return latents * std + mean

def _ensure_tensor_decode_output(decode_output: Any) -> torch.Tensor:
    if isinstance(decode_output, tuple):
        return decode_output[0]
    if hasattr(decode_output, 'sample'):
        return decode_output.sample
    return decode_output

def _move_vae_for_videoedit(vae: torch.nn.Module, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    if getattr(vae, '_sglang_videoedit_native_vae', False):
        return vae.to(device=device)
    return vae.to(device=device, dtype=dtype)

def _keep_stage_tensors():
    return os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_KEEP_STAGE_TENSORS') == '1'

@contextmanager
def _videoedit_vae_residency(vae, device, dtype, server_args):
    """Stage-scoped VAE weights, with causal feature caches cleared on failure."""
    offload = getattr(server_args, 'vae_cpu_offload', False)
    if offload:
        if any((t.device.type != 'cpu' for t in (*vae.parameters(), *vae.buffers()))):
            vae.to('cpu')
        target_dtype = None if getattr(vae, '_sglang_videoedit_native_vae', False) else dtype
        residency = readonly_model_on_device(vae, device, dtype=target_dtype)
    else:
        residency = nullcontext()
    with residency:
        try:
            yield vae
        finally:
            clear_cache = getattr(vae, 'clear_cache', None)
            if callable(clear_cache):
                clear_cache()

def _videoedit_vae_sp_enabled(server_args: ServerArgs, operation: str) -> bool:
    if not getattr(server_args.pipeline_config, 'vae_sp', False):
        return False
    mode = os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_VAE_SP_MODE', 'both').lower()
    if mode not in {'encode', 'decode', 'both'}:
        raise ValueError(f'NNDEPLOY_DIT_WAN_VIDEOEDIT_VAE_SP_MODE must be encode, decode, or both, got {mode!r}')
    group = _conditioning_group()
    return group is not None and group.world_size > 1 and (mode in {operation, 'both'})

def _uses_spatial_tiles(vae, tensor: torch.Tensor, *, latent: bool) -> bool:
    (height, width) = tensor.shape[-2:]
    if latent:
        ratio = vae.spatial_compression_ratio
        height *= ratio
        width *= ratio
    return bool(getattr(vae, 'use_tiling', False) and (width > vae.tile_sample_min_width or height > vae.tile_sample_min_height))

def _decode_vae_for_videoedit(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    try:
        decoded = vae.decode(latents, return_dict=False)
    except TypeError:
        decoded = vae.decode(latents)
    return _ensure_tensor_decode_output(decoded)

def _ensure_tensor_transformer_output(transformer_output: Any) -> torch.Tensor:
    if isinstance(transformer_output, torch.Tensor):
        return transformer_output
    if isinstance(transformer_output, tuple):
        return transformer_output[0]
    if hasattr(transformer_output, 'sample'):
        return transformer_output.sample
    raise TypeError(f'VideoEdit transformer output must be a tensor, tuple, or object with .sample; got {type(transformer_output).__name__}')

def _transformer_cache_context(transformer: torch.nn.Module, name: str):
    if hasattr(transformer, 'cache_context'):
        return transformer.cache_context(name)
    return nullcontext()

def calc_current_cfg(max_cfg: float, current_step: int, max_step: int=15, min_cfg: float=1.0, dynamic_cfg: bool=True) -> tuple[float, bool]:
    if dynamic_cfg:
        if current_step < max_step:
            add_cfg = max((max_cfg - min_cfg) / math.pow(max_cfg - min_cfg, 1.0 / max_step) ** current_step, 0)
        else:
            add_cfg = 0.0
        current_cfg = min_cfg + add_cfg
        return (current_cfg, current_cfg > 1.0)
    return (max_cfg, max_cfg != 1.0)

def _prepare_clip_pixel_values(image, *, clip_preprocess: str, image_processor: Any | None, device: torch.device) -> torch.Tensor:
    if clip_preprocess == 'diffuser':
        if image_processor is None:
            raise ValueError("VideoEdit clip_preprocess='diffuser' requires the checkpoint image_processor component")
        processed = image_processor(images=image, return_tensors='pt')
        pixel_values = processed['pixel_values'] if isinstance(processed, dict) else processed.pixel_values
        return pixel_values.to(device=device, dtype=torch.float32)
    if clip_preprocess != 'diffsynth':
        raise ValueError(f'Unsupported VideoEdit clip_preprocess={clip_preprocess!r}')
    pixel_values = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    pixel_values = pixel_values.mul(2.0).sub(1.0).to(device=device, dtype=torch.float32)
    pixel_values = F.interpolate(pixel_values, size=(224, 224), mode='bicubic', align_corners=False)
    pixel_values = pixel_values.mul(0.5).add(0.5)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return (pixel_values - mean) / std

@register_type
class VideoEditWindowValidationNode(StateNode):

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        frames = params.runtime_window_frames
        masks = params.runtime_window_masks
        if frames is None or masks is None:
            raise ValueError('VideoEdit window frames and masks must be materialized')
        if len(frames) != params.infer_len:
            raise ValueError(f'VideoEdit window must contain {params.infer_len} frames, got {len(frames)}')
        if len(masks) != params.infer_len:
            raise ValueError(f'VideoEdit window must contain {params.infer_len} masks, got {len(masks)}')
        if params.infer_len < 1 or (params.infer_len - 1) % 4 != 0:
            raise ValueError('VideoEdit stages require infer_len >= 1 and (infer_len-1)%4=0')
        (width, height) = frames[0].size
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f'VideoEdit window size must be divisible by 16, got {width}x{height}')
        if os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_AUTO_RESIDENCY') == '1':
            if not getattr(server_args, 'dit_layerwise_offload', False):
                raise ValueError('AUTO_RESIDENCY requires dit_layerwise_offload=true')
            VideoEditResidencyPlanner().plan((1, 16, (len(frames) - 1) // 4 + 1, height // 8, width // 8), _device_total_bytes(), sp_degree=int(server_args.sp_degree))
        params.runtime_height = height
        params.runtime_width = width
        params.runtime_num_frames = len(frames)
        params.runtime_window_validated = True
        batch.height = height
        batch.width = width
        batch.num_frames = len(frames)
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditTextEncodingNode(StateNode):

    def setup(self, text_encoder: torch.nn.Module, tokenizer: Any, transformer: torch.nn.Module, next_image_encoder=None):
        self.text_encoder = text_encoder
        self.text_stage = VideoEditTextEncoder([text_encoder], [tokenizer])
        self.transformer = transformer
        self.next_image_encoder = next_image_encoder
        self.cached_negative_text: str | None = None
        self.cached_negative_embeds: torch.Tensor | None = None
        self.positive_encode_calls = 0
        self.negative_encode_calls = 0

    @torch.no_grad()
    def encode_texts_to_cpu(self, texts: list[str], server_args: ServerArgs) -> list[torch.Tensor]:
        """Encode several texts during one model residency interval."""
        if not texts:
            return []
        device = get_local_torch_device()
        target_dtype = _module_dtype(self.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision])
        autocast_enabled = target_dtype != torch.float32 and (not server_args.disable_autocast)
        preserve_cpu = server_args.text_encoder_cpu_offload and os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_ENCODER_CPU_MASTER') == '1'
        residency = readonly_model_on_device(self.text_encoder, device) if preserve_cpu else nullcontext()
        results = []
        try:
            with residency:
                if not preserve_cpu and hasattr(self.text_encoder, 'to'):
                    self.text_encoder = self.text_encoder.to(device)
                    self.text_stage.text_encoders[0] = self.text_encoder
                for text in texts:
                    with torch.autocast(device_type=current_platform.device_type, dtype=target_dtype, enabled=autocast_enabled):
                        (embeds, _, _) = self.text_stage.encode_text(text, server_args, encoder_index=[0], return_attention_mask=True, dtype=target_dtype)
                    results.append(embeds[0].detach().to('cpu'))
        finally:
            if not preserve_cpu and server_args.text_encoder_cpu_offload and hasattr(self.text_encoder, 'to'):
                self.text_encoder = self.text_encoder.to('cpu')
                self.text_stage.text_encoders[0] = self.text_encoder
        return results

    @torch.no_grad()
    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        device = get_local_torch_device()
        target_dtype = _module_dtype(self.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision])
        autocast_enabled = target_dtype != torch.float32 and (not server_args.disable_autocast)
        preserve_cpu = server_args.text_encoder_cpu_offload and os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_ENCODER_CPU_MASTER') == '1'
        residency = readonly_model_on_device(self.text_encoder, device) if preserve_cpu else nullcontext()
        try:
            with residency:
                if not preserve_cpu and hasattr(self.text_encoder, 'to'):
                    self.text_encoder = self.text_encoder.to(device)
                    self.text_stage.text_encoders[0] = self.text_encoder
                next_encoder = getattr(self, 'next_image_encoder', None)
                if next_encoder is not None and device.type == 'cuda' and getattr(params, 'use_clip', False) and getattr(server_args, 'image_encoder_cpu_offload', False) and (os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_ENCODER_CPU_MASTER') == '1') and (os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_PREFETCH_IMAGE_ENCODER') == '1'):
                    tensors = list(next_encoder.parameters()) + list(next_encoder.buffers())
                    required = sum((t.numel() * t.element_size() for t in tensors))
                    reserve = env_nonnegative_int('NNDEPLOY_DIT_WAN_VIDEOEDIT_STAGE_PREFETCH_RESERVE_MB', 4096) * 1024 ** 2
                    (free, _) = torch.cuda.mem_get_info(device)
                    if all((t.device.type == 'cpu' for t in tensors)) and required + reserve <= free:
                        params.runtime_encoder_prefetch = StagePrefetch(next_encoder, device)
                with torch.autocast(device_type=current_platform.device_type, dtype=target_dtype, enabled=autocast_enabled):
                    (prompt_embeds, _, _) = self.text_stage.encode_text(params.prompt or ' ', server_args, encoder_index=[0], return_attention_mask=True, dtype=target_dtype)
                params.runtime_prompt_embeds = prompt_embeds[0]
                params.runtime_do_cfg = float(params.guidance_scale) > 1.0
                if params.runtime_do_cfg:
                    with torch.autocast(device_type=current_platform.device_type, dtype=target_dtype, enabled=autocast_enabled):
                        (neg_embeds, _, _) = self.text_stage.encode_text(params.negative_prompt or '', server_args, encoder_index=[0], return_attention_mask=True, dtype=target_dtype)
                    params.runtime_negative_prompt_embeds = neg_embeds[0]
                else:
                    params.runtime_negative_prompt_embeds = None
        except BaseException:
            close_encoder_prefetch(params)
            raise
        finally:
            if not preserve_cpu and server_args.text_encoder_cpu_offload and hasattr(self.text_encoder, 'to'):
                self.text_encoder = self.text_encoder.to('cpu')
                self.text_stage.text_encoders[0] = self.text_encoder
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditImageEncodingNode(StateNode):

    def setup(self, transformer: torch.nn.Module, image_encoder: torch.nn.Module | None = None, image_processor: Any | None = None):
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.transformer = transformer

    @torch.no_grad()
    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        params.runtime_image_embeds = None
        if not params.use_clip:
            return batch
        if self.image_encoder is None:
            raise ValueError('VideoEdit use_clip=True requires an image_encoder component. Provide --image-encoder-path/--component_paths.image_encoder or set use_clip=False.')
        if params.runtime_window_frames is None or not params.runtime_window_frames:
            raise ValueError('VideoEdit window frames must be materialized before image encoding')
        if params.runtime_height is None or params.runtime_width is None:
            raise ValueError('VideoEdit window must be validated before image encoding')
        device = get_local_torch_device()
        preserve_cpu = server_args.image_encoder_cpu_offload and os.environ.get('NNDEPLOY_DIT_WAN_VIDEOEDIT_ENCODER_CPU_MASTER') == '1'
        residency = getattr(params, 'runtime_encoder_prefetch', None)
        params.runtime_encoder_prefetch = None
        if residency is None:
            residency = readonly_model_on_device(self.image_encoder, device) if preserve_cpu else nullcontext()
        try:
            with residency:
                if not preserve_cpu and hasattr(self.image_encoder, 'to'):
                    self.image_encoder = self.image_encoder.to(device)
                image = params.runtime_window_frames[0].convert('RGB')
                image = image.resize((params.runtime_width, params.runtime_height))
                pixel_values = _prepare_clip_pixel_values(image, clip_preprocess=params.clip_preprocess, image_processor=self.image_processor, device=device)
                image_dtype = _module_dtype(self.image_encoder, torch.float32)
                target_dtype = _module_dtype(self.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision])
                autocast_enabled = target_dtype != torch.float32 and (not server_args.disable_autocast)
                with torch.autocast(device_type=current_platform.device_type, dtype=target_dtype, enabled=autocast_enabled):
                    outputs = self.image_encoder(pixel_values=pixel_values.to(dtype=image_dtype), **server_args.pipeline_config.image_encoder_extra_args)
                    image_embeds = server_args.pipeline_config.postprocess_image(outputs)
                params.runtime_image_embeds = image_embeds.to(device=device, dtype=target_dtype)
        finally:
            if not preserve_cpu and server_args.image_encoder_cpu_offload and hasattr(self.image_encoder, 'to'):
                self.image_encoder = self.image_encoder.to('cpu')
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditConditioningEncodingNode(StateNode):
    """Run request text and per-window CLIP on their owners, then distribute."""

    def setup(self, text_stage: VideoEditVideoEditTextEncoder, image_stage: VideoEditImageEncodingStage):
        self.text_stage = text_stage
        self.image_stage = image_stage
        self.startup_negative_seconds: float | None = None
        self.last_text_seconds: float | None = None
        self.last_clip_seconds: float | None = None

    @staticmethod
    def _sync_for_timing() -> None:
        device = get_local_torch_device()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def precompute_default_negative(self, server_args: ServerArgs) -> None:
        group = _conditioning_group()
        world_size = group.world_size if group is not None else 1
        rank = group.rank_in_group if group is not None else 0
        text_owner = _stage_owner('text', 0, world_size)
        value = None
        local_error = None
        if rank == text_owner:
            try:
                self._sync_for_timing()
                started = time.perf_counter()
                value = self.text_stage.encode_texts_to_cpu([DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT], server_args)[0]
                self._sync_for_timing()
                self.startup_negative_seconds = time.perf_counter() - started
                self.text_stage.negative_encode_calls += 1
            except BaseException as exc:
                local_error = f'{type(exc).__name__}: {exc}'
        _collect_parallel_error(group, local_error, 'default VideoEdit negative prompt precomputation')
        value = broadcast_tensor_from_owner(group, value, text_owner, 'negative_prompt_embeds')
        self.text_stage.cached_negative_text = DEFAULT_VIDEOEDIT_NEGATIVE_PROMPT
        self.text_stage.cached_negative_embeds = value.detach().to('cpu')
        logger.info('Precomputed the default VideoEdit negative prompt on text owner rank %s in %s seconds', text_owner, self.startup_negative_seconds)

    @torch.no_grad()
    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        group = _conditioning_group()
        world_size = group.world_size if group is not None else 1
        rank = group.rank_in_group if group is not None else 0
        text_owner = _stage_owner('text', 0, world_size)
        clip_owner = _stage_owner('clip', 1 if world_size > 1 else 0, world_size)
        device = get_local_torch_device()
        target_dtype = _module_dtype(self.text_stage.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision])
        params.runtime_do_cfg = float(params.guidance_scale) > 1.0
        prompt_needed = params.runtime_request_prompt_embeds is None
        negative_text = params.negative_prompt or ''
        negative_update = bool(params.runtime_do_cfg and (self.text_stage.cached_negative_embeds is None or self.text_stage.cached_negative_text != negative_text))
        prompt_value = None
        negative_value = None
        image_value = None
        local_error = None
        self.last_text_seconds = None
        self.last_clip_seconds = None
        try:
            if rank == text_owner and (prompt_needed or negative_update):
                texts = []
                if prompt_needed:
                    texts.append(params.prompt or ' ')
                if negative_update:
                    texts.append(negative_text)
                self._sync_for_timing()
                started = time.perf_counter()
                encoded = self.text_stage.encode_texts_to_cpu(texts, server_args)
                self._sync_for_timing()
                self.last_text_seconds = time.perf_counter() - started
                cursor = 0
                if prompt_needed:
                    prompt_value = encoded[cursor]
                    self.text_stage.positive_encode_calls += 1
                    cursor += 1
                if negative_update:
                    negative_value = encoded[cursor]
                    self.text_stage.negative_encode_calls += 1
            if params.use_clip and rank == clip_owner:
                self._sync_for_timing()
                started = time.perf_counter()
                self.image_stage.compute(batch, server_args)
                self._sync_for_timing()
                self.last_clip_seconds = time.perf_counter() - started
                image_value = params.runtime_image_embeds
        except BaseException as exc:
            local_error = f'{type(exc).__name__}: {exc}'
            close_encoder_prefetch(params)
        _collect_parallel_error(group, local_error, 'VideoEdit text/CLIP preparation')
        if prompt_needed:
            prompt_value = broadcast_tensor_from_owner(group, prompt_value, text_owner, 'prompt_embeds')
            params.runtime_request_prompt_embeds = prompt_value.detach().to('cpu')
        if params.runtime_request_prompt_embeds is None:
            raise RuntimeError('VideoEdit request prompt embedding was not prepared')
        if params.runtime_do_cfg:
            if negative_update:
                negative_value = broadcast_tensor_from_owner(group, negative_value, text_owner, 'negative_prompt_embeds')
                self.text_stage.cached_negative_text = negative_text
                self.text_stage.cached_negative_embeds = negative_value.detach().to('cpu')
            if self.text_stage.cached_negative_embeds is None:
                raise RuntimeError('VideoEdit negative prompt cache is empty')
            params.runtime_request_negative_prompt_embeds = self.text_stage.cached_negative_embeds
        else:
            params.runtime_request_negative_prompt_embeds = None
        params.runtime_prompt_embeds = params.runtime_request_prompt_embeds.to(device=device, dtype=target_dtype)
        params.runtime_negative_prompt_embeds = params.runtime_request_negative_prompt_embeds.to(device=device, dtype=target_dtype) if params.runtime_request_negative_prompt_embeds is not None else None
        if params.use_clip:
            params.runtime_image_embeds = broadcast_tensor_from_owner(group, image_value, clip_owner, 'image_embeds').to(device=device, dtype=target_dtype)
        else:
            params.runtime_image_embeds = None
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditConditionEncodingNode(StateNode):

    def setup(self, vae: torch.nn.Module):
        self.vae = vae

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor: torch.Tensor, *, vae_dtype: torch.dtype, output_dtype: torch.dtype, autocast_enabled: bool, use_spatial_parallel: bool=False) -> torch.Tensor:
        device = get_local_torch_device()
        self.vae = _move_vae_for_videoedit(self.vae, device=device, dtype=vae_dtype)
        video_tensor = video_tensor.to(device=device, dtype=vae_dtype)
        with torch.autocast(device_type=current_platform.device_type, dtype=output_dtype, enabled=autocast_enabled):
            if use_spatial_parallel and _uses_spatial_tiles(self.vae, video_tensor, latent=False):
                moments = distributed_tiled_encode(self.vae, video_tensor, _conditioning_group())
                if moments.shape[1] % 2:
                    raise ValueError('VAE encoder moments must contain mean and log-variance channels')
                latents = moments.chunk(2, dim=1)[0]
            else:
                encoded = self.vae.encode(video_tensor)
                latents = _retrieve_latents(encoded)
            latents = _normalize_vae_latents(latents, self.vae)
        return latents.to(device=device, dtype=output_dtype)

    @torch.no_grad()
    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        if not params.runtime_window_validated:
            raise ValueError('VideoEdit window must be validated before condition encoding')
        device = get_local_torch_device()
        tensor_dtype = PRECISION_TO_TYPE[params.dtype]
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        prepared = prepare_window_inputs(params.runtime_window_frames, params.runtime_window_masks, device=device, dtype=tensor_dtype, mask_downsample_mode=params.mask_downsample_mode, preserve_first_frame=True, return_intermediates=_keep_stage_tensors())
        params.runtime_masked_video_tensor = prepared['masked_video_tensor'] if _keep_stage_tensors() else None
        params.runtime_mask_video_tensor = prepared['mask_video_tensor']
        params.runtime_cond_masks = prepared['cond_masks']
        if server_args.pipeline_config.vae_tiling and hasattr(self.vae, 'enable_tiling'):
            self.vae.enable_tiling()
        elif hasattr(self.vae, 'disable_tiling'):
            self.vae.disable_tiling()
        autocast_enabled = tensor_dtype != torch.float32 and (not server_args.disable_autocast)
        masked_video = prepared['masked_video_tensor'].permute(1, 0, 2, 3).unsqueeze(0)
        try:
            with _videoedit_vae_residency(self.vae, device, vae_dtype, server_args):
                params.runtime_cond_latents = self._encode_video_latents(masked_video, vae_dtype=vae_dtype, output_dtype=tensor_dtype, autocast_enabled=autocast_enabled, use_spatial_parallel=_videoedit_vae_sp_enabled(server_args, 'encode'))
            params.runtime_condition_latent = torch.cat([params.runtime_cond_masks, params.runtime_cond_latents], dim=1) if _keep_stage_tensors() else None
        except BaseException:
            if not _keep_stage_tensors():
                params.runtime_cond_latents = None
                params.runtime_cond_masks = None
            raise
        finally:
            if not _keep_stage_tensors():
                params.runtime_masked_video_tensor = None
                params.runtime_mask_video_tensor = None
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditLatentPreparationNode(StateNode):

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _videoedit_params(batch)
        shape_source = params.runtime_cond_latents
        if shape_source is None:
            raise ValueError('VideoEdit latent shape source must be prepared before noise')
        device = shape_source.device
        generator = torch.Generator(device='cpu').manual_seed(int(params.seed))
        params.runtime_generator = generator
        params.runtime_noise = torch.randn(shape_source.shape, generator=generator, device='cpu', dtype=torch.float32).to(device=device)
        params.runtime_latents = params.runtime_noise
        batch.generator = generator
        batch.latents = params.runtime_latents
        batch.raw_latent_shape = tuple(params.runtime_latents.shape)
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditTimestepPreparationNode(StateNode):

    def setup(self, scheduler: Any):
        self.scheduler = scheduler

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        device = get_local_torch_device()
        from copy import deepcopy
        self.scheduler = deepcopy(self.resources.modules["scheduler"])
        batch.extra["scheduler"] = self.scheduler
        self.scheduler.set_timesteps(params.num_inference_steps, shift=server_args.pipeline_config.flow_shift or 5.0, device=device)
        timesteps = self.scheduler.timesteps
        effective_steps = params.num_inference_steps
        params.runtime_timesteps = timesteps.to(device)
        params.runtime_effective_num_inference_steps = int(effective_steps)
        params.runtime_num_warmup_steps = len(timesteps) - int(effective_steps)
        batch.timesteps = params.runtime_timesteps
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditLatentInitNode(StateNode):

    def setup(self, scheduler: Any):
        self.scheduler = scheduler

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _videoedit_params(batch)
        if params.runtime_timesteps is None:
            raise ValueError('VideoEdit timesteps must be prepared before latent init')
        params.runtime_initial_timestep = params.runtime_timesteps[:1]
        params.runtime_latents = params.runtime_noise
        batch.latents = params.runtime_latents
        batch.raw_latent_shape = tuple(params.runtime_latents.shape)
        if not _keep_stage_tensors():
            params.runtime_noise = None
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditDenoisingNode(VideoEditDenoiseSupport):

    def setup(self, transformer: torch.nn.Module, scheduler: Any, pipeline=None):
        self.transformer, self.scheduler = transformer, scheduler
        self.pipeline = weakref.ref(pipeline) if pipeline else None

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        params = _videoedit_params(batch)
        result = VerificationResult()
        result.add_check('runtime_timesteps', params.runtime_timesteps, V.is_tensor)
        result.add_check('runtime_effective_num_inference_steps', params.runtime_effective_num_inference_steps, V.positive_int)
        result.add_check('runtime_latents', params.runtime_latents, [V.is_tensor, V.with_dims(5)])
        result.add_check('runtime_cond_masks', params.runtime_cond_masks, [V.is_tensor, V.with_dims(5)])
        result.add_check('runtime_cond_latents', params.runtime_cond_latents, [V.is_tensor, V.with_dims(5)])
        result.add_check('runtime_prompt_embeds', params.runtime_prompt_embeds, V.is_tensor)
        result.add_check('runtime_image_embeds', params.runtime_image_embeds, lambda value: not params.use_clip or V.is_tensor(value))
        result.add_check('runtime_generator', params.runtime_generator, V.generator_or_list_generators)
        result.add_check('runtime_do_cfg', params.runtime_do_cfg, V.bool_value)
        result.add_check('runtime_negative_prompt_embeds', params.runtime_negative_prompt_embeds, lambda value: not params.runtime_do_cfg or V.is_tensor(value))
        return result

    @torch.no_grad()
    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        self.scheduler = batch.extra["scheduler"]
        self._residency_observation = None
        completed = False
        try:
            result = (yield from self._forward_denoising(batch, server_args))
            completed = True
            return result
        finally:
            if completed and self._residency_observation is not None:
                (key, initial_bytes, capacity) = self._residency_observation
                peak = torch.cuda.max_memory_allocated()
                self.residency_planner.observe(key, peak_allocated_bytes=peak, initial_allocated_bytes=initial_bytes, weight_capacity_bytes=capacity)
                self.residency_plan['observed_peak_allocated_bytes'] = peak
            for manager in getattr(self.transformer, 'layerwise_offload_managers', []):
                if getattr(manager, 'release_after_denoise', False) or getattr(manager, 'auto_residency', False):
                    manager.release_all()
            if batch is not None and (not _keep_stage_tensors()):
                params = _videoedit_params(batch)
                for name in ('runtime_cond_masks', 'runtime_cond_latents', 'runtime_condition_latent', 'runtime_noise', 'runtime_prompt_embeds', 'runtime_negative_prompt_embeds', 'runtime_image_embeds'):
                    setattr(params, name, None)

    def _forward_denoising(self, batch: Req, server_args: ServerArgs) -> Req:
        if server_args.cfg_parallel_degree > 1:
            raise NotImplementedError('VideoEdit MVP does not support CFG parallel yet')
        params = _videoedit_params(batch)
        if params.runtime_latents is None or params.runtime_timesteps is None:
            raise ValueError('VideoEdit latents and timesteps must be prepared')
        if params.runtime_prompt_embeds is None:
            raise ValueError('VideoEdit prompt embeds must be prepared')
        target_dtype = _module_dtype(self.transformer, PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision])
        autocast_enabled = target_dtype != torch.float32 and (not server_args.disable_autocast)
        timesteps = params.runtime_timesteps
        timesteps_cpu = timesteps.detach().cpu()
        latents = params.runtime_latents
        image_embeds = params.runtime_image_embeds
        self._manage_device_placement(self.transformer, None, server_args)
        managers = [m for m in getattr(self.transformer, 'layerwise_offload_managers', []) if getattr(m, 'auto_residency', False) and m.enabled]
        if managers:
            if target_dtype != torch.bfloat16 or getattr(self.transformer, 'hidden_size', None) != 5120 or getattr(server_args, 'ring_degree', 1) != 1 or (str(getattr(server_args, 'attention_backend', '')).lower() not in ('fa', 'attentionbackendenum.fa')):
                raise ValueError('AUTO_RESIDENCY currently requires Wan 14B BF16, FA and pure Ulysses')
            for manager in managers:
                manager.release_all()
            torch.cuda.current_stream().synchronize()
            (free, _) = torch.cuda.mem_get_info()
            initial_bytes = torch.cuda.memory_allocated()
            reusable = torch.cuda.memory_reserved() - initial_bytes
            if not hasattr(self, 'residency_planner'):
                self.residency_planner = VideoEditResidencyPlanner()
            largest_layer = max((size for manager in managers for size in manager.layer_bytes()), default=0)
            (key, plan) = self.residency_planner.plan(tuple(latents.shape), free + reusable, sp_degree=int(server_args.sp_degree), cfg=params.runtime_do_cfg, teacache=bool(batch.enable_teacache), minimum_reserve_bytes=env_nonnegative_int('NNDEPLOY_DIT_WAN_VIDEOEDIT_ACTIVATION_RESERVE_MB', 4096) * 1024 ** 2, full_frame_reserve_bytes=env_nonnegative_int('NNDEPLOY_DIT_WAN_VIDEOEDIT_FULL_FRAME_RESERVE_MB', 16384) * 1024 ** 2, margin_bytes=env_nonnegative_int('NNDEPLOY_DIT_WAN_VIDEOEDIT_RESIDENCY_MARGIN_MB', 2048) * 1024 ** 2, weight_limit_bytes=env_nonnegative_int('NNDEPLOY_DIT_WAN_VIDEOEDIT_WEIGHT_BUDGET_MB') * 1024 ** 2, minimum_weight_bytes=largest_layer)
            cpu_group = None
            if torch.distributed.is_initialized():
                from nndeploy.e_dit.distributed.model_parallel import get_sp_group
                cpu_group = get_sp_group().cpu_group
            budget = common_weight_budget(plan['local_weight_budget_bytes'], cpu_group)
            plan['common_weight_budget_bytes'] = budget
            total = sum((m.cpu_weight_bytes for m in managers))
            for manager in managers:
                share = budget * manager.cpu_weight_bytes // max(1, total)
                manager.configure_memory_budget(share)
            capacity = sum((m.weight_capacity_bytes for m in managers))
            plan['weight_capacity_bytes'] = capacity
            plan['resident_layers'] = [m.resident_layers for m in managers]
            plan['prefetch_layers'] = [m.prefetch_size for m in managers]
            self.residency_plan = plan
            self._residency_observation = (key, initial_bytes, capacity)
            torch.cuda.reset_peak_memory_stats()
            logger.info('VideoEdit adaptive residency: %s', plan)
        batch.do_classifier_free_guidance = bool(params.runtime_do_cfg)
        batch.is_cfg_negative = False
        self._maybe_enable_cache_dit(params.runtime_effective_num_inference_steps, batch)
        with torch.autocast(device_type=current_platform.device_type, dtype=target_dtype, enabled=autocast_enabled):
            with self.progress_bar(total=params.runtime_effective_num_inference_steps) as progress_bar:
                for (i, t_host) in enumerate(timesteps_cpu):
                    yield
                    check_request_timeout(batch)
                    t_device = timesteps[i]
                    (current_cfg, do_cfg) = calc_current_cfg(max_cfg=float(params.guidance_scale), current_step=i, max_step=int(params.dynamic_cfg_max_step), min_cfg=float(params.dynamic_cfg_min), dynamic_cfg=bool(params.dynamic_cfg))
                    params.runtime_current_step = i
                    params.runtime_current_timestep = t_device
                    params.runtime_current_cfg = float(current_cfg)
                    latent_model_input = torch.cat([latents, params.runtime_cond_masks, params.runtime_cond_latents], dim=1).to(target_dtype)
                    scheduler_timestep = t_device.to(dtype=target_dtype)
                    timestep = scheduler_timestep.expand(latents.shape[0])
                    attn_metadata = self._build_attn_metadata(i, batch, server_args, timestep_value=int(t_host.item()), timesteps=timesteps_cpu)
                    batch.is_cfg_negative = False
                    with set_forward_context(current_timestep=i, attn_metadata=attn_metadata, forward_batch=batch):
                        with _transformer_cache_context(self.transformer, 'cond'):
                            noise_pred = _ensure_tensor_transformer_output(self.transformer(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=params.runtime_prompt_embeds, encoder_hidden_states_image=image_embeds, attention_kwargs=None, return_dict=False))
                    check_request_timeout(batch)
                    if do_cfg:
                        if params.runtime_negative_prompt_embeds is None:
                            raise ValueError('Negative prompt embeds are required for CFG')
                        batch.is_cfg_negative = True
                        with set_forward_context(current_timestep=i, attn_metadata=attn_metadata, forward_batch=batch):
                            with _transformer_cache_context(self.transformer, 'uncond'):
                                noise_uncond = _ensure_tensor_transformer_output(self.transformer(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=params.runtime_negative_prompt_embeds, encoder_hidden_states_image=image_embeds, attention_kwargs=None, return_dict=False))
                        check_request_timeout(batch)
                        noise_pred = noise_uncond + current_cfg * (noise_pred - noise_uncond)
                        batch.is_cfg_negative = False
                    latents = self.scheduler.step(noise_pred, scheduler_timestep, latents)
                    params.runtime_latents = latents
                    batch.latents = latents
                    params.runtime_progress = float(i + 1) / float(len(timesteps))
                    write_videoedit_progress(params.progress_path, build_window_progress_payload(stage='denoising', total_frames=params.runtime_num_input_frames, infer_len=params.infer_len, overlap=params.overlap, total_windows=len(params.runtime_window_specs or [None]), current_window_index=params.runtime_progress_window_index if params.runtime_progress_window_index is not None else params.runtime_window_index, current_step_index=i, steps_per_window=len(timesteps)))
                    if progress_bar is not None:
                        progress_bar.update()
        return batch

@register_type
class VideoEditDecodingNode(StateNode):

    def setup(self, vae: torch.nn.Module):
        self.vae = vae

    @torch.no_grad()
    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        params = _videoedit_params(batch)
        if params.runtime_latents is None:
            raise ValueError('VideoEdit denoised latents must be prepared')
        device = get_local_torch_device()
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        try:
            with _videoedit_vae_residency(self.vae, device, vae_dtype, server_args):
                self.vae = _move_vae_for_videoedit(self.vae, device=device, dtype=vae_dtype)
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
                latents = params.runtime_latents.to(device=device, dtype=vae_dtype)
                autocast_enabled = vae_dtype != torch.float32 and (not server_args.disable_autocast)
                previous_use_feature_cache = getattr(self.vae, 'use_feature_cache', None)
                if server_args.pipeline_config.vae_tiling and previous_use_feature_cache is not None:
                    self.vae.use_feature_cache = False
                try:
                    with torch.autocast(device_type=current_platform.device_type, dtype=vae_dtype, enabled=autocast_enabled):
                        latents = _denormalize_vae_latents(latents, self.vae)
                        if _videoedit_vae_sp_enabled(server_args, 'decode') and _uses_spatial_tiles(self.vae, latents, latent=True):
                            decoded = distributed_tiled_decode(self.vae, latents, _conditioning_group())
                        else:
                            decoded = _decode_vae_for_videoedit(self.vae, latents)
                finally:
                    if previous_use_feature_cache is not None:
                        self.vae.use_feature_cache = previous_use_feature_cache
            decoded_frames = (decoded / 2 + 0.5).clamp(0, 1)[0].permute(1, 0, 2, 3)
            del decoded, latents
            params.runtime_decoded_video_tensor = decoded_frames if _keep_stage_tensors() else None
            transfer_batch = env_nonnegative_int('NNDEPLOY_DIT_WAN_VIDEOEDIT_DECODE_TRANSFER_FRAMES', 4)
            params.runtime_window_output_frames = decoded_frames_to_images(decoded_frames, transfer_batch)
        finally:
            if not _keep_stage_tensors():
                params.runtime_decoded_video_tensor = None
                params.runtime_latents = None
                batch.latents = None
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

@register_type
class VideoEditWindowPostprocessNode(StateNode):

    def compute(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        params = _videoedit_params(batch)
        frames = params.runtime_window_output_frames
        if frames is None:
            raise ValueError('VideoEdit window output frames are missing')
        if len(frames) != params.infer_len:
            raise ValueError(f'VideoEdit decoded window must contain {params.infer_len} frames, got {len(frames)}')
        params.runtime_window_metadata = {'window_index': params.runtime_window_index, 'num_frames': len(frames), 'height': frames[0].height if frames else None, 'width': frames[0].width if frames else None}
        return batch

    @operation
    def run(self, batch, server_args):
        return self.compute(batch, server_args)

"""Request tensors and state shared by diffusion algorithms."""

from __future__ import annotations

import os
import pprint
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any

import PIL.Image
import torch

from nndeploy.e_dit.config.sampling_params import SamplingParams
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.utils.logging_utils import _sanitize_for_logging, init_logger
from nndeploy.e_dit.utils.perf_logger import RequestMetrics

logger = init_logger(__name__)

SAMPLING_PARAMS_FIELDS = {f.name for f in fields(SamplingParams)}


@dataclass(init=False)
class Req:
    """Mutable request state passed through the pipeline stages."""

    sampling_params: SamplingParams | None = None
    generator: torch.Generator | list[torch.Generator] | None = None
    image_embeds: list[torch.Tensor] = field(default_factory=list)
    original_condition_image_size: tuple[int, int] | None = None
    condition_image: torch.Tensor | PIL.Image.Image | None = None
    vae_image: torch.Tensor | PIL.Image.Image | None = None
    pixel_values: torch.Tensor | PIL.Image.Image | None = None
    video: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    bbox: tuple[int, int, int, int] | None = None
    crop_bbox: tuple[int, int, int, int] | None = None
    crop_video: torch.Tensor | None = None
    crop_mask: torch.Tensor | None = None
    padded_video: torch.Tensor | None = None
    padded_mask: torch.Tensor | None = None
    masked_video: torch.Tensor | None = None
    cond_latents: torch.Tensor | None = None
    cond_masks: torch.Tensor | None = None
    mask_values: torch.Tensor | None = None
    noisy_latents: torch.Tensor | None = None
    preprocessed_image: torch.Tensor | None = None
    output_file_ext: str | None = None
    prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    negative_prompt_embeds: list[torch.Tensor] | torch.Tensor | None = None
    prompt_attention_mask: list[torch.Tensor] | torch.Tensor | None = None
    negative_attention_mask: list[torch.Tensor] | torch.Tensor | None = None
    pooled_embeds: list[torch.Tensor] = field(default_factory=list)
    neg_pooled_embeds: list[torch.Tensor] = field(default_factory=list)
    max_sequence_length: int | None = None
    do_classifier_free_guidance: bool = False
    is_prompt_processed: bool = False
    latents: torch.Tensor | None = None
    noise_pred: torch.Tensor | None = None
    image_latent: torch.Tensor | list[torch.Tensor] | None = None
    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None
    timesteps: torch.Tensor | None = None
    paired_timesteps: torch.Tensor | None = None
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None
    latent_timestep: torch.Tensor | None = None
    latent_shape: tuple[int, int, int, int, int] | None = None
    rope_interpolation_scale: tuple[float, float, float] | None = None
    eta: float = 0.0
    sigmas: list[float] | None = None
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)
    modules: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    is_warmup: bool = False
    metrics: RequestMetrics | None = None
    output: torch.Tensor | None = None
    decoded_video: torch.Tensor | None = None
    output_video: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    audio_sample_rate: int | None = None

    def __init__(self, **kwargs):
        for name, dataclass_field in self.__class__.__dataclass_fields__.items():
            if name in kwargs:
                object.__setattr__(self, name, kwargs.pop(name))
            elif dataclass_field.default is not MISSING:
                object.__setattr__(self, name, dataclass_field.default)
            elif dataclass_field.default_factory is not MISSING:
                object.__setattr__(self, name, dataclass_field.default_factory())

        for name, value in kwargs.items():
            setattr(self, name, value)

        self.validate()

    def __getattr__(self, name: str) -> Any:
        if name == "sampling_params":
            raise AttributeError(name)

        sampling_params = object.__getattribute__(self, "sampling_params")
        if sampling_params is not None and hasattr(sampling_params, name):
            return getattr(sampling_params, name)

        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "sampling_params" or name in self.__class__.__dataclass_fields__:
            object.__setattr__(self, name, value)
            return

        sampling_params = getattr(self, "sampling_params", None)
        if sampling_params is not None and hasattr(sampling_params, name):
            setattr(sampling_params, name, value)
            return

        if sampling_params is None and name in SAMPLING_PARAMS_FIELDS:
            new_sampling_params = SamplingParams()
            object.__setattr__(self, "sampling_params", new_sampling_params)
            setattr(new_sampling_params, name, value)
            return

        object.__setattr__(self, name, value)

    @property
    def batch_size(self) -> int:
        prompt = self.prompt
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt is not None:
            batch_size = 1
        elif isinstance(self.prompt_embeds, torch.Tensor):
            batch_size = self.prompt_embeds.shape[0]
        elif self.prompt_embeds:
            batch_size = self.prompt_embeds[0].shape[0]
        else:
            batch_size = 1
        return batch_size * self.num_outputs_per_prompt

    def output_file_path(self, num_outputs: int = 1, output_idx: int | None = None):
        output_file_name = self.output_file_name
        if num_outputs > 1 and output_file_name and output_idx is not None:
            base, ext = os.path.splitext(output_file_name)
            output_file_name = f"{base}_{output_idx}{ext}"
        if self.output_path is None or not output_file_name:
            return None
        return os.path.join(self.output_path, output_file_name)

    def set_as_warmup(self, warmup_steps: int = 1) -> None:
        self.is_warmup = True
        self.save_output = False
        self.suppress_logs = True
        self.extra["transformer_cache_mode_before_warmup"] = getattr(
            self, "transformer_cache_mode", "off"
        )
        if hasattr(self.sampling_params, "transformer_cache_mode"):
            self.transformer_cache_mode = "off"
        self.extra["cache_dit_num_inference_steps"] = self.num_inference_steps
        self.num_inference_steps = warmup_steps

    def copy_as_warmup(self, warmup_steps: int = 1) -> "Req":
        req = deepcopy(self)
        req.set_as_warmup(warmup_steps)
        return req

    def validate(self) -> None:
        cfg_scale = (
            self.true_cfg_scale
            if self.true_cfg_scale is not None
            else self.guidance_scale
        )
        self.do_classifier_free_guidance = (
            cfg_scale > 1.0 and self.negative_prompt is not None
        )
        if self.negative_prompt_embeds is None:
            self.negative_prompt_embeds = []
        self.metrics = RequestMetrics(request_id=self.request_id)

    def adjust_size(self, server_args: ServerArgs) -> None:
        return None

    def __str__(self) -> str:
        return pprint.pformat(asdict(self), indent=2, width=120)

    def log(self, server_args: ServerArgs) -> None:
        if self.is_warmup or self.suppress_logs:
            return
        logger.info(
            "Sampling params: width=%s height=%s num_frames=%s fps=%s prompt=%s neg_prompt=%s seed=%s infer_steps=%s outputs=%s guidance_scale=%s output_file=%s",
            self.width,
            self.height,
            self.num_frames,
            self.fps,
            _sanitize_for_logging(self.prompt, "prompt"),
            _sanitize_for_logging(self.negative_prompt, "negative_prompt"),
            self.seed,
            self.num_inference_steps,
            self.num_outputs_per_prompt,
            self.guidance_scale,
            self.output_file_path(),
        )


@dataclass
class OutputBatch:
    """Final pipeline output returned by executors."""

    output: torch.Tensor | None = None
    audio: torch.Tensor | None = None
    audio_sample_rate: int | None = None
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_decoded: list[torch.Tensor] | None = None
    error: str | None = None
    output_file_paths: list[str] | None = None

    # logged metrics info, directly from Req.timings
    metrics: Optional["RequestMetrics"] = None

    # For ComfyUI integration: noise prediction from denoising stage
    noise_pred: torch.Tensor | None = None
    peak_memory_mb: float = 0.0

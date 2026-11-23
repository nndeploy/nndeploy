"""Shared helpers for the EraserDiT erase stages.

Numerical helpers preserve the per-window draw order:
encode sample -> init noise -> decode noise.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from nndeploy.e_dit.utils.latent import normalize_latents, denormalize_latents, latent_frame_count

__all__ = [
    "EraserDiTTaskState",
    "field_summary",
    "get_task_state",
    "linear_quadratic_schedule",
    "retrieve_timesteps",
    "get_timesteps",
    "denormalize_latents",
    "normalize_latents",
    "TASK_STATE_KEY",
    "STYLE_VIDEO_KEY",
    "STYLE_MASK_KEY",
    "NEW_FRAMES_KEY",
    "PREFIX_LEN_KEY",
    "ORIG_SIZE_KEY",
    "MODEL_FRAMES_KEY",
]

TASK_STATE_KEY = "eraserdit_task_state"
STYLE_VIDEO_KEY = "eraserdit_style_video"
STYLE_MASK_KEY = "eraserdit_style_mask"
NEW_FRAMES_KEY = "eraserdit_new_frames"
PREFIX_LEN_KEY = "eraserdit_prefix_len"
ORIG_SIZE_KEY = "eraserdit_orig_size"
MODEL_FRAMES_KEY = "eraserdit_model_frames"


@dataclass
class EraserDiTTaskState:
    """Per-request adapter state.

    Lives on ``batch.extra`` so every window sees the same object.  The baseline
    seeds one ``torch.Generator`` per request and never re-seeds it between
    windows, so the generator must survive across windows; the raw tail plays the
    role of ``pre_video_shift``.  Both are isolated per request.
    """

    generator: torch.Generator | None = None
    # Raw (pre-colour-fix) pipeline tail of the previous window, [F, C, H, W].
    prev_raw_tail: torch.Tensor | None = None
    windows_seen: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def get_task_state(batch) -> EraserDiTTaskState:
    state = batch.extra.get(TASK_STATE_KEY)
    if not isinstance(state, EraserDiTTaskState):
        raise RuntimeError(
            "EraserDiT task state is missing; the pipeline must install it on "
            "batch.extra before running the stage chain"
        )
    return state


def field_summary(name: str, value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"{name}=Tensor{tuple(value.shape)}:{value.dtype}@{value.device}"
    if value is None:
        return f"{name}=None"
    return f"{name}={value!r}"


def _linear_quadratic_schedule(
    num_steps: int,
    threshold_noise: float = 0.025,
    linear_steps: Optional[int] = None,
) -> torch.Tensor:
    if linear_steps is None:
        linear_steps = num_steps // 2
    if num_steps < 2:
        return torch.tensor([1.0])
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (
        quadratic_steps**2
    )
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule[:-1])


linear_quadratic_schedule = _linear_quadratic_schedule


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[list] = None,
    sigmas: Optional[list] = None,
    **kwargs,
):
    """Port of the baseline's ``retrieve_timesteps`` (diffusers helper, inlined)."""
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not "
                "support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not "
                "support custom sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_timesteps(scheduler, num_inference_steps: int, strength: float, device):
    """Port of ``LTXVideoToVideoPipeline.get_timesteps``."""
    init_timestep = min(num_inference_steps * strength, num_inference_steps)
    t_start = int(max(num_inference_steps - init_timestep, 0))
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(t_start * scheduler.order)
    return timesteps, num_inference_steps - t_start


__all__ += ["randn_tensor", "latent_frame_count"]

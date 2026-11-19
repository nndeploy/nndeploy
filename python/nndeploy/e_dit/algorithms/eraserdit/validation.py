"""Model sampling validation, independent of HTTP transport."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nndeploy.e_dit.algorithms.eraserdit.config import ERASERDIT_NEGATIVE_PROMPT


class EraserDiTRequestOptions(BaseModel):
    """Request-time knobs for the EraserDiT erase pipeline."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = "There is a bridge over the lake."
    negative_prompt: str = ERASERDIT_NEGATIVE_PROMPT
    seed: int = 42
    fps: int = Field(default=25, ge=1, le=240)
    # The RoPE temporal scale is pinned to 25 by the baseline and does not follow
    # the input frame rate; exposed only so the value is explicit in the request.
    frame_rate: int = Field(default=25, ge=1, le=240)
    num_inference_steps: int = Field(default=50, ge=1, le=1000)
    guidance_scale: float = Field(default=3.0, ge=0.0)
    strength: float = Field(default=0.8, gt=0.0, le=1.0)
    infer_len: int = Field(default=121, ge=9)
    overlap: int = Field(default=9, ge=0)
    compact_tail_padding: bool = False
    streaming_cache_dtype: Literal["bf16", "uint8"] = "bf16"
    max_sequence_length: int = Field(default=128, ge=1, le=512)
    mask_dilate_iter: int = Field(default=9, ge=0)
    mask_ksize: int = Field(default=9, ge=1)
    decode_timestep: float = Field(default=0.0, ge=0.0, le=1.0)
    decode_noise_scale: float = Field(default=0.0, ge=0.0, le=1.0)

    transformer_cache_mode: Literal["off", "teacache", "cache_dit"] = "off"
    transformer_cache_force_compute: bool = False
    cache_text_projections: bool | None = None
    cache_residual_predictor: Literal["none", "linear"] = "none"
    cache_probe_metric: Literal["global", "mask_frame_max"] = "global"
    teacache_threshold: float = Field(default=0.3, gt=0, allow_inf_nan=False)
    max_teacache_consecutive_skip: int = Field(default=1, ge=1)
    teacache_warmup_steps: int = Field(default=4, ge=0)
    cache_dit_front_blocks: int = Field(default=1, ge=1)
    cache_dit_back_blocks: int = Field(default=0, ge=0)
    cache_dit_warmup_steps: int = Field(default=4, ge=0)
    cache_dit_residual_diff_threshold: float = Field(default=0.3, gt=0, lt=1, allow_inf_nan=False)
    cache_dit_max_consecutive_cached_steps: int = Field(default=1, ge=1)
    cache_end_guard_steps: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "EraserDiTRequestOptions":
        if self.overlap >= self.infer_len:
            raise ValueError("overlap must be smaller than infer_len")
        if self.compact_tail_padding and (self.overlap % 8 != 1 or self.infer_len % 8 != 1):
            raise ValueError('compact_tail_padding requires 8k+1 infer_len/overlap')
        if self.mask_ksize % 2 == 0:
            raise ValueError("mask_ksize must be odd")
        from nndeploy.e_dit.algorithms.eraserdit.cache.config import resolve_eraserdit_cache_params
        resolve_eraserdit_cache_params(self)
        return self


def validate_request(payload: dict[str, Any], server_args: Any) -> None:
    from nndeploy.e_dit.algorithms.eraserdit.parallel.validation import validate_nccl_dit
    validate_nccl_dit(server_args, payload['sampling'])
    from nndeploy.e_dit.algorithms.eraserdit.optimizations.compile_config import validate_transformer_compile
    validate_transformer_compile(server_args, payload["sampling"])
    from nndeploy.e_dit.algorithms.eraserdit.cache.config import resolve_eraserdit_cache_params
    resolve_eraserdit_cache_params(
        payload["sampling"], enable_torch_compile=server_args.enable_torch_compile,
    )



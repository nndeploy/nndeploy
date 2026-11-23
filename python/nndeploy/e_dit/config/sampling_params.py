"""Common diffusion sampling parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


def generate_request_id() -> str:
    """Create a stable request identifier for local runs."""
    return f"req_{uuid4().hex[:12]}"


@dataclass
class SamplingParams:
    """Minimal request-time parameters shared by all runtime pipelines."""

    request_id: str = field(default_factory=generate_request_id)
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] | None = None
    seed: int | None = None
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    fps: int = 25
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_scale_2: float | None = None
    true_cfg_scale: float | None = None
    strength: float = 1.0
    num_outputs_per_prompt: int = 1
    output_path: str | None = None
    output_file_name: str | None = None
    output_file_ext: str | None = None
    image_path: str | None = None
    save_output: bool = True
    suppress_logs: bool = False
    profile: bool = False
    profile_all_stages: bool = False
    num_profiled_timesteps: int = -1
    perf_dump_path: str | None = None

    def __post_init__(self) -> None:
        if self.guidance_scale_2 is None:
            self.guidance_scale_2 = self.guidance_scale

    @classmethod
    def from_user_kwargs(cls, **kwargs) -> "SamplingParams":
        return cls(**kwargs)

    @classmethod
    def from_user_sampling_params_args(cls, **kwargs) -> "SamplingParams":
        return cls.from_user_kwargs(**kwargs)


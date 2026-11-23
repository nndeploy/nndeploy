# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang multimodal_gen/runtime/entrypoints/openai.
"""Normal/DMD policy from SGLang VideoEdit dual_service_gateway."""
from typing import Any
from .schemas import VideoRepairRequest

_NORMAL_MODELS = {None, "", "videoedit", "videoedit-normal", "normal"}


_DMD_MODELS = {"videoedit-dmd", "dmd"}


def resolve_variant(model: Any) -> str:
    value = None if model is None else str(model).strip().lower()
    if value in _NORMAL_MODELS:
        return "normal"
    if value in _DMD_MODELS:
        return "dmd"
    raise ValueError(f"Unsupported VideoEdit model: {model}")


def apply_variant_policy(
    request_model: VideoRepairRequest, variant: str
) -> VideoRepairRequest:
    if variant != "dmd":
        return request_model
    return request_model.model_copy(
        update={
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "dynamic_cfg": False,
            "negative_prompt": None,
        }
    )


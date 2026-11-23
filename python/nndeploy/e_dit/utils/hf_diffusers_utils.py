"""Small Hugging Face and diffusers helpers for the minimal runtime."""

from __future__ import annotations

import json
import os
from typing import Any

from transformers import AutoConfig, PretrainedConfig


def maybe_download_model(model_path: str, force_diffusers_model: bool = False) -> str:
    """Phase 0 only supports local model directories."""
    del force_diffusers_model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    return model_path


def _load_json_dict(file_path: str) -> dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload.pop("_diffusers_version", None)
    return payload


def load_json_dict(file_path: str) -> dict[str, Any]:
    """Public helper for loading JSON config payloads."""
    return _load_json_dict(file_path)


def verify_model_config_and_directory(model_path: str) -> dict[str, Any]:
    """Load and validate the root `model_index.json` file."""
    config_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing model_index.json in {model_path}")
    config = _load_json_dict(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid model_index.json in {model_path}")
    return config


def get_diffusers_component_config(component_path: str) -> dict[str, Any]:
    """Load a diffusers component config as a plain dict."""
    component_path = maybe_download_model(component_path)
    config_names = ["generation_config.json"]
    if os.path.basename(component_path) == "scheduler":
        config_names.append("scheduler_config.json")
    else:
        config_names.append("config.json")

    combined: dict[str, Any] = {}
    for config_name in config_names:
        combined.update(_load_json_dict(os.path.join(component_path, config_name)))
    return combined


def get_component_directory(
    model_path: str,
    component_name: str,
    override_path: str | None = None,
) -> str:
    """Resolve a component directory from a diffusers-style model root."""
    if override_path is not None:
        resolved = maybe_download_model(override_path)
    else:
        resolved = os.path.join(maybe_download_model(model_path), component_name)
    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"Missing component directory for '{component_name}': {resolved}"
        )
    return resolved


def get_hf_config(
    model_path: str,
    trust_remote_code: bool = False,
    revision: str | None = None,
    model_override_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PretrainedConfig:
    """Load a transformers config for native model fallback."""
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )
    if model_override_args:
        config.update(model_override_args)
    return config


"""Registered diffusion transformer loading."""

from __future__ import annotations

import os
from typing import Any

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.loader.component_loaders.component_loader import ComponentLoader
from nndeploy.e_dit.models.registry import ModelRegistry
from nndeploy.e_dit.utils.hf_diffusers_utils import load_json_dict


class TransformerLoader(ComponentLoader):
    """Loader for registered diffusion transformers."""

    component_names = ["transformer"]
    expected_library = "diffusers"

    def should_offload(self, server_args: ServerArgs) -> bool:
        return server_args.dit_cpu_offload or server_args.dit_layerwise_offload

    def load_component(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        dtype,
    ) -> tuple[Any, dict[str, Any] | None]:
        del transformers_or_diffusers
        config_path = os.path.join(component_model_path, "config.json")
        config = load_json_dict(config_path)
        architecture = server_args.component_architectures.get(
            component_name
        ) or config.get("_class_name")
        if not architecture:
            raise ValueError(f"No architecture declared for component {component_name!r}")
        model_cls, _ = ModelRegistry.resolve_model_cls(architecture)
        from nndeploy.e_dit.loader.meta_load import load_safetensors_model
        model, loading_info = load_safetensors_model(
            lambda: model_cls.from_config(config), component_model_path, dtype=dtype,
        )
        extra_configs = getattr(server_args.pipeline_config, "component_extra_configs", {}).get(component_name, {})
        for attribute, filename in extra_configs.items():
            setattr(model, attribute, load_json_dict(os.path.join(component_model_path, filename)))
        return model, loading_info

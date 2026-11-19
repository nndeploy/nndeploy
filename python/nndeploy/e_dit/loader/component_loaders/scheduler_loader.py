"""Diffusion scheduler loader."""

from __future__ import annotations

import os
from typing import Any

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.loader.component_loaders.component_loader import ComponentLoader
from nndeploy.e_dit.models.registry import ModelRegistry
from nndeploy.e_dit.utils.hf_diffusers_utils import load_json_dict


class SchedulerLoader(ComponentLoader):
    """Loader for the local flow-match scheduler."""

    component_names = ["scheduler"]
    expected_library = "diffusers"

    def load_component(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        dtype,
    ) -> tuple[Any, dict[str, Any] | None]:
        del dtype, transformers_or_diffusers
        config_path = os.path.join(component_model_path, "scheduler_config.json")
        config = load_json_dict(config_path)
        architecture = server_args.component_architectures.get(
            component_name
        ) or config.get("_class_name", "FlowMatchEulerDiscreteScheduler")
        scheduler_cls, _ = ModelRegistry.resolve_model_cls(architecture)
        scheduler = scheduler_cls.from_pretrained(
            component_model_path,
            local_files_only=True,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        return scheduler, None

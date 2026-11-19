"""Component loader registry and device placement."""

from __future__ import annotations

import importlib
import os
import pkgutil
from abc import ABC
from typing import Any

import torch
from torch import nn

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.loader.utils import (
    _normalize_component_type,
    component_name_to_loader_cls,
    get_memory_usage_of_component,
    log_loading_info,
)
from nndeploy.e_dit.utils.logging_utils import init_logger
from nndeploy.e_dit.utils.platform import current_platform, get_local_torch_device

logger = init_logger(__name__)

class ComponentLoader(ABC):
    """Base class for loading registered model components."""

    component_names: list[str] = []
    expected_library: str = ""
    _loaders_registered = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for component_name in cls.component_names:
            component_name_to_loader_cls[component_name] = cls

    def should_offload(self, server_args: ServerArgs) -> bool:
        del server_args
        return False

    def target_device(self, should_offload: bool, server_args: ServerArgs) -> torch.device:
        if should_offload:
            return torch.device("cpu")
        return get_local_torch_device(server_args.device)

    def target_dtype(
        self,
        server_args: ServerArgs,
        component_name: str,
    ) -> torch.dtype | None:
        return server_args.resolve_component_dtype(_normalize_component_type(component_name))

    def load(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
    ) -> tuple[Any, float]:
        normalized_component = _normalize_component_type(component_name)
        if self.expected_library and transformers_or_diffusers != self.expected_library:
            raise ValueError(
                f"Component '{component_name}' expects {self.expected_library}, "
                f"got {transformers_or_diffusers}"
            )

        gpu_mem_before_loading = current_platform.get_available_gpu_memory()
        dtype = self.target_dtype(server_args, normalized_component)
        should_offload = self.should_offload(server_args)
        device = self.target_device(should_offload, server_args)

        logger.info(
            "Loading %s from %s on %s with dtype=%s. avail mem: %.2f GB",
            normalized_component,
            component_model_path,
            device,
            dtype,
            gpu_mem_before_loading,
        )

        component, loading_info = self.load_component(
            component_model_path=component_model_path,
            server_args=server_args,
            component_name=normalized_component,
            transformers_or_diffusers=transformers_or_diffusers,
            dtype=dtype,
        )
        log_loading_info(normalized_component, loading_info)

        if isinstance(component, nn.Module):
            if dtype is not None:
                component = component.to(dtype=dtype)
            component = component.to(device).eval()
        consumed = max(
            0.0,
            gpu_mem_before_loading - current_platform.get_available_gpu_memory(),
        )
        logger.info(
            "Loaded %s: %s (device=%s, dtype=%s, size=%.2f GB)",
            normalized_component,
            component.__class__.__name__,
            getattr(component, "device", "n/a"),
            getattr(component, "dtype", dtype),
            get_memory_usage_of_component(component) or 0.0,
        )
        return component, consumed

    def load_component(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        dtype: torch.dtype | None,
    ) -> tuple[Any, dict[str, Any] | None]:
        del component_model_path, server_args, component_name, transformers_or_diffusers, dtype
        raise NotImplementedError

    @classmethod
    def _ensure_loaders_registered(cls) -> None:
        if cls._loaders_registered:
            return
        package_dir = os.path.dirname(__file__)
        package_name = __package__ or "nndeploy.e_dit.loader.component_loaders"
        for _, name, _ in pkgutil.iter_modules([package_dir]):
            if name == "component_loader":
                continue
            importlib.import_module(f".{name}", package=package_name)
        cls._loaders_registered = True

    @classmethod
    def for_component_type(
        cls,
        component_name: str,
        transformers_or_diffusers: str,
    ) -> "ComponentLoader":
        cls._ensure_loaders_registered()
        normalized_name = _normalize_component_type(component_name)
        loader_cls = component_name_to_loader_cls.get(normalized_name)
        if loader_cls is None:
            raise ValueError(f"No loader registered for component '{component_name}'")
        loader = loader_cls()
        if loader.expected_library and transformers_or_diffusers != loader.expected_library:
            raise ValueError(
                f"Loader for '{component_name}' expects {loader.expected_library}, "
                f"got {transformers_or_diffusers}"
            )
        return loader


class PipelineComponentLoader:
    """Convenience entrypoint used by composed pipelines."""

    @staticmethod
    def load_component(
        component_name: str,
        component_model_path: str,
        transformers_or_diffusers: str,
        server_args: ServerArgs,
    ) -> tuple[Any, float]:
        overrides = getattr(server_args.pipeline_config, "component_loaders", {})
        path = overrides.get(component_name)
        if path is not None:
            module, name = path.split(":", 1)
            loader_class = getattr(importlib.import_module(module), name)
            if not issubclass(loader_class, ComponentLoader):
                raise TypeError(f"Component loader {path} must inherit ComponentLoader")
            loader = loader_class()
        else:
            loader = ComponentLoader.for_component_type(component_name, transformers_or_diffusers)
        return loader.load(
            component_model_path=component_model_path,
            server_args=server_args,
            component_name=component_name,
            transformers_or_diffusers=transformers_or_diffusers,
        )

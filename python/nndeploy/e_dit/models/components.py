"""Component loading and ownership for diffusion resources."""

from __future__ import annotations

import os
from typing import Any

import torch

from nndeploy.e_dit.config.server_args import ServerArgs, set_global_server_args
from nndeploy.e_dit.loader.component_loaders.component_loader import PipelineComponentLoader
from nndeploy.e_dit.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from nndeploy.e_dit.utils.logging_utils import init_logger
from nndeploy.e_dit.utils.platform import current_platform

logger = init_logger(__name__)


class ComponentOwner:
    """Load named model components; no execution or request ownership."""
    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        return self.modules.get(module_name, default_value)

    def add_module(self, module_name: str, module: Any) -> None:
        self.modules[module_name] = module

    def _load_config(self) -> dict[str, Any]:
        self.model_path = maybe_download_model(
            self.model_path, force_diffusers_model=True
        )
        if self._declared_model_index is not None:
            return {name: list(entry) for name, entry in self._declared_model_index.items()}
        return verify_model_config_and_directory(self.model_path)

    @property
    def required_config_modules(self) -> list[str]:
        return self._required_config_modules

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        override_path = server_args.component_paths.get(module_name)
        if override_path is not None:
            return maybe_download_model(override_path)
        return os.path.join(self.model_path, load_module_name)

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        from time import perf_counter
        from nndeploy.e_dit.loader.contracts import (
            ComponentSpec, ComponentLoadReport, order_components,
        )
        adapter = getattr(self, 'adapter', None)
        specs = order_components(adapter.component_specs(server_args) if adapter else
                                 [ComponentSpec(name) for name in self.required_config_modules])
        supplied = dict(loaded_modules or {})
        unknown = set(supplied) - {spec.name for spec in specs}
        if unknown:
            raise ValueError(f'Undeclared injected components: {sorted(unknown)}')
        # A fully injected graph must not download or read a checkpoint.
        needs_index = any(spec.name not in supplied for spec in specs)
        model_index = self._load_config() if needs_index else {}
        self.loading_report = []
        loaded_components = {}
        module = None
        try:
            for spec in specs:
                name = spec.name
                source = None
                start = perf_counter()
                status = 'injected' if name in supplied else 'loaded'
                missing_dependencies = set(spec.dependencies) - set(loaded_components)
                if missing_dependencies:
                    raise ValueError(f'{name}: unavailable dependencies {sorted(missing_dependencies)}')
                if name in supplied:
                    module = supplied[name]
                    if module is None:
                        raise ValueError(f'{name}: injected component cannot be None')
                else:
                    entry = model_index.get(name)
                    if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not all(isinstance(v, str) and v for v in entry):
                        if not spec.required and name not in server_args.component_paths:
                            self.loading_report.append(ComponentLoadReport(name, None, 'skipped'))
                            continue
                        raise ValueError(f"Required module '{name}' missing or invalid in model_index.json")
                    library, architecture = entry
                    load_name = self._extra_config_module_map.get(name, name)
                    source = self._resolve_component_path(server_args, name, load_name)
                    module, memory_usage = PipelineComponentLoader.load_component(
                        component_name=load_name, component_model_path=source,
                        transformers_or_diffusers=library, server_args=server_args,
                    )
                    self.memory_usages[load_name] = memory_usage
                loaded_components[name] = module
                parameters = list(module.parameters()) if isinstance(module, torch.nn.Module) else []
                self.loading_report.append(ComponentLoadReport(
                    name, source, status, perf_counter() - start,
                    sum(p.numel() for p in parameters),
                    tuple(sorted({str(p.dtype) for p in parameters})),
                    tuple(sorted({str(p.device) for p in parameters})),
                ))
                parameters.clear()
            return loaded_components
        except BaseException as exc:
            self.loading_report.append(ComponentLoadReport(
                name, source, 'failed', perf_counter() - start, error=str(exc),
            ))
            # Drop only our references; never close or move caller-owned modules.
            loaded_components.clear()
            self.memory_usages.clear()
            module = None
            raise

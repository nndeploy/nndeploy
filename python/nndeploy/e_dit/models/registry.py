"""Lazy component architecture resolution for registered algorithms."""

from __future__ import annotations

import ast
import importlib
import os
from functools import lru_cache
from typing import Any

from torch import nn

from nndeploy.e_dit.utils.logging_utils import init_logger

logger = init_logger(__name__)
MODELS_PATH = os.path.dirname(__file__)


@lru_cache(maxsize=None)
def _discover_models() -> dict[str, tuple[str, str]]:
    discovered: dict[str, tuple[str, str]] = {}
    for root, _dirs, files in os.walk(MODELS_PATH):
        for filename in files:
            if not filename.endswith('.py') or filename.startswith('__'):
                continue
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(path, MODELS_PATH)
            module_name = 'nndeploy.e_dit.models.' + rel_path[:-3].replace(os.sep, '.')
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    tree = ast.parse(fh.read(), filename=filename)
            except Exception as exc:
                logger.debug('Skipping model parse for %s: %s', path, exc)
                continue
            entry_candidates: list[str] = []
            first_class_name: str | None = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and first_class_name is None:
                    first_class_name = node.name
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == 'EntryClass':
                            value = node.value
                            if isinstance(value, ast.Name):
                                entry_candidates.append(value.id)
                            elif isinstance(value, (ast.List, ast.Tuple)):
                                for elt in value.elts:
                                    if isinstance(elt, ast.Name):
                                        entry_candidates.append(elt.id)
                                    elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        entry_candidates.append(elt.value)
            if not entry_candidates and first_class_name is not None:
                entry_candidates.append(first_class_name)
            for class_name in entry_candidates:
                discovered[class_name] = (module_name, class_name)
    return discovered


class _ModelRegistry:
    def resolve_model_cls(self, architecture: str) -> tuple[type[nn.Module], str]:
        discovered = dict(_discover_models())
        from nndeploy.e_dit.registry import component_paths
        discovered.update({name: tuple(path.split(":", 1)) for name, path in component_paths().items()})
        if architecture not in discovered:
            raise ValueError(f'Unsupported model architecture: {architecture}')
        module_name, class_name = discovered[architecture]
        module = importlib.import_module(module_name)
        model_cls = getattr(module, class_name)
        return model_cls, architecture

    def architectures(self) -> list[str]:
        from nndeploy.e_dit.registry import component_paths
        return sorted(set(_discover_models()) | set(component_paths()))


ModelRegistry = _ModelRegistry()

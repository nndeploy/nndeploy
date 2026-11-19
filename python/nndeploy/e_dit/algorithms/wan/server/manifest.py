"""VideoEdit-owned validation for the common service manifest."""
from functools import partial
from typing import Literal

from pydantic import Field

from nndeploy.server import ServiceAdapter
from nndeploy.server.config import StrictModel, ExecutionConfig


class ServiceOptions(StrictModel):
    models: dict[str, dict] = Field(min_length=1)
    execution: dict | None = None
    secret_key_env: str | None = None
    contract: Literal['backend', 'gateway'] = 'gateway'
    storage: dict | None = None
    audit_directory: str | None = None


def create_service(**options):
    from . import add_videoedit
    value = ServiceOptions.model_validate(options)
    if set(value.models) - {'normal', 'dmd'}:
        raise ValueError('VideoEdit models must contain normal and/or dmd configurations')
    for variant, params in value.models.items():
        expected = 'wan_videoedit_dmd' if variant == 'dmd' else 'wan_videoedit'
        if params.get('model', expected) != expected:
            raise ValueError(f'{variant} requires {expected}')
    execution = dict(isolation='process', workers=1, max_queue_size=0)
    execution.update(value.execution or {})
    if execution.get('workers') != 1 or execution.get('max_queue_size') != 0 or execution.get('scheduling', 'serial') != 'serial':
        raise ValueError('VideoEdit requires one serial worker with no waiting queue')
    ExecutionConfig.model_validate(execution)
    return ServiceAdapter(partial(add_videoedit, **value.model_dump(exclude_none=True)), exclusive_key='videoedit')

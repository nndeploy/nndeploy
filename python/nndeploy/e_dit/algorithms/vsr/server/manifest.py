"""VSR-owned validation for the common service manifest."""
from functools import partial

from pydantic import Field

from nndeploy.server import ServiceAdapter
from nndeploy.server.config import StrictModel, ExecutionConfig


class ServiceOptions(StrictModel):
    params: dict = Field(min_length=1)
    execution: dict | None = None
    secret_key_env: str | None = None


def create_service(**options):
    from nndeploy.e_dit import DiTParams
    from .app import add_vsr
    value = ServiceOptions.model_validate(options)
    params = DiTParams(**value.params)
    if params.model != 'wan_vsr':
        raise ValueError('VSR endpoints require model="wan_vsr"')
    execution = dict(isolation='process', workers=1, max_queue_size=0)
    execution.update(value.execution or {})
    if execution.get('workers') != 1 or execution.get('max_queue_size') != 0 or execution.get('scheduling', 'serial') != 'serial':
        raise ValueError('VSR requires one serial worker with no waiting queue')
    ExecutionConfig.model_validate(execution)
    return ServiceAdapter(partial(add_vsr, **value.model_dump(exclude_none=True)), exclusive_key='vsr')

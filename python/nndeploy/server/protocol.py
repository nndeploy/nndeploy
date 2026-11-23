"""Public Deploy and task schemas, independent of execution bookkeeping."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


class DeployRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    inputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = None

    @field_validator('callback_url')
    @classmethod
    def check_callback(cls, value):
        from .callbacks import validate_url
        return validate_url(value)

    mode: Literal['sync', 'async', 'stream'] = 'sync'
    wait_timeout_seconds: float | None = Field(default=None, gt=0, le=3600, allow_inf_nan=False)

    @model_validator(mode='after')
    def check_wait(self):
        if self.wait_timeout_seconds is not None and self.mode != 'sync':
            raise ValueError('wait_timeout_seconds is only supported for sync requests')
        return self


def node_request(config, body, *, untyped=False):
    """Keep transport options compatible while accepting nodes at the root."""
    if (('inputs' in body or 'parameters' in body)
            and not ({'inputs', 'parameters'} & (config.inputs.keys() | config.parameters.keys()))):
        return body
    from .parameters import split_node_request
    fields = config.inputs.keys() | config.parameters.keys()
    options = {key: value for key, value in body.items()
               if key in ('mode', 'wait_timeout_seconds', 'idempotency_key', 'callback_url') and key not in fields}
    # Only declared node fields belong to the shorthand. Preserve unknown
    # envelope keys so strict transport validation rejects them before any
    # batch item is admitted. Untyped code entries also accept free-form inputs.
    nodes = {key: value for key, value in body.items() if key in fields or untyped and key not in options}
    unknown = {key: value for key, value in body.items() if key not in nodes and key not in options}
    inputs, parameters = split_node_request(config, nodes)
    return {'inputs': inputs, 'parameters': parameters, **options, **unknown}


class TaskResponse(BaseModel):
    task_id: str
    request_id: str | None = None
    name: str
    state: Literal['pending', 'dispatched', 'running', 'succeeded', 'failed', 'cancelled', 'expired']
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    inputs: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    outputs: Any = None
    error: dict[str, Any] | None = None
    cancel_requested: bool = False
    purge_after: float | None = None
    callback: dict[str, Any] | None = None


class TaskInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    sequence: int = Field(ge=1, le=100001)
    data: Any = None
    end: bool = False

    @model_validator(mode='after')
    def check_end(self):
        if self.end and self.data is not None:
            raise ValueError('End-of-input messages cannot carry data')
        return self


class BatchItem(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    inputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = None

    @field_validator('callback_url')
    @classmethod
    def check_callback(cls, value):
        from .callbacks import validate_url
        return validate_url(value)

    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    items: list[BatchItem] = Field(min_length=1, max_length=64)
    mode: Literal['sync', 'async'] = 'sync'
    wait_timeout_seconds: float | None = Field(default=None, gt=0, le=3600, allow_inf_nan=False)

    @model_validator(mode='after')
    def check_wait(self):
        if self.wait_timeout_seconds is not None and self.mode != 'sync':
            raise ValueError('wait_timeout_seconds is only supported for sync requests')
        return self

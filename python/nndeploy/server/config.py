"""Plain workflow.server.json configuration and supported node discovery."""

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer
from .mcp.config import ToolConfig


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {value}")))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FieldConfig(StrictModel):
    type: Literal["string", "integer", "number", "boolean", "json", "image", "file", "array", "tensor"] = "json"
    node: list[str] | None = None
    edge: str | None = None
    media_type: str | None = None
    sample_rate: int | None = Field(default=None, ge=8000, le=192000)
    required: bool = True
    default: Any = None
    exposed: list[list[str]] | None = None
    minimum: float | None = None
    maximum: float | None = None
    dtype: Literal['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                   'float16', 'float32', 'float64'] | None = None
    shape: list[int | None] | None = Field(default=None, max_length=8)
    max_elements: int | None = Field(default=None, ge=1, le=16777216)
    max_bytes: int | None = Field(default=None, ge=1, le=64 * 1024**2)
    device: str | None = Field(default=None, pattern=r'^[a-z][a-z0-9_]*(?::[0-9]{1,5})?$')
    data_format: Literal['N', 'NC', 'NCL', 'S1D', 'NCHW', 'NHWC', 'OIHW', 'NCDHW', 'NDHWC', 'NDCHW'] | None = None

    @model_serializer(mode='wrap')
    def serialize(self, handler):
        value = handler(self)
        if self.exposed is None:
            value.pop('exposed', None)
        if self.media_type is None:
            value.pop('media_type', None)
        if self.sample_rate is None:
            value.pop('sample_rate', None)
        if self.data_format is None:
            value.pop('data_format', None)
        if self.type not in ('array', 'tensor'):
            # Preserve saved scalar schemas and shared queue fingerprints when
            # adding optional numeric-array configuration support.
            for name in ('dtype', 'shape', 'max_elements', 'max_bytes', 'device'):
                value.pop(name, None)
        return value

    @model_validator(mode="after")
    def check(self):
        if self.exposed is not None:
            from .parameters import parameter_paths
            if self.node is None or self.type != 'json' or not isinstance(self.default, dict):
                raise ValueError('exposed requires a node parameter object')
            available = {tuple(path) for path in parameter_paths(self.default)}
            if any(tuple(path) not in available for path in self.exposed):
                raise ValueError('exposed must name existing node parameter fields')
        if self.data_format is not None:
            if self.type != 'tensor' or self.shape is None or len(self.shape) != len(self.data_format):
                raise ValueError('data_format requires a Tensor field with the corresponding rank')
            if self.data_format == 'S1D' and self.shape[1] != 1:
                raise ValueError('S1D requires a fixed singleton middle dimension')
        if self.media_type is not None:
            from .storage.media import MEDIA_EXTENSIONS
            numeric_file = self.type in ('array', 'tensor') and (self.media_type == 'application/x-npy' or self.is_audio())
            if not numeric_file and (self.type not in ('image', 'file') or self.media_type not in MEDIA_EXTENSIONS):
                raise ValueError('media_type requires a supported file MIME type; numeric fields use NPY or audio')
            if self.type == 'image' and not self.media_type.startswith('image/'):
                raise ValueError('Image fields require an image MIME type')
        numeric = self.type in ('array', 'tensor')
        if self.is_audio():
            if (self.sample_rate is None or self.dtype != 'float32' or self.shape is None or
                    len(self.shape) not in (1, 2) or
                    len(self.shape) == 2 and (self.shape[1] is None or not 1 <= self.shape[1] <= 8) or
                    self.shape[0] == 0):
                raise ValueError('Audio samples require float32, sample_rate and shape [frames] or [frames, 1–8 channels]')
            if self.data_format not in (None, 'N', 'NC'):
                raise ValueError('Audio Tensor formats use N or NC sample/channel axes')
        elif self.sample_rate is not None:
            raise ValueError('sample_rate is only valid for numeric audio fields')
        if numeric:
            if self.max_elements is None:
                self.max_elements = 1048576
            if self.max_bytes is None:
                self.max_bytes = 8 * 1024**2
            if self.dtype is None or self.shape is None:
                raise ValueError('Array and Tensor fields require dtype and shape')
            if any(dim is not None and not 0 <= dim <= min(self.max_elements, 2**31 - 1) for dim in self.shape):
                raise ValueError('Shape dimensions must be bounded nonnegative integers or null')
            if all(dim is not None for dim in self.shape):
                from .arrays import check_shape
                check_shape(self, self.shape)
            if self.type == 'tensor' and (len(self.shape) > 5 or self.dtype == 'bool'):
                raise ValueError('Native Tensor fields support numeric dtypes and at most five dimensions')
        elif any(getattr(self, name) is not None for name in ('dtype', 'shape', 'max_elements', 'max_bytes', 'device')):
            raise ValueError('Array metadata is only valid on array or tensor fields')
        if self.device is not None:
            if self.type != 'tensor':
                raise ValueError('Only Tensor inputs can specify a target device')
            from nndeploy.base import DeviceType
            DeviceType(self.device)
        if self.node is not None and (not self.node or any(not p for p in self.node)):
            raise ValueError("Node paths must contain nonempty names")
        if (self.node is not None) + (self.edge is not None) > 1:
            raise ValueError("Use either node or edge")
        if self.edge == "":
            raise ValueError("Edge names cannot be empty")
        if not self.required:
            if "default" not in self.model_fields_set:
                raise ValueError("Optional fields require an explicit default")
            self.validate_value(self.default)
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum exceeds maximum")
        return self

    def check_file(self, record):
        if self.type == 'image' and not record['media_type'].startswith('image/'):
            raise ValueError('Expected a validated image file')
        if self.media_type is not None and record['media_type'] != self.media_type:
            raise ValueError('File MIME type does not match the field')

    def is_file(self):
        return self.type in ('image', 'file') or self.media_type == 'application/x-npy' or self.is_audio()

    def is_audio(self):
        return self.type in ('array', 'tensor') and self.media_type in (
            'audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/mp4')

    def output_media_type(self):
        return self.media_type or ('image/png' if self.type == 'image' else 'application/octet-stream')

    def validate_value(self, value, *, allow_local_files=False):
        if allow_local_files and self.is_file():
            path = value if isinstance(value, str) else None
            if isinstance(value, dict) and set(value) == {'path'}:
                path = value['path']
            elif isinstance(value, dict) and set(value) == {'file_id'}:
                candidate = value['file_id']
                if isinstance(candidate, str) and not candidate.startswith('file_'):
                    path = candidate
            if path is not None:
                if not isinstance(path, str) or not path or '\x00' in path:
                    raise ValueError('Local file path must be a nonempty string')
                from pathlib import Path
                return {'path': str(Path(path).expanduser().resolve())}
        if self.type in ('array', 'tensor') and self.is_file():
            if not isinstance(value, dict) or set(value) != {'file_id'} or not isinstance(value['file_id'], str) or not value['file_id']:
                raise ValueError('Numeric file fields require a file_id')
            return copy.deepcopy(value)
        if self.type in ('array', 'tensor'):
            from .arrays import validate
            return validate(self, value)
        valid = {"string": isinstance(value, str), "integer": type(value) is int,
                 "number": type(value) in (int, float) and math.isfinite(value),
                 "boolean": type(value) is bool, "json": True,
                 "image": isinstance(value, dict) and set(value) == {"file_id"} and isinstance(value.get("file_id"), str),
                 "file": isinstance(value, dict) and set(value) == {"file_id"} and isinstance(value.get("file_id"), str)}[self.type]
        if not valid:
            raise ValueError(f"Expected {self.type}")
        json.dumps(value, allow_nan=False)
        if self.type in ("integer", "number"):
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"Value is below {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"Value exceeds {self.maximum}")
        return copy.deepcopy(value)


class BatchingConfig(StrictModel):
    enabled: bool = False
    max_size: int = Field(default=8, ge=1, le=64)
    max_wait_ms: float = Field(default=5.0, ge=0, le=1000, allow_inf_nan=False)


class WarmupRequest(StrictModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_size(self):
        encoded = json.dumps(self.model_dump(), allow_nan=False).encode('utf-8')
        if len(encoded) > 1024 * 1024:
            raise ValueError('Each warmup request is limited to 1 MiB; use file references for media')
        return self


class ExecutionConfig(StrictModel):
    scheduling: Literal['serial', 'concurrent', 'engine'] = 'serial'
    max_concurrency_per_worker: int = Field(default=1, ge=1, le=64)
    cooperative_cancel: bool = False
    workers: int = Field(default=1, ge=1, le=64)
    max_queue_size: int = Field(default=32, ge=0)
    queue_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    isolation: Literal['thread', 'process'] = 'thread'
    cpu_threads: int | None = Field(default=None, ge=1, le=65536)
    startup_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    execution_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    http_wait_timeout_seconds: float = Field(default=60.0, gt=0, le=3600, allow_inf_nan=False)
    restore_on_restart: bool = True
    task_retention_seconds: float | None = Field(default=None, ge=60, allow_inf_nan=False)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    warmup: list[WarmupRequest] = Field(default_factory=list, max_length=16)
    devices: list[str] = Field(default_factory=list, max_length=64)
    memory_bytes: dict[str, int] = Field(default_factory=dict, max_length=64)
    temporary_memory_bytes: dict[str, int] = Field(default_factory=dict, max_length=64)
    priority: int = Field(default=0, ge=-10, le=10)

    @model_serializer(mode='wrap')
    def serialize(self, handler):
        value = handler(self)
        # Optional budgeting must not change existing shared fingerprints.
        if not self.devices:
            value.pop('devices', None)
        if not self.memory_bytes:
            value.pop('memory_bytes', None)
        if not self.temporary_memory_bytes:
            value.pop('temporary_memory_bytes', None)
        if self.priority == 0:
            value.pop('priority', None)
        if self.cpu_threads is None:
            value.pop('cpu_threads', None)
        return value

    @model_validator(mode='after')
    def check_timeout(self):
        if self.cpu_threads is not None and self.isolation != 'process':
            raise ValueError('cpu_threads requires process isolation because native thread settings can affect the entire process')
        if (len(set(self.devices)) != len(self.devices) or
            any(not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', device) for device in self.devices)):
            raise ValueError('Devices must be distinct bounded identifiers')
        if set(self.memory_bytes) - set(self.devices) or any(type(value) is not int or not 0 < value <= (2**63 - 1) // self.workers for value in self.memory_bytes.values()):
            raise ValueError('memory_bytes must specify positive per-worker budgets for declared devices')
        batch_size = self.batching.max_size if self.batching.enabled else 1
        if (set(self.temporary_memory_bytes) - set(self.devices) or
            any(type(value) is not int or value <= 0 or
                value * batch_size + self.memory_bytes.get(device, 0) * self.workers > 2**63 - 1
                for device, value in self.temporary_memory_bytes.items())):
            raise ValueError('temporary_memory_bytes must specify positive per-request budgets fitting the declared device integer range')
        if self.batching.enabled and (self.scheduling != 'serial' or self.isolation != 'thread'):
            raise ValueError('Dynamic batching requires a serial thread worker; engines manage their own batches')
        if self.scheduling == 'serial' and self.max_concurrency_per_worker != 1:
            raise ValueError('Serial execution requires one request per worker')
        if self.cooperative_cancel and self.scheduling != 'engine':
            raise ValueError('Cooperative cancellation requires an async engine')
        if self.scheduling != 'serial' and self.isolation != 'thread':
            raise ValueError('Concurrent and engine scheduling currently require thread isolation')
        if self.execution_timeout_seconds is not None and self.isolation != 'process' and not (
                self.scheduling == 'engine' and self.cooperative_cancel):
            raise ValueError('Execution deadlines require process isolation or a cooperatively cancellable async engine')
        return self

    @property
    def capacity(self):
        return self.max_queue_size + self.workers * self.request_slots

    @property
    def request_slots(self):
        return self.batching.max_size if self.batching.enabled else self.max_concurrency_per_worker


class StreamingConfig(StrictModel):
    enabled: bool = False
    duplex: bool = False
    max_pending_inputs: int = Field(default=64, ge=1, le=1024)
    max_events: int = Field(default=10000, ge=1, le=100000)
    max_event_bytes: int = Field(default=65536, ge=64, le=1048576)
    max_duration_seconds: float = Field(default=300.0, gt=0, le=86400, allow_inf_nan=False)

    @model_serializer(mode='wrap')
    def serialize(self, handler):
        value = handler(self)
        if not self.duplex:
            value.pop('duplex', None)
            value.pop('max_pending_inputs', None)
        return value

    @model_validator(mode='after')
    def check_duplex(self):
        if self.duplex and not self.enabled:
            raise ValueError('Duplex inputs require streaming outputs')
        return self

    def input_state(self):
        if not self.duplex:
            return None
        return dict(received=0, consumed=0, closed=False, max_events=self.max_events,
                    max_bytes=self.max_event_bytes, max_pending=self.max_pending_inputs)


class ServerConfig(StrictModel):
    mcp: ToolConfig | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        value = handler(self)
        if self.mcp is None:
            value.pop("mcp", None)
        return value

    name: str = Field(default="default", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    inputs: dict[str, FieldConfig] = Field(default_factory=dict)
    parameters: dict[str, FieldConfig] = Field(default_factory=dict)
    outputs: dict[str, FieldConfig] = Field(default_factory=dict)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)

    @model_validator(mode="after")
    def check_targets(self):
        if self.streaming.duplex and 'input_stream' in self.parameters:
            raise ValueError('input_stream is reserved for duplex input')
        if self.execution.batching.enabled and self.streaming.enabled:
            raise ValueError('Dynamic batching requires non-streaming outputs; streaming engines batch internally')
        seen = set()
        for group in ("inputs", "parameters", "outputs"):
            for name, field in getattr(self, group).items():
                if group == 'outputs' and field.device is not None:
                    raise ValueError('Output tensors are copied to host; target device only applies to inputs')
                if not name:
                    raise ValueError("Field names cannot be empty")
                if field.node is not None:
                    target = ("parameter" if group == 'parameters' else "node", *field.node)
                elif field.edge is not None:
                    target = ("output" if group == "outputs" else "input", field.edge)
                else:
                    continue
                if target in seen:
                    raise ValueError("Multiple fields write the same target")
                seen.add(target)
        return self

    def validate_payload(self, inputs, parameters):
        result = {}
        for group, values in (("inputs", inputs), ("parameters", parameters)):
            fields = getattr(self, group)
            unknown = values.keys() - fields.keys()
            if unknown:
                raise ValueError(f"Unknown {group}: {', '.join(sorted(unknown))}")
            result[group] = {}
            for name, field in fields.items():
                if name not in values and field.required:
                    raise ValueError(f"Missing {group}.{name}")
                try:
                    result[group][name] = field.validate_value(values.get(name, field.default), allow_local_files=True)
                    if group == 'parameters' and field.node is not None and field.type == 'json' and isinstance(field.default, dict):
                        from .parameters import parameter_changes
                        changes = parameter_changes(field.default, result[group][name], name)
                        if name in values and field.exposed is not None:
                            from .parameters import exposed_defaults
                            parameter_changes(exposed_defaults(field.default, field.exposed), values[name], name)
                        # Defaults are resident configuration, protected by the
                        # entry's resource lease. Only request overrides belong
                        # to the task and its caller-owned file references.
                        result[group][name] = changes
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError(f"Invalid {group}.{name}: {exc}") from exc
        return result

    def public_schema(self):
        schema = {}
        for group in ("inputs", "parameters", "outputs"):
            schema[group] = {}
            for name, field in getattr(self, group).items():
                if group == 'inputs' and field.node is not None:
                    key = 'path_' if field.is_file() else 'str_'
                    if any(p.node == field.node and p.exposed is not None and [key] not in p.exposed
                           and key in p.default for p in self.parameters.values()):
                        continue
                excluded = {"node", "edge", "default", "device"}
                if field.type not in ('array', 'tensor'):
                    excluded.update({'max_elements', 'max_bytes'})
                value = field.model_dump(exclude=excluded, exclude_none=True)
                if not field.required:
                    value["default"] = field.default
                if group == 'parameters' and field.exposed is not None:
                    from .parameters import exposed_defaults
                    if not field.exposed:
                        continue
                    value['default'] = exposed_defaults(field.default, field.exposed)
                schema[group][name] = value
        return schema


def validate_request(config, inputs, parameters, *, input_model=None, untyped=False):
    """Use the same validation for external requests and configured warmup."""
    inputs = copy.deepcopy(inputs)
    # A closed input-node path uses the workflow value, including for legacy
    # requests. It cannot bypass the node's selected public fields via inputs.
    for field in config.parameters.values():
        if field.exposed is None:
            continue
        for input_name, input_field in config.inputs.items():
            if input_field.node != field.node:
                continue
            key = 'path_' if input_field.is_file() else 'str_'
            if [key] not in field.exposed and key in field.default:
                if input_name in inputs:
                    raise ValueError(f'Node parameter is not exposed: {input_name}.{key}')
                inputs[input_name] = {'path': field.default[key]} if input_field.is_file() else field.default[key]
    if parameters and input_model is None and not untyped:
        # Existing input-node fields can also be overridden by node name.
        # Route them through the same input validation/file handling as inputs.
        inputs = copy.deepcopy(inputs)
        for name, patch in parameters.items():
            field = config.parameters.get(name)
            if (field is None or field.node is None or field.type != 'json'
                    or not isinstance(field.default, dict) or not isinstance(patch, dict)):
                continue
            for input_name, input_field in config.inputs.items():
                if input_field.node != field.node:
                    continue
                key = 'path_' if input_field.is_file() else 'str_'
                if key in patch:
                    if input_name in inputs:
                        # A default copied from the schema does not override an
                        # explicitly supplied input. A changed node value does.
                        if patch[key] == field.default.get(key):
                            continue
                    inputs[input_name] = {'path': patch[key]} if input_field.is_file() else patch[key]
    if input_model is not None:
        inputs = input_model.model_validate(inputs, strict=True).model_dump(mode='json')
        if parameters:
            raise ValueError('Typed code inputs do not accept undeclared parameters')
        if config.inputs or config.parameters:
            return config.validate_payload(inputs, {})
        return dict(inputs=inputs, parameters={})
    if untyped:
        if parameters:
            raise ValueError('Declare parameter fields before using parameters')
        json.dumps(inputs, allow_nan=False)
        return dict(inputs=copy.deepcopy(inputs), parameters={})
    return config.validate_payload(inputs, parameters or {})


def suggest_config(graph):
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", graph.get("name_", "default")).strip("._-")[:128] or "default"
    result = dict(name=name, inputs={}, parameters={}, outputs={})
    targets = set()

    def add(group, name, value):
        target = tuple(value.get("node", [])) or value.get("edge")
        if (group, target) in targets:
            return
        targets.add((group, target))
        candidate, count = name, 2
        while candidate in result[group]:
            candidate, count = f"{name}_{count}", count + 1
        result[group][candidate] = value

    def visit(value, path=()):
        for node in value.get("node_repository_", []):
            current = (*path, node["name_"])
            key = node.get("key_")
            from .parameters import node_parameters
            defaults = node_parameters(node)
            if defaults:
                add('parameters', '_'.join(current), dict(type='json', node=list(current), required=False, default=defaults))
            if key == "nndeploy.basic.InputStr":
                add("inputs", "_".join(current), dict(type="string", node=list(current)))
            elif key == "nndeploy.codec.PILImageDecodec":
                add("inputs", "_".join(current), dict(type="image", node=list(current)))
            elif key == "nndeploy.codec.PILImageEncodec":
                add("outputs", "_".join(current), dict(type="image", node=list(current)))
            elif key in ('nndeploy::codec::OpenCvImageDecode', 'nndeploy::codec::OpenCvImageEncode'):
                add('inputs' if key.endswith('Decode') else 'outputs', '_'.join(current), dict(type='image', node=list(current)))
            elif key in ('nndeploy::codec::OpenCvVideoDecode', 'nndeploy::codec::OpenCvVideoEncode'):
                add('inputs' if key.endswith('Decode') else 'outputs', '_'.join(current),
                    dict(type='file', media_type='video/mp4', node=list(current)))
            elif key == "nndeploy.basic.ConsolePrintNode" and not path:
                for edge in node.get("inputs_", []):
                    add("outputs", node["name_"], dict(type="json", edge=edge["name_"]))
            visit(node, current)
    visit(graph)
    # Exposed scalar/PIL/byte ports already describe their value type in the
    # connected nodes. Root Graph declarations may still contain NotSet.
    # Numeric ndarray/Tensor ports need explicit dtype/shape and are not guessed.
    port_types = {'str': 'string', 'int': 'integer', 'float': 'number', 'bool': 'boolean',
                  'Image': 'image', 'PIL.Image.Image': 'image', 'bytes': 'file'}
    for group in ('inputs', 'outputs'):
        for edge in graph.get(group + '_', []):
            declarations = [edge]
            for node in graph.get('node_repository_', []):
                declarations.extend(port for key in ('inputs_', 'outputs_') for port in node.get(key, [])
                                    if port['name_'] == edge['name_'])
            types = {port.get('type_') for port in declarations} - {None, '', 'NotSet'}
            if len(types) == 1 and next(iter(types)) in port_types:
                add(group, edge['name_'], dict(type=port_types[next(iter(types))], edge=edge['name_']))
    return result

"""Public numeric requests exercise real NumPy, Tensor and Graph conversion."""
import asyncio
import copy
import gc
import weakref
import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import nndeploy.base
import nndeploy.dag
from nndeploy import _nndeploy_internal as native
from nndeploy.device import Tensor
from nndeploy.server import Server
from nndeploy.server.config import FieldConfig, ServerConfig
from nndeploy.server.errors import ServerError
from test_db import db


def wire(data, dtype='float32', shape=None):
    return {'dtype': dtype, 'shape': shape if shape is not None else [len(data)], 'data': data}


def config(kind='array', dtype='float32', shape=None, **execution):
    field = {'type': kind, 'dtype': dtype, 'shape': [None] if shape is None else shape, 'max_elements': 1024, 'max_bytes': 8192}
    return {'inputs': {'value': dict(field)}, 'outputs': {'value': dict(field)}, 'execution': execution}


def echo(values):
    value = values['value']
    assert isinstance(value, (np.ndarray, native.device.Tensor))
    return {'value': value}


@pytest.mark.parametrize('dtype,values', [
    ('bool', [True, False]), ('int8', [-128, 127]), ('uint8', [0, 255]),
    ('int16', [-32768, 32767]), ('uint16', [0, 65535]), ('int32', [-2**31, 2**31-1]),
    ('uint32', [0, 2**32-1]), ('int64', [-2**63, 2**63-1]), ('uint64', [0, 2**64-1]),
    ('float16', [0.5, -2.0]), ('float32', [0.5, -2.0]), ('float64', [0.5, -2.0])])
def test_numpy_http_dtype_roundtrip_and_durable_result(tmp_path, dtype, values):
    server = Server(deploy=echo, config=config(dtype=dtype), resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        body = wire(values, dtype)
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}})
        assert response.status_code == 200, response.text
        assert response.json()['outputs']['value'] == body
        saved = client.get('/v1/tasks/' + response.json()['task_id']).json()
        assert saved['outputs']['value'] == body
        assert server.db.get_task(response.json()['task_id'])['inputs']['value'] == body
        schema = client.get('/v1/server/default/schema').json()['schema']
        assert schema['inputs']['value']['dtype'] == dtype
        assert schema['inputs']['value']['shape'] == [None]
        assert schema['inputs']['value']['max_elements'] == 1024


class NumericCopy(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs, tensor=True):
        super().__init__(name, inputs, outputs)
        self.tensor = tensor
        self.set_input_type(native.device.Tensor if tensor else np.ndarray)
        self.set_output_type(native.device.Tensor if tensor else np.ndarray)
        self.buffer = None

    def run(self):
        value = self.get_input(0).get(self)
        assert isinstance(value, native.device.Tensor if self.tensor else np.ndarray)
        array = np.asarray(value)
        if self.buffer is None or self.buffer.shape != array.shape or self.buffer.dtype != array.dtype:
            self.buffer = np.empty_like(array)
        np.copyto(self.buffer, array)
        self.get_output(0).set(Tensor.from_numpy(self.buffer) if self.tensor else self.buffer)
        return nndeploy.base.Status.ok()


def numeric_graph(tensor=True):
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('numeric', [source], [target])
    graph.add_node(NumericCopy('copy', [source], [target], tensor))
    return graph


def tensor_graph():
    return numeric_graph(True)


def test_graph_factory_retains_python_owned_external_ports():
    def factory():
        source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
        references = weakref.ref(source), weakref.ref(target)
        graph = nndeploy.dag.Graph('numeric', [source], [target])
        graph.add_node(NumericCopy('copy', [source], [target]))
        return graph, references
    graph, references = factory()
    gc.collect()
    assert all(reference() is not None for reference in references)
    assert [edge.get_name() for edge in graph.get_all_input()] == ['input']
    assert [edge.get_name() for edge in graph.get_all_output()] == ['output']
    del graph
    gc.collect()
    assert all(reference() is None for reference in references)


def array_graph():
    return numeric_graph(False)


def opencv_array_graph():
    graph = numeric_graph(False)
    node = graph.get_node('copy')
    for info in node.get_input_type_info() + node.get_output_type_info():
        info.set_cvmat_type()
    return graph


def graph_config(kind='tensor', dtype='float32', shape=None, **execution):
    value = config(kind, dtype, [None, 2] if shape is None else shape, **execution)
    value['inputs']['value']['edge'] = 'input'
    value['outputs']['value']['edge'] = 'output'
    return value


@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('dtype,shape,data', [
    ('float32', [2, 2], [1.0, 2.0, 3.0, 4.0]), ('float16', [1, 2, 1, 1], [0.5, 1.5]),
    ('int64', [], [2**60]), ('uint64', [2], [2**63, 2**64-1]), ('int32', [0, 2], [])])
def test_native_tensor_graph_roundtrip_and_buffer_reuse(tmp_path, isolation, dtype, shape, data):
    settings = graph_config(dtype=dtype, shape=shape, isolation=isolation)
    body = wire(data, dtype, shape)
    settings['execution']['warmup'] = [{'inputs': {'value': body}}]
    server = Server(graph_factory=tensor_graph, config=settings, resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        first = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}})
        assert first.status_code == 200, first.text
        assert first.json()['outputs']['value'] == body
        changed = wire([0 for _ in data], dtype, shape)
        second = client.post('/v1/server/default/deploy', json={'inputs': {'value': changed}})
        assert second.status_code == 200, second.text
        assert second.json()['outputs']['value'] == changed
        assert client.get('/v1/tasks/' + first.json()['task_id']).json()['outputs']['value'] == body


@pytest.mark.parametrize('shape', [[2, 2], [2, 2, 3]])
def test_graph_numpy_edges_preserve_supported_opencv_shapes(tmp_path, shape):
    body = wire(list(range(int(np.prod(shape)))), 'float32', shape)
    with TestClient(Server(graph_factory=array_graph, config=graph_config('array', shape=shape), resources=tmp_path, data_dir=tmp_path).app) as client:
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}})
        assert response.status_code == 200, response.text
        assert response.json()['outputs']['value'] == body


@pytest.mark.parametrize('invalid', [
    {'dtype': 'int32', 'shape': [2], 'data': [1, 2]},
    {'dtype': 'float32', 'shape': [2], 'data': [1]},
    {'dtype': 'float32', 'shape': [2], 'data': [[1], [2]]},
    {'dtype': 'float32', 'shape': [-1], 'data': []},
    {'dtype': 'float32', 'shape': [True], 'data': [1]},
    {'dtype': 'float32', 'shape': [10**12], 'data': []},
    {'dtype': 'float32', 'shape': [1, 2], 'data': [1, 2]},
    {'dtype': 'float32', 'shape': [1], 'data': [1e300]},
    {'dtype': 'float32', 'shape': [1], 'data': [True]},
    {'dtype': 'float32', 'shape': [1], 'data': [1], 'device': 'cuda:0'},
])
def test_bad_arrays_are_rejected_before_algorithm_execution(tmp_path, invalid):
    seen = []
    def deploy(values):
        seen.append(True)
        return values
    with TestClient(Server(deploy=deploy, config=config(), resources=tmp_path, data_dir=tmp_path).app) as client:
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': invalid}})
        assert response.status_code == 422, response.text
        assert not seen


@pytest.mark.parametrize('field', [
    {'type': 'tensor', 'dtype': 'bool', 'shape': [2]},
    {'type': 'tensor', 'dtype': 'float32', 'shape': [1] * 6},
    {'type': 'array', 'dtype': 'float32'},
    {'type': 'array', 'dtype': 'float32', 'shape': [100, 100], 'max_elements': 10},
    {'type': 'array', 'dtype': 'float64', 'shape': [2], 'max_bytes': 8},
    {'type': 'array', 'dtype': 'object', 'shape': [2]},
    {'type': 'string', 'dtype': 'float32'},
    {'type': 'tensor', 'dtype': 'float32', 'shape': [2], 'device': 'unknown'},
])
def test_invalid_numeric_field_declarations_fail_early(field):
    with pytest.raises(ValidationError):
        FieldConfig.model_validate(field)


def test_scalar_serialization_stays_compatible_with_existing_queue_fingerprints():
    assert FieldConfig(type='string').model_dump() == {
        'type': 'string', 'node': None, 'edge': None, 'required': True,
        'default': None, 'minimum': None, 'maximum': None}
    saved = ServerConfig.model_validate(config()).model_dump()
    assert ServerConfig.model_validate(saved).model_dump() == saved


@pytest.mark.parametrize('dtype,value', [('int8', 128), ('uint8', -1), ('int32', 1.0), ('bool', 1),
                                        ('float16', 65505.0), ('float64', float('nan')), ('float32', float('inf'))])
def test_element_bounds_reject_overflow_coercion_and_nonfinite_values(dtype, value):
    field = FieldConfig(type='array', dtype=dtype, shape=[1])
    with pytest.raises(ValueError):
        field.validate_value(wire([value], dtype))


def test_element_minimum_and_maximum_apply_to_inputs_and_native_outputs(tmp_path):
    settings = config()
    settings['inputs']['value']['minimum'] = 0.0
    settings['outputs']['value']['maximum'] = 1.0
    with TestClient(Server(deploy=echo, config=settings, resources=tmp_path, data_dir=tmp_path).app) as client:
        assert client.post('/v1/server/default/deploy', json={'inputs': {'value': wire([-1])}}).status_code == 422
        assert client.post('/v1/server/default/deploy', json={'inputs': {'value': wire([2])}}).status_code == 500
        assert client.post('/v1/server/default/deploy', json={'inputs': {'value': wire([1])}}).status_code == 200


@pytest.mark.parametrize('kind,dtype,shape,factory', [
    ('array', 'float32', [4], opencv_array_graph), ('array', 'float32', [2, 2, 1], opencv_array_graph),
    ('array', 'float16', [2, 2], opencv_array_graph), ('array', 'float32', [2, 2], tensor_graph),
    ('tensor', 'float32', [2, 2], array_graph),
])
def test_incompatible_graph_numeric_binding_fails_before_ready(tmp_path, kind, dtype, shape, factory):
    server = Server(graph_factory=factory, config=graph_config(kind, dtype, shape), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        try:
            with pytest.raises(ValueError):
                await server.start()
            assert server.describe('default')['status'] == 'failed'
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('fault', ['dtype', 'shape', 'nonfinite'])
def test_bad_native_outputs_fail_without_corrupting_following_requests(tmp_path, fault):
    def deploy(values):
        array = values['value']
        if array[0] == 1:
            if fault == 'dtype':
                array = array.astype(np.int32)
            elif fault == 'shape':
                array = np.zeros(2048, dtype=np.float32)
            else:
                array[0] = np.inf
        return {'value': array}
    with TestClient(Server(deploy=deploy, config=config(), resources=tmp_path, data_dir=tmp_path).app) as client:
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': wire([1])}})
        assert response.status_code == 500
        good = client.post('/v1/server/default/deploy', json={'inputs': {'value': wire([2])}})
        assert good.status_code == 200 and good.json()['outputs']['value'] == wire([2])


@pytest.mark.parametrize('scheduling', ['serial', 'concurrent', 'engine'])
def test_streaming_arrays_are_encoded_before_event_persistence(tmp_path, scheduling):
    def stream(values):
        yield {'value': values['value']}
        yield {'value': values['value'] * 2}
    async def async_stream(values):
        for value in stream(values):
            yield value
    settings = config(scheduling=scheduling)
    settings['streaming'] = {'enabled': True}
    server = Server(deploy=async_stream if scheduling == 'engine' else stream, config=settings, resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': wire([2, 3])}})
        assert response.status_code == 200, response.text
        assert response.json()['outputs']['value'] == wire([4, 6])
        events = server.db.read_task_events(response.json()['task_id'])
        assert [event['data']['data']['value'] for event in events if event['kind'] == 'delta'] == [wire([2, 3]), wire([4, 6])]


class NumericBatch:
    def batch_key(self, values):
        assert isinstance(values['value'], np.ndarray)
        return list(values['value'].shape)

    def deploy_batch(self, items):
        return [{'value': item['inputs']['value'] + np.float32(1)} for item in items]


def test_dynamic_batches_receive_native_arrays_and_keep_json_task_records(tmp_path):
    settings = config(batching={'enabled': True}, warmup=[{'inputs': {'value': wire([1, 2])}}])
    server = Server(handler_factory=NumericBatch, config=settings, resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/default/deploy/batch', json={'items': [
            {'inputs': {'value': wire([1, 2])}}, {'inputs': {'value': wire([4, 5])}}]})
        assert response.status_code == 200, response.text
        assert [item['outputs']['value'] for item in response.json()['items']] == [wire([2, 3]), wire([5, 6])]


def test_large_numeric_tasks_and_mysql_legacy_text_migration(db, tmp_path):
    from sqlalchemy import inspect
    from sqlalchemy.dialects.mysql import LONGTEXT
    legacy = uuid.uuid4().hex
    db.create_task(legacy, 'legacy-' + legacy, time.time(), {'text': 'keep'}, {})
    if db.engine.dialect.name == 'mysql':
        # The supplied databases are exclusive disposable test databases. Build
        # the prior column layout, then let normal Server startup migrate it.
        with db.engine.begin() as connection:
            for column in ('inputs', 'parameters', 'outputs', 'error'):
                connection.exec_driver_sql(f'ALTER TABLE server_tasks MODIFY {column} TEXT NULL')
            connection.exec_driver_sql("UPDATE alembic_version SET version_num = '0012_config_writes'")
    settings = config(dtype='float64')
    for group in ('inputs', 'outputs'):
        settings[group]['value'].update(max_elements=16384, max_bytes=262144)
    server = Server(deploy=echo, config=settings, resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False))
    assert db.get_task(legacy)['inputs'] == {'text': 'keep'}
    if db.engine.dialect.name == 'mysql':
        assert all(isinstance(column['type'], LONGTEXT) for column in inspect(db.engine).get_columns('server_tasks')
                   if column['name'] in ('inputs', 'parameters', 'outputs', 'error'))
    body = wire([0.1234567890123456] * 8192, 'float64')
    with TestClient(server.app) as client:
        result = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}}, headers={'Idempotency-Key': legacy})
        assert result.status_code == 200, result.text
        assert result.json()['outputs']['value'] == body
        task_id = result.json()['task_id']
        assert db.get_task(task_id)['inputs']['value'] == body
        assert client.get('/v1/tasks/' + task_id).json()['outputs']['value'] == body
        replay = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}}, headers={'Idempotency-Key': legacy})
        assert replay.json() == result.json()

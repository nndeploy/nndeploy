"""Explicit native Tensor formats must move values, not just relabel shapes."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import nndeploy.base
import nndeploy.dag
from nndeploy.device import Tensor
from nndeploy.server import Server
from nndeploy.server.config import FieldConfig
from test_db import db
from test_redis_queue import backend, redis_service


def formatted(value, data_format):
    tensor = Tensor.from_numpy(value)
    tensor.set_data_format(nndeploy.base.DataFormat.from_name(data_format))
    return tensor


class ChannelFirst(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('test.server.ChannelFirstFormat')
        self.set_input_type(Tensor)
        self.set_output_type(Tensor)
        self.buffer = None

    def run(self):
        value = self.get_input(0).get(self)
        assert value.get_data_format() == nndeploy.base.DataFormat.NHWC
        self.buffer = np.ascontiguousarray(np.asarray(value).transpose(0, 3, 1, 2))
        self.get_output(0).set(formatted(self.buffer, 'NCHW'))
        return nndeploy.base.Status.ok()


def build_graph():
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('formats', [source], [target])
    graph.add_node(ChannelFirst('channels', [source], [target]))
    return graph


class ChannelFirstCreator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = ChannelFirst(name, inputs, outputs)
        return self.node


channel_first_creator = ChannelFirstCreator()
nndeploy.dag.register_node('test.server.ChannelFirstFormat', channel_first_creator)


def config():
    field = {'type': 'tensor', 'dtype': 'float32', 'shape': [1, 2, 3, 2], 'data_format': 'NHWC'}
    return {'inputs': {'value': dict(field, edge='input')}, 'outputs': {'value': dict(field, edge='output')}}


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_graph_formats_transform_input_and_output_with_durable_values(db, tmp_path, isolation):
    settings = config()
    settings['execution'] = {'isolation': isolation}
    server = Server(name='formats', graph_factory=build_graph, config=settings, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    source = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    body = {'dtype': 'float32', 'shape': list(source.shape), 'data_format': 'NCHW', 'data': source.ravel().tolist()}
    expected = {'dtype': 'float32', 'shape': [1, 2, 3, 2], 'data_format': 'NHWC',
                'data': source.transpose(0, 2, 3, 1).ravel().tolist()}
    with TestClient(server.app) as client:
        result = client.post('/v1/server/formats/deploy', json={'inputs': {'value': body}})
        assert result.status_code == 200, result.text
        assert result.json()['outputs']['value'] == expected
        task_id = result.json()['task_id']
        # Omitting the request format uses the schema's NHWC convention.
        second = dict(expected, data=[0] * 12)
        second.pop('data_format')
        result = client.post('/v1/server/formats/deploy', json={'inputs': {'value': second}})
        assert result.status_code == 200, result.text
        assert result.json()['outputs']['value']['data'] == [0] * 12
        assert client.get(f'/v1/tasks/{task_id}').json()['outputs']['value'] == expected
        assert client.get('/v1/server/formats/schema').json()['schema']['inputs']['value']['data_format'] == 'NHWC'


@pytest.mark.parametrize('target', ['NCDHW', 'NDHWC', 'NDCHW'])
def test_five_dimensional_formats_preserve_axis_values(target):
    from nndeploy.server.arrays import decode, encode
    source = np.arange(48, dtype=np.float32).reshape(1, 2, 3, 2, 4)
    axes = tuple('NCDHW'.index(axis) for axis in target)
    expected = source.transpose(axes)
    field = FieldConfig(type='tensor', dtype='float32', shape=list(expected.shape), data_format=target)
    wire = {'dtype': 'float32', 'shape': list(source.shape), 'data': source.ravel().tolist(), 'data_format': 'NCDHW'}
    tensor = decode(field, wire)
    assert tensor.get_data_format() == nndeploy.base.DataFormat.from_name(target)
    np.testing.assert_array_equal(np.asarray(tensor), expected)
    value = encode(field, formatted(source, 'NCDHW'))
    assert value['shape'] == list(expected.shape)
    assert value['data_format'] == target
    assert value['data'] == expected.ravel().tolist()


@pytest.mark.parametrize('data_format,shape', [('N', [3]), ('NC', [2, 3]), ('NCL', [1, 2, 3]),
                                            ('S1D', [2, 1, 3]), ('OIHW', [2, 1, 2, 3])])
def test_explicit_formats_keep_native_metadata_and_values(data_format, shape):
    from nndeploy.server.arrays import decode, encode
    field = FieldConfig(type='tensor', dtype='int64', shape=shape, data_format=data_format)
    body = {'dtype': 'int64', 'shape': shape, 'data': list(range(int(np.prod(shape))))}
    tensor = decode(field, body)
    assert tensor.get_data_format() == nndeploy.base.DataFormat.from_name(data_format)
    assert encode(field, tensor) == dict(body, data_format=data_format)


@pytest.mark.parametrize('options', [
    {'type': 'array', 'dtype': 'float32', 'shape': [1, 2, 3, 4], 'data_format': 'NCHW'},
    {'type': 'tensor', 'dtype': 'float32', 'shape': [1, 2, 3], 'data_format': 'NCHW'},
    {'type': 'tensor', 'dtype': 'float32', 'shape': [1, 2, 3, 4], 'data_format': 'NC4HW'},
    {'type': 'tensor', 'dtype': 'float32', 'shape': [1, 2, 3, 4], 'data_format': 'Auto'},
    {'type': 'tensor', 'dtype': 'float32', 'shape': [None, None, 3], 'data_format': 'S1D'},
    {'type': 'string', 'data_format': 'N'},
])
def test_invalid_format_declarations_fail_before_startup(options):
    with pytest.raises(ValueError):
        FieldConfig.model_validate(options)


@pytest.mark.parametrize('change', [
    {'data_format': 'OIHW'}, {'data_format': 'NC4HW'}, {'data_format': None}, {'data_format': []},
    {'shape': [1, 2, 3]}, {'shape': [True, 2, 2, 3]}, {'shape': [1, 2, 2, 3_000_000]},
    {'data': [0] * 11}, {'data': [float('inf')] * 12},
])
def test_invalid_formatted_input_is_rejected_before_buffer_allocation(monkeypatch, change):
    from nndeploy.server.arrays import decode
    field = FieldConfig(type='tensor', dtype='float32', shape=[1, 2, 3, 2], data_format='NHWC', max_bytes=48)
    body = dict(dtype='float32', shape=[1, 2, 2, 3], data_format='NCHW', data=[0] * 12)
    body.update(change)
    def allocate(*args, **kwargs):
        pytest.fail('Invalid metadata reached NumPy allocation')
    monkeypatch.setattr(np, 'array', allocate)
    with pytest.raises(ValueError):
        decode(field, body)


def test_format_free_fields_preserve_saved_schema_and_reject_wire_override():
    field = FieldConfig(type='tensor', dtype='float32', shape=[1, 2, 3, 4])
    assert 'data_format' not in field.model_dump()
    with pytest.raises(ValueError):
        field.validate_value({'dtype': 'float32', 'shape': [1, 2, 3, 4], 'data': [0] * 24, 'data_format': 'NHWC'})


def test_declared_output_format_converts_wire_results_and_rejects_wrong_native_axes():
    from nndeploy.server.arrays import encode
    field = FieldConfig(type='tensor', dtype='float32', shape=[1, 2, 3, 2], data_format='NHWC')
    source = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    wire = {'dtype': 'float32', 'shape': list(source.shape), 'data': source.ravel().tolist(), 'data_format': 'NCHW'}
    output = encode(field, wire)
    assert output['data'] == source.transpose(0, 2, 3, 1).ravel().tolist()
    assert output['shape'] == [1, 2, 3, 2] and output['data_format'] == 'NHWC'
    with pytest.raises(ValueError):
        encode(field, formatted(source, 'OIHW'))


def test_empty_tensor_permutation_preserves_zero_dimension():
    from nndeploy.server.arrays import decode, encode
    field = FieldConfig(type='tensor', dtype='float32', shape=[1, 0, 3, 2], data_format='NHWC')
    tensor = decode(field, {'dtype': 'float32', 'shape': [1, 2, 0, 3], 'data': [], 'data_format': 'NCHW'})
    assert encode(field, tensor) == {'dtype': 'float32', 'shape': [1, 0, 3, 2], 'data': [], 'data_format': 'NHWC'}


def code_output(values):
    tensor = values['value']
    assert tensor.get_data_format() == nndeploy.base.DataFormat.NHWC
    data = np.ascontiguousarray(np.asarray(tensor).transpose(0, 3, 1, 2))
    return {'value': formatted(data, 'NCHW')}


def test_shared_process_code_formats_and_warmup(backend, tmp_path):
    database, redis_config = backend
    settings = config()
    for group in ('inputs', 'outputs'):
        settings[group]['value'].pop('edge')
    body = {'dtype': 'float32', 'shape': [1, 2, 2, 3], 'data': list(range(12)), 'data_format': 'NCHW'}
    settings['execution'] = {'isolation': 'process', 'warmup': [{'inputs': {'value': body}}]}
    server = Server(name='formats', deploy=code_output, config=settings, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/formats/deploy', json={'inputs': {'value': body}})
        assert response.status_code == 200, response.text
        value = response.json()['outputs']['value']
        assert value['data_format'] == 'NHWC'
        assert value['data'] == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]


async def format_stream(values):
    yield code_output(values)
    value = values['value']
    np.asarray(value)[:] = 0
    yield code_output(values)


@pytest.mark.parametrize('scheduling', ['serial', 'engine'])
def test_format_conversion_in_streams_detaches_each_event(tmp_path, scheduling):
    from test_streaming import parse_events
    settings = config()
    settings.update(streaming={'enabled': True}, execution={'scheduling': scheduling})
    server = Server(name='formats', deploy=format_stream, config=settings, data_dir=tmp_path)
    body = {'dtype': 'float32', 'shape': [1, 2, 2, 3], 'data': list(range(12)), 'data_format': 'NCHW'}
    with TestClient(server.app) as client:
        response = client.post('/v1/server/formats/deploy', json={'mode': 'stream', 'inputs': {'value': body}})
        assert response.status_code == 200, response.text
        events = parse_events(response.text)
        deltas = [event['data']['data']['value'] for event in events if event['kind'] == 'delta']
        assert len(deltas) == 2, response.text
        assert deltas[0]['data'] == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        assert deltas[1]['data'] == [0] * 12
        assert all(value['data_format'] == 'NHWC' and value['shape'] == [1, 2, 3, 2] for value in deltas)
        assert events[-1]['data']['state'] == 'succeeded'


def test_bad_format_is_rejected_at_http_admission(tmp_path):
    calls = []
    def deploy(values):
        calls.append(True)
        return code_output(values)
    with TestClient(Server(name='formats', deploy=deploy, config=config(), data_dir=tmp_path).app) as client:
        body = {'dtype': 'float32', 'shape': [1, 2, 2, 3], 'data': list(range(12)), 'data_format': 'OIHW'}
        result = client.post('/v1/server/formats/deploy', json={'inputs': {'value': body}})
        assert result.status_code == 422, result.text
        assert not calls


def test_target_device_transfer_receives_the_declared_format(monkeypatch):
    from nndeploy.server.arrays import decode
    actual = Tensor.to
    formats = []
    def move(value, device):
        formats.append(value.get_data_format())
        return actual(value, device)
    monkeypatch.setattr(Tensor, 'to', move)
    field = FieldConfig(type='tensor', dtype='float32', shape=[1, 2, 3, 2], data_format='NHWC')
    tensor = decode(field, {'dtype': 'float32', 'shape': [1, 2, 2, 3], 'data': list(range(12)), 'data_format': 'NCHW'})
    assert formats == [nndeploy.base.DataFormat.NHWC]
    assert tensor.get_data_format() == nndeploy.base.DataFormat.NHWC


def test_workflow_json_and_server_sidecar_use_existing_tensor_formats(tmp_path):
    import json
    path = tmp_path / 'workflow.json'
    original = build_graph().serialize()
    path.write_text(original)
    settings = config()
    settings['name'] = 'formats'
    sidecar = path.with_suffix('.server.json')
    saved = json.dumps(settings)
    sidecar.write_text(saved)
    body = {'dtype': 'float32', 'shape': [1, 2, 2, 3], 'data': list(range(12)), 'data_format': 'NCHW'}
    with TestClient(Server(workflow=path, data_dir=tmp_path / 'server').app) as client:
        result = client.post('/v1/server/formats/deploy', json={'inputs': {'value': body}})
        assert result.status_code == 200, result.text
        assert result.json()['outputs']['value']['data'] == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        assert path.read_text() == original
        assert sidecar.read_text() == saved

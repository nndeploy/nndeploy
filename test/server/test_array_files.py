"""Numeric files use the existing upload, Task, and managed-output lifecycle."""
import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from nndeploy.server.arrays import read_npy
from test_arrays import array_graph, tensor_graph, config, graph_config, echo
from test_db import db
from test_storage import s3_service
from test_redis_queue import backend, redis_service
from test_auth import credentials

MIME = 'application/x-npy'


def content(array, version=None):
    data = io.BytesIO()
    np.lib.format.write_array(data, array, version=version, allow_pickle=True)
    return data.getvalue()


def settings(kind='array', dtype='float32', shape=None, graph=False, **execution):
    value = (graph_config if graph else config)(kind, dtype, shape, **execution)
    for group in ('inputs', 'outputs'):
        value[group]['value']['media_type'] = MIME
    return value


def upload(client, array):
    response = client.post('/v1/files', files={'file': ('input.npy', content(array), MIME)})
    assert response.status_code == 201, response.text
    return {'file_id': response.json()['file_id']}


def download(client, reference):
    response = client.get('/v1/files/' + reference['file_id'] + '/content')
    assert response.status_code == 200, response.text
    assert response.headers['content-type'].startswith(MIME)
    return np.load(io.BytesIO(response.content), allow_pickle=False)


@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('kind,graph', [('array', False), ('array', True), ('tensor', True)])
def test_npy_code_and_graph_keep_immutable_task_inputs_and_outputs(db, tmp_path, isolation, kind, graph):
    server = Server(**({'graph_factory': tensor_graph if kind == 'tensor' else array_graph} if graph else {'deploy': echo}),
                    config=settings(kind, 'int64', [None, 2], graph, isolation=isolation), data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False), admin_token='test-admin')
    with TestClient(server.app) as client:
        outputs = []
        for array in [np.array([[2**60, -2**60]], dtype='int64'), np.zeros((3, 2), dtype='int64')]:
            source = upload(client, array)
            response = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
            assert response.status_code == 200, response.text
            saved = server.db.get_task(response.json()['task_id'])
            assert saved['inputs']['value'] == source
            target = response.json()['outputs']['value']
            assert set(target) == {'file_id'} and target != source
            np.testing.assert_array_equal(download(client, target), array)
            outputs.append(target)
            assert client.delete('/v1/files/' + source['file_id'], headers={'Authorization': 'Bearer test-admin'}).status_code == 409
        assert outputs[0] != outputs[1]
        np.testing.assert_array_equal(download(client, outputs[0]), [[2**60, -2**60]])


@pytest.mark.parametrize('array', [np.array(2**63, dtype='uint64'), np.zeros((0, 2, 1), dtype='float16'),
    np.asfortranarray(np.arange(12, dtype='>i4').reshape(3, 4)), np.ones((2, 1, 1, 1, 1, 1, 1, 1), dtype=bool)])
def test_npy_endian_fortran_scalar_empty_and_rank_eight(tmp_path, array):
    server = Server(deploy=echo, config=settings(dtype=array.dtype.name, shape=list(array.shape)), data_dir=tmp_path)
    with TestClient(server.app) as client:
        source = upload(client, array)
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert response.status_code == 200, response.text
        value = download(client, response.json()['outputs']['value'])
        assert value.shape == array.shape and value.dtype.name == array.dtype.name
        assert value.dtype.isnative
        np.testing.assert_array_equal(value, array)


@pytest.mark.parametrize('data', [b'not numpy', content(np.array([object()], dtype=object)),
    content(np.array([1j], dtype='complex64')), content(np.array([(1,)], dtype=[('x', 'i4')])),
    content(np.array([np.nan], dtype='float32')), content(np.ones(2))[:-1], content(np.ones(2)) + b'junk',
    b'\x93NUMPY\x02\x00' + struct.pack('<I', 2**32-1)])
def test_npy_invalid_upload_does_not_create_a_file(tmp_path, data):
    server = Server(deploy=echo, config=settings(), data_dir=tmp_path)
    with TestClient(server.app) as client:
        response = client.post('/v1/files', files={'file': ('input.npy', data, MIME)})
        assert response.status_code == 422, response.text
        assert response.json()['error']['code'] == 'invalid_array'
        assert not list(server.storage.root.glob('*.npy'))


@pytest.mark.parametrize('array', [np.ones(3, dtype='float32'), np.ones(2, dtype='int32'),
                                  np.array([0, 11], dtype='float32')])
def test_npy_schema_rejected_before_task_admission(tmp_path, array):
    schema = settings(shape=[2])
    schema['inputs']['value']['maximum'] = 10
    server = Server(deploy=echo, config=schema, data_dir=tmp_path, admin_token='test-admin')
    with TestClient(server.app) as client:
        source = upload(client, array)
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert response.status_code == 422, response.text
        assert server.inflight == 0
        assert client.delete('/v1/files/' + source['file_id'], headers={'Authorization': 'Bearer test-admin'}).status_code == 200


def test_npy_large_metadata_is_rejected_before_payload_read():
    data = io.BytesIO()
    np.lib.format.write_array_header_2_0(data, {'descr': '<f8', 'fortran_order': False, 'shape': (2**30,)})
    with pytest.raises(ValueError, match='shape'):
        read_npy(io.BytesIO(data.getvalue()))


def test_npy_v2_header_supported():
    expected = np.arange(6, dtype='float16').reshape(2, 3)
    np.testing.assert_array_equal(read_npy(io.BytesIO(content(expected, (2, 0)))), expected)


@pytest.mark.parametrize('dtype,number,maximum', [('uint64', 2**53 + 1, float(2**53)),
                                                ('float16', 65504, 65503.0)])
def test_npy_element_bounds_do_not_round_to_array_dtype(tmp_path, dtype, number, maximum):
    schema = settings(dtype=dtype, shape=[1])
    schema['inputs']['value']['maximum'] = maximum
    with TestClient(Server(deploy=echo, config=schema, data_dir=tmp_path).app) as client:
        source = upload(client, np.array([number], dtype=dtype))
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert response.status_code == 422, response.text


async def array_stream(values):
    yield {'value': values['value']}
    values['value'][:] = 0
    yield {'value': values['value']}


@pytest.mark.parametrize('scheduling', ['serial', 'engine'])
def test_npy_streams_detach_every_event_before_buffer_reuse(tmp_path, scheduling):
    from test_streaming import parse_events
    schema = settings(scheduling=scheduling)
    schema['streaming'] = {'enabled': True}
    server = Server(deploy=array_stream, config=schema, data_dir=tmp_path)
    with TestClient(server.app) as client:
        source = upload(client, np.arange(4, dtype='float32'))
        response = client.post('/v1/server/default/deploy', json={'mode': 'stream', 'inputs': {'value': source}})
        assert response.status_code == 200, response.text
        events = parse_events(response.text)
        assert events[-1]['data']['state'] == 'succeeded', response.text
        outputs = [event['data']['data']['value'] for event in events if event['kind'] == 'delta']
        assert len(outputs) == 2 and outputs[0] != outputs[1]
        np.testing.assert_array_equal(download(client, outputs[0]), np.arange(4))
        np.testing.assert_array_equal(download(client, outputs[1]), np.zeros(4))
        np.testing.assert_array_equal(download(client, source), np.arange(4))


@pytest.mark.parametrize('graph,isolation', [(False, 'thread'), (False, 'process'), (True, 'thread'), (True, 'process')])
def test_npy_warmup_publishes_no_files_or_tasks(tmp_path, graph, isolation):
    from test_warmup import counts
    server = Server(data_dir=tmp_path)
    source = server.storage.save(io.BytesIO(content(np.arange(4, dtype='float32'))), MIME)
    schema = settings(shape=[4], graph=graph, isolation=isolation,
                      warmup=[{'inputs': {'value': {'file_id': source['file_id']}}}])
    server.add(**({'graph_factory': array_graph} if graph else {'deploy': echo}), config=schema)
    with TestClient(server.app) as client:
        assert counts(server) == [0, 1, 0]
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': {'file_id': source['file_id']}}})
        assert response.status_code == 200, response.text
        assert counts(server) == [1, 2, 3]


def test_npy_batch_results_keep_individual_task_ownership(tmp_path):
    from test_arrays import NumericBatch
    schema = settings(batching={'enabled': True, 'max_size': 4, 'max_wait_ms': 50})
    server = Server(handler_factory=NumericBatch, config=schema, data_dir=tmp_path)
    with TestClient(server.app) as client:
        source = upload(client, np.arange(4, dtype='float32'))
        response = client.post('/v1/server/default/deploy/batch', json={'items': [{'inputs': {'value': source}}] * 3})
        assert response.status_code == 200, response.text
        items = response.json()['items']
        assert all(item['status_code'] == 200 for item in items), items
        for item in items:
            ref = item['outputs']['value']
            assert server.db.get_file(ref['file_id'])['producer_task_id'] == item['task_id']
            np.testing.assert_array_equal(download(client, ref), np.arange(4) + 1)
        assert len({item['outputs']['value']['file_id'] for item in items}) == 3


def test_npy_ownership_checked_before_reading_content(tmp_path, credentials, monkeypatch):
    auth, headers = credentials
    server = Server(name='echo', deploy=echo, config=settings(), data_dir=tmp_path, auth_config=auth)
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('input.npy', content(np.ones(2, dtype='float32')), MIME)},
                               headers=headers['alice'])
        assert uploaded.status_code == 201, uploaded.text
        def forbidden(*args, **kwargs):
            pytest.fail('Foreign file must not be opened')
        monkeypatch.setattr(server.storage, 'open', forbidden)
        rejected = client.post('/v1/server/echo/deploy', json={'inputs': {'value': {'file_id': uploaded.json()['file_id']}}},
                               headers=headers['bob'])
        assert rejected.status_code == 422, rejected.text


def test_npy_shared_process_restores_input_and_output_from_minio(backend, s3_service, tmp_path, monkeypatch):
    import uuid
    database, redis_config = backend
    endpoint, environment, client = s3_service
    monkeypatch.setenv('NNDEPLOY_ARRAY_ACCESS', environment['MINIO_ROOT_USER'])
    monkeypatch.setenv('NNDEPLOY_ARRAY_SECRET', environment['MINIO_ROOT_PASSWORD'])
    storage = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
                   prefix='npy-' + uuid.uuid4().hex, access_key_env='NNDEPLOY_ARRAY_ACCESS', secret_key_env='NNDEPLOY_ARRAY_SECRET')
    schema = settings('tensor', 'float32', [None, 2], True, isolation='process')
    server = Server(graph_factory=tensor_graph, config=schema, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config, storage_config=storage)
    with TestClient(server.app) as http:
        expected = np.arange(8, dtype='float32').reshape(4, 2)
        source = upload(http, expected)
        server.storage.resolve(source['file_id'])[0].unlink()
        response = http.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert response.status_code == 200, response.text
        target = response.json()['outputs']['value']
        server.storage.resolve(target['file_id'])[0].unlink()
        np.testing.assert_array_equal(download(http, target), expected)


async def engine_echo(values):
    return echo(values)


def test_npy_engine_file_writes_allow_other_requests_and_cancellation_joins_io(tmp_path, monkeypatch):
    import asyncio
    import threading
    from test_engine import until
    server = Server(deploy=engine_echo, config=settings(scheduling='engine', max_concurrency_per_worker=2,
                    cooperative_cancel=True), data_dir=tmp_path)
    source = server.storage.save(io.BytesIO(content(np.arange(4, dtype='float32'))), MIME)
    started, release = threading.Event(), threading.Event()
    save = server.storage.save
    blocked = []
    def slow_save(stream, *args, **kwargs):
        if not started.is_set():
            blocked.append(kwargs['task_id'])
            started.set()
            assert release.wait(15)
        return save(stream, *args, **kwargs)
    monkeypatch.setattr(server.storage, 'save', slow_save)
    async def scenario():
        await server.start()
        try:
            task, future = await server.submit('default', {'value': {'file_id': source['file_id']}})
            await until(started.is_set)
            # The engine loop must remain free while the first output's storage
            # write is blocked. The second algorithm call actually completes.
            second = await asyncio.wait_for(server.deploy('default', {'value': {'file_id': source['file_id']}}), 5)
            assert second['task_id'] != task
            cancellation = asyncio.create_task(server.cancel_task(task))
            await until(lambda: server.db.get_task(task)['cancel_requested'])
            await asyncio.sleep(.05)
            assert server.db.get_task(task)['state'] == 'running'
            assert not future.done()
            release.set()
            await cancellation
            with pytest.raises(Exception):
                await future
            assert server.db.get_task(task)['state'] == 'cancelled'
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_npy_tensor_file_uses_declared_layout_and_converts_output(tmp_path):
    schema = settings('tensor', 'float32', [1, 2, 2, 3])
    schema['inputs']['value']['data_format'] = 'NCHW'
    schema['outputs']['value'].update(data_format='NHWC', shape=[1, 2, 3, 2])
    with TestClient(Server(deploy=echo, config=schema, data_dir=tmp_path).app) as client:
        array = np.arange(12, dtype='float32').reshape(1, 2, 2, 3)
        source = upload(client, array)
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert response.status_code == 200, response.text
        np.testing.assert_array_equal(download(client, response.json()['outputs']['value']), array.transpose(0, 2, 3, 1))


@pytest.mark.parametrize('failure', ['validation', 'storage'])
def test_npy_task_failure_releases_partial_output_for_cleanup(tmp_path, monkeypatch, failure):
    from sqlalchemy import select
    from nndeploy.server.db import files
    schema = settings()
    schema['outputs']['second'] = dict(schema['outputs']['value'])
    def wrong(values):
        return {'value': values['value'], 'second': np.array([np.nan], dtype='float32') if failure == 'validation' else values['value']}
    server = Server(deploy=wrong, config=schema, data_dir=tmp_path)
    original = server.storage.save
    published = []
    def save(stream, *args, **kwargs):
        if failure == 'storage' and kwargs.get('task_id') and published:
            raise RuntimeError('Second output storage unavailable')
        result = original(stream, *args, **kwargs)
        if kwargs.get('task_id'):
            published.append(result['file_id'])
        return result
    monkeypatch.setattr(server.storage, 'save', save)
    with TestClient(server.app) as client:
        source = upload(client, np.ones(2, dtype='float32'))
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert response.status_code == 500, response.text
        assert len(published) == (1 if failure == 'storage' else 0)
        server.storage.cleanup()
        with server.db.engine.connect() as connection:
            assert list(connection.execute(select(files.c.file_id)).scalars()) == [source['file_id']]
        np.testing.assert_array_equal(download(client, source), np.ones(2))


def test_npy_multi_megabyte_file_avoids_json_expansion_and_idempotent_replay(db, tmp_path):
    import uuid
    schema = settings('array', 'float64', [None])
    for group in ('inputs', 'outputs'):
        schema[group]['value'].update(max_elements=1048576, max_bytes=8 * 1024**2)
    server = Server(deploy=echo, config=schema, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        array = np.arange(1048576, dtype='float64') / 3
        source = upload(client, array)
        body = {'inputs': {'value': source}}
        headers = {'Idempotency-Key': uuid.uuid4().hex}
        first = client.post('/v1/server/default/deploy', json=body, headers=headers)
        assert first.status_code == 200 and len(first.content) < 512, first.text
        repeated = client.post('/v1/server/default/deploy', json=body, headers=headers)
        assert repeated.json() == first.json()
        np.testing.assert_array_equal(download(client, first.json()['outputs']['value']), array)


class ReusingArrayEngine:
    def __init__(self):
        self.buffer = np.zeros(4, dtype='float32')

    async def deploy(self, values):
        np.copyto(self.buffer, values['value'])
        return {'value': self.buffer}


def test_npy_engine_captures_output_before_yielding_to_storage_thread(tmp_path, monkeypatch):
    import asyncio
    import threading
    from test_engine import until
    server = Server(handler_factory=ReusingArrayEngine,
                    config=settings(scheduling='engine', max_concurrency_per_worker=2), data_dir=tmp_path)
    sources = [server.storage.save(io.BytesIO(content(value)), MIME) for value in
               (np.arange(4, dtype='float32'), np.zeros(4, dtype='float32'))]
    started, release = threading.Event(), threading.Event()
    original = asyncio.to_thread
    async def delayed(operation, *args, **kwargs):
        if operation.__name__ in ('validate_result', 'encode_output', '_save_array_files') and not started.is_set():
            def gated():
                started.set()
                assert release.wait(15)
                return operation(*args, **kwargs)
            return await original(gated)
        return await original(operation, *args, **kwargs)
    monkeypatch.setattr(asyncio, 'to_thread', delayed)
    async def scenario():
        await server.start()
        try:
            first, future = await server.submit('default', {'value': {'file_id': sources[0]['file_id']}})
            await until(started.is_set)
            await asyncio.wait_for(server.deploy('default', {'value': {'file_id': sources[1]['file_id']}}), 5)
            release.set()
            result = await future
            with server.storage.open(result['outputs']['value']['file_id']) as stream:
                np.testing.assert_array_equal(np.load(stream, allow_pickle=False), np.arange(4))
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('file_group', ['inputs', 'outputs'])
def test_npy_workflow_json_sidecar_and_inline_fields_can_mix(tmp_path, file_group):
    import json
    from test_tensor_formats import build_graph, config as tensor_config
    workflow = tmp_path / 'workflow.json'
    original = build_graph().serialize()
    workflow.write_text(original)
    schema = tensor_config()
    schema['name'] = 'formats'
    schema[file_group]['value']['media_type'] = MIME
    sidecar = workflow.with_suffix('.server.json')
    saved = json.dumps(schema)
    sidecar.write_text(saved)
    with TestClient(Server(workflow=workflow, data_dir=tmp_path / 'server').app) as client:
        array = np.arange(12, dtype='float32').reshape(1, 2, 3, 2)
        body = upload(client, array) if file_group == 'inputs' else {
            'dtype': 'float32', 'shape': list(array.shape), 'data_format': 'NHWC', 'data': array.ravel().tolist()}
        response = client.post('/v1/server/formats/deploy', json={'inputs': {'value': body}})
        assert response.status_code == 200, response.text
        output = response.json()['outputs']['value']
        if file_group == 'outputs':
            np.testing.assert_array_equal(download(client, output), array)
        else:
            assert output == {'dtype': 'float32', 'shape': list(array.shape), 'data_format': 'NHWC', 'data': array.ravel().tolist()}
        assert workflow.read_text() == original
        assert sidecar.read_text() == saved


@pytest.mark.parametrize('scheduling', ['engine', 'concurrent', 'batch'])
def test_npy_code_warmup_other_schedulers_publishes_no_files(tmp_path, scheduling):
    from test_warmup import counts
    from test_arrays import NumericBatch
    server = Server(data_dir=tmp_path)
    source = server.storage.save(io.BytesIO(content(np.arange(4, dtype='float32'))), MIME)
    execution = {'batching': {'enabled': True}} if scheduling == 'batch' else {'scheduling': scheduling, 'max_concurrency_per_worker': 2}
    execution['warmup'] = [{'inputs': {'value': {'file_id': source['file_id']}}}]
    schema = settings(**execution)
    options = {'handler_factory': NumericBatch} if scheduling == 'batch' else {'deploy': engine_echo if scheduling == 'engine' else echo}
    server.add(config=schema, **options)
    with TestClient(server.app) as client:
        assert counts(server) == [0, 1, 0]
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': {'file_id': source['file_id']}}})
        assert response.status_code == 200, response.text
        np.testing.assert_array_equal(download(client, response.json()['outputs']['value']),
                                      np.arange(4) + (1 if scheduling == 'batch' else 0))

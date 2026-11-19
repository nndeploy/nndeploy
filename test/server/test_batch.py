"""Batch requests remain bounded collections of independently owned Tasks."""
import asyncio
import io
import threading
import uuid

from fastapi.testclient import TestClient
import pytest
from pydantic import BaseModel
from PIL import Image

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_auth import credentials
from test_db import db
from test_engine import build_engine, engine_config, until


class BatchInputs(BaseModel):
    text: str


def deploy_item(values):
    if values['text'] == 'fail':
        raise ServerError(503, 'algorithm_unavailable', 'Controlled failure', retryable=True)
    return {'text': values['text']}


def test_batch_preserves_order_errors_ownership_and_per_item_idempotency(db, tmp_path, credentials):
    auth, headers = credentials
    server = Server(name='echo', deploy=deploy_item, input_model=BatchInputs, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False), auth_config=auth)
    prefix = uuid.uuid4().hex
    items = [{'inputs': {'text': value}, 'idempotency_key': prefix + str(index)} for index, value in enumerate(('first', 'fail', 'last'))]
    items.insert(1, {'inputs': {'text': 123}})
    with TestClient(server.app) as client:
        assert client.post('/v1/server/echo/deploy/batch', json={'items': items}).status_code == 401
        assert client.post('/v1/server/echo/deploy/batch', json={'items': items}, headers=headers['reader']).status_code == 403
        response = client.post('/v1/server/echo/deploy/batch', json={'items': items}, headers=headers['alice'])
        assert response.status_code == 200, response.text
        results = response.json()['items']
        assert [item['index'] for item in results] == [0, 1, 2, 3]
        assert [item['status_code'] for item in results] == [200, 422, 503, 200]
        assert [item['accepted'] for item in results] == [True, False, True, True]
        assert results[2]['error']['retryable'] is True
        assert results[0]['outputs'] == {'text': 'first'} and results[3]['outputs'] == {'text': 'last'}
        assert all(db.get_task(item['task_id'])['owner'] == 'alice' for item in results if item['accepted'])
        retried = client.post('/v1/server/echo/deploy/batch', json={'items': items}, headers=headers['alice']).json()['items']
        assert [item.get('task_id') for item in retried] == [item.get('task_id') for item in results]
        assert [item['status_code'] for item in retried] == [200, 422, 503, 200]
        assert retried[2]['error'] == results[2]['error']
        bob = client.post('/v1/server/echo/deploy/batch', json={'items': [items[0]]}, headers=headers['bob']).json()['items'][0]
        assert bob['task_id'] != results[0]['task_id']
        assert client.get('/v1/tasks/' + results[0]['task_id'], headers=headers['bob']).status_code == 404
        for bad in ({'items': []}, {'items': [{}] * 65}, {'items': [{}], 'mode': 'stream'}, {'items': [{'unknown': 1}]}):
            assert client.post('/v1/server/echo/deploy/batch', json=bad, headers=headers['alice']).status_code == 422
        assert client.post('/v1/server/echo/deploy/batch', json={'items': [{}]},
            headers=dict(headers['alice'], **{'Idempotency-Key': 'unsupported'})).status_code == 422


def test_async_batch_capacity_and_cancellation_use_existing_tasks(tmp_path):
    server = Server(name='engine', handler_factory=build_engine, resources=tmp_path, data_dir=tmp_path, config=engine_config(cooperative=True))
    release = tmp_path / 'release'
    async def scenario():
        await server.start()
        try:
            result = await server.deploy_batch('engine', [{'inputs': {'release': str(release)}} for _ in range(4)], mode='async')
            assert [item['status_code'] for item in result['items']] == [202, 202, 429, 429]
            accepted = result['items'][:2]
            await until(lambda: all(server.db.get_task(item['task_id'])['state'] == 'running' for item in accepted))
            await server.cancel_task(accepted[0]['task_id'])
            await until(lambda: server.db.get_task(accepted[0]['task_id'])['state'] == 'cancelled')
            assert server.db.get_task(accepted[1]['task_id'])['state'] == 'running'
            assert (await server.deploy('engine', {}))['outputs']['text'] == 'echo'
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_batch_stops_admission_after_database_failure_without_executing(tmp_path, monkeypatch):
    calls = []
    server = Server(name='echo', deploy=lambda value: calls.append(value), resources=tmp_path, data_dir=tmp_path)
    writes = []
    def fail(*args, **kwargs):
        writes.append(1)
        raise RuntimeError('DB unavailable')
    monkeypatch.setattr(server.db, 'create_task', fail)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/deploy/batch', json={'items': [{}] * 4})
        assert [item['status_code'] for item in response.json()['items']] == [503] * 4
        assert not calls and len(writes) == 1


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_batch_runs_native_image_graph_with_independent_outputs(tmp_path, isolation):
    from test_server_api import image_graph
    from nndeploy.server.config import suggest_config
    graph = image_graph()
    config = suggest_config(graph)
    config['execution'] = {'isolation': isolation}
    server = Server(workflow=graph, config=config, resources=tmp_path, data_dir=tmp_path)
    colors = [(11, 22, 33), (44, 55, 66)]
    with TestClient(server.app) as client:
        items = []
        for color in colors:
            picture = io.BytesIO()
            Image.new('RGB', (3, 4), color).save(picture, format='PNG')
            uploaded = client.post('/v1/files', files={'file': ('image.png', picture.getvalue(), 'image/png')})
            assert uploaded.status_code == 201
            items.append({'inputs': {'decode': {'file_id': uploaded.json()['file_id']}}})
        response = client.post('/v1/server/image/deploy/batch', json={'items': items})
        assert response.status_code == 200, response.text
        results = response.json()['items']
        identities = [item['outputs']['encode']['file_id'] for item in results]
        assert len(set(identities)) == 2
        for identity, color in zip(identities, colors):
            content = client.get(f'/v1/files/{identity}/content')
            picture = Image.open(io.BytesIO(content.content))
            assert picture.size == (3, 4) and picture.getpixel((0, 0)) == color


def test_batch_does_not_silently_mix_entries_during_reload(tmp_path, monkeypatch):
    versions = []
    class Handler:
        def __init__(self):
            self.version = len(versions) + 1
            versions.append(self.version)
        def deploy(self, values):
            return {'version': self.version}
    server = Server(name='code', handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    entered, release = threading.Event(), threading.Event()
    create = server.db.create_task
    def delayed(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError('Test DB write was not released')
        return create(*args, **kwargs)
    monkeypatch.setattr(server.db, 'create_task', delayed)
    async def scenario():
        await server.start()
        try:
            old = server.entries['code']
            batch = asyncio.create_task(server.deploy_batch('code', [{}] * 3))
            assert await asyncio.to_thread(entered.wait, 5)
            reloading = asyncio.create_task(server.reload_entry('code'))
            await until(lambda: server.entries['code'] is not old)
            release.set()
            results = (await batch)['items']
            assert [item['status_code'] for item in results] == [200, 409, 409]
            assert results[0]['outputs'] == {'version': 1}
            assert all(item['error']['code'] == 'entry_changed' for item in results[1:])
            await reloading
            assert (await server.deploy('code', {}))['outputs'] == {'version': 2}
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_cancelled_python_waiter_does_not_break_subsequent_batch_execution(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def deploy(values):
        entered.set()
        if not release.wait(10):
            raise RuntimeError('Test computation was not released')
        return values
    server = Server(name='echo', deploy=deploy, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        try:
            identity, observer = await server.submit('echo', {'text': 'first'})
            assert await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(observer, .01)
            assert observer.cancelled() and not server.db.get_task(identity)['cancel_requested']
            assert server.db.get_task(identity)['state'] == 'running'
            batch = asyncio.create_task(server.deploy_batch('echo', [{'inputs': {'text': 'next'}}]))
            release.set()
            result = await asyncio.wait_for(batch, 5)
            assert result['items'][0]['outputs'] == {'text': 'next'}
            assert server.db.get_task(identity)['outputs'] == {'text': 'first'}
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_node_shorthand_batch_rejects_unknown_envelope_before_admission(tmp_path):
    calls = []

    def echo(inputs):
        calls.append(inputs)
        return inputs

    server = Server(name='echo', deploy=echo, resources=tmp_path, data_dir=tmp_path,
                    config={'inputs': {'text': {'type': 'string'}}})
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/deploy/batch',
                               json={'items': [{'text': 'valid'}, {'text': 'invalid', 'unknown': 1}]})
        assert response.status_code == 422
        assert calls == []
        response = client.post('/v1/server/echo/deploy/batch', json={'items': [{'text': 'valid'}]})
        assert response.status_code == 200
        assert response.json()['items'][0]['outputs'] == {'text': 'valid'}
        assert calls == [{'text': 'valid'}]

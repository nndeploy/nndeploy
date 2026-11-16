"""Algorithm-owned protocols use the normal Server execution and ownership path."""
import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nndeploy.server import Server
from nndeploy.server.dependencies import get_deploy, get_entry, get_db, get_storage, get_redis
from nndeploy.server.protocol import DeployRequest
from test_auth import credentials
from test_db import db
from test_redis_queue import redis_service, backend


class Message(BaseModel):
    text: str
    mode: Literal['sync', 'async', 'stream'] = 'sync'


def algorithm_router():
    router = APIRouter()

    @router.post('/messages')
    async def messages(body: Message, deploy=Depends(get_deploy)):
        result = await deploy(DeployRequest(inputs={'text': body.text}, mode=body.mode))
        if isinstance(result, dict):
            return {'id': result['task_id'], 'text': result['outputs']['text']}
        return result

    @router.get('/info')
    async def info(entry=Depends(get_entry), db=Depends(get_db), storage=Depends(get_storage)):
        return {'name': entry['name'], 'db': db is storage.db}

    @router.get('/redis')
    async def redis(client=Depends(get_redis)):
        return {'ready': await client.ping()}

    return router


def echo(values):
    return {'text': values['text']}


def test_custom_protocol_shares_ownership_idempotency_and_task_history(db, tmp_path, credentials):
    auth, headers = credentials
    router = algorithm_router()
    server = Server(name='echo', deploy=echo, router=router, auth_config=auth, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    server.add(name='private', deploy=echo, router=router)
    with TestClient(server.app) as client:
        url = '/v1/server/echo/messages'
        assert client.post(url, json={'text': 'hello'}).status_code == 401
        assert client.post(url, json={'text': 'hello'}, headers=headers['reader']).status_code == 403
        assert client.post('/v1/server/private/messages', json={'text': 'hello'}, headers=headers['alice']).status_code == 403
        owned = dict(headers['alice'], **{'Idempotency-Key': 'one'})
        result = client.post(url, json={'text': 'hello'}, headers=owned)
        assert result.status_code == 200, result.text
        value = result.json()
        assert value['text'] == 'hello'
        assert client.post(url, json={'text': 'hello'}, headers=owned).json() == value
        # A standard route retry identifies the exact same accepted task.
        standard = client.post('/v1/server/echo/deploy', json={'inputs': {'text': 'hello'}}, headers=owned)
        assert standard.json()['task_id'] == value['id']
        assert client.post(url, json={'text': 'changed'}, headers=owned).status_code == 409
        assert db.get_task(value['id'])['owner'] == 'alice'
        assert client.get('/v1/tasks/' + value['id'], headers=headers['bob']).status_code == 404
        assert client.get('/v1/server/echo/info', headers=headers['alice']).json() == {'name': 'echo', 'db': True}
        assert client.get('/v1/server/echo/redis', headers=headers['alice']).status_code == 503
        pending = client.post(url, json={'text': 'later', 'mode': 'async'}, headers=headers['alice'])
        assert pending.status_code == 202, pending.text
        assert pending.json()['status_url'].startswith('/v1/tasks/')
        assert client.post(url, json={'text': 'x', 'mode': 'stream'}, headers=headers['alice']).status_code == 422
        paths = client.get('/openapi.json').json()['paths']
        assert '/v1/server/private/messages' in paths
        assert '/v1/server/echo/messages' in paths


def test_custom_route_and_standard_route_share_bounded_queue_and_stop(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow(values):
        entered.set()
        assert release.wait(10)
        return echo(values)

    server = Server(name='echo', deploy=slow, router=algorithm_router(), data_dir=tmp_path,
                    admin_token='admin', config={'inputs': {'text': {'type': 'string'}}, 'execution': {'max_queue_size': 0}})
    with TestClient(server.app) as client, ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, '/v1/server/echo/messages', json={'text': 'one'})
        try:
            assert entered.wait(5)
            result = client.post('/v1/server/echo/deploy', json={'inputs': {'text': 'two'}})
            assert result.status_code == 429, result.text
        finally:
            release.set()
        assert future.result().json()['text'] == 'one'
        assert client.post('/api/server/echo/stop', headers={'Authorization': 'Bearer admin'}).status_code == 200
        assert client.post('/v1/server/echo/messages', json={'text': 'no'}).status_code == 503


def stream(values):
    yield {'text': values['text'][:1]}
    yield {'text': values['text']}


def test_custom_stream_uses_accept_limit_events_and_existing_replay(tmp_path):
    server = Server(name='echo', deploy=stream, router=algorithm_router(), data_dir=tmp_path,
                    config={'inputs': {'text': {'type': 'string'}}, 'streaming': {'enabled': True}})
    with TestClient(server.app) as client:
        url = '/v1/server/echo/messages'
        body = {'text': 'hello', 'mode': 'stream'}
        assert client.post(url, json=body, headers={'Accept': 'application/json'}).status_code == 406
        response = client.post(url, json=body, headers={'Accept': 'text/event-stream'})
        assert response.status_code == 200, response.text
        assert 'event: delta' in response.text and 'event: done' in response.text
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        task_id = events[0]['task_id']
        replay = client.get(f'/v1/tasks/{task_id}/events', headers={'Accept': 'text/event-stream'})
        assert replay.status_code == 200 and 'event: done' in replay.text
        assert server.stream_connections == 0


def test_lazy_registry_is_local_and_missing_optional_dependency_does_not_load(tmp_path, monkeypatch):
    module_name = 'test_algorithm_optional'
    marker = tmp_path / 'imported'
    (tmp_path / f'{module_name}.py').write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).write_text("yes")\n'
        'def create(suffix="!"):\n'
        '    def deploy(value):\n'
        '        return {"text": value["text"] + suffix}\n'
        '    return {"deploy": deploy}\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    server = Server(data_dir=tmp_path / 'server')
    server.register('optional', module_name + ':create')
    server.register('missing', 'nndeploy_test_missing_optional_sdk:create')
    assert not marker.exists() and module_name not in sys.modules
    with pytest.raises(ValueError, match='Cannot load registered algorithm missing'):
        server.add(name='broken', algorithm='missing')
    assert 'broken' not in server.entries
    server.add(name='echo', algorithm='optional', options={'suffix': '?'}, router=algorithm_router())
    assert marker.exists()
    try:
        with TestClient(server.app) as client:
            assert client.post('/v1/server/echo/messages', json={'text': 'hello'}).json()['text'] == 'hello?'
            assert client.post('/v1/server/nndeploy_test_missing_optional_sdk:create/deploy', json={}).status_code == 404
        other = Server(data_dir=tmp_path / 'other')
        with pytest.raises(ValueError, match='Unknown registered algorithm'):
            other.add(algorithm='optional')
        asyncio.run(other.close())
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize('path', ['/deploy', '/deploy/custom', '/schema', '/{name}/messages', '/'])
def test_custom_routes_cannot_shadow_standard_routes(tmp_path, path):
    router = APIRouter()
    router.add_api_route(path, lambda: {})
    server = Server(data_dir=tmp_path)
    before = len(server.app.routes)
    with pytest.raises(ValueError, match='literal first path'):
        server.add(name='echo', deploy=echo, router=router)
    assert not server.entries and len(server.app.routes) == before
    asyncio.run(server.close())


def test_duplicate_registry_bad_factories_and_late_routes_fail_without_mutation(tmp_path):
    server = Server(data_dir=tmp_path)
    server.register('echo', lambda: {'deploy': echo})
    with pytest.raises(ValueError, match='Duplicate'):
        server.register('echo', lambda: {})
    with pytest.raises(ValueError, match='syntax'):
        server.register('bad', 'os.system')
    server.register('invalid', lambda: {'app': 'not supported'})
    with pytest.raises(ValueError, match='must return'):
        server.add(algorithm='invalid')
    with pytest.raises(ValueError, match='either'):
        server.add(algorithm='echo', deploy=echo)
    with pytest.raises(ValueError, match='Options require'):
        server.add(deploy=echo, options={})
    with TestClient(server.app):
        with pytest.raises(ValueError, match='before starting'):
            server.add(name='late', deploy=echo, router=algorithm_router())
        assert 'late' not in server.entries


def shared_algorithm():
    return {'deploy': echo, 'router': algorithm_router()}


def test_registered_algorithm_uses_real_redis_and_shared_task_records(backend, tmp_path):
    db, redis_config = backend
    server = Server(data_dir=tmp_path, redis_config=redis_config, database_url=db.url.render_as_string(hide_password=False))
    server.register('text', shared_algorithm).add(name='echo', algorithm='text')
    with TestClient(server.app) as client:
        assert client.get('/v1/server/echo/redis').json() == {'ready': True}
        response = client.post('/v1/server/echo/messages', json={'text': 'shared'})
        assert response.status_code == 200, response.text
        result = response.json()
        record = db.get_task(result['id'])
        assert record['state'] == 'succeeded' and record['worker_token']
        assert record['outputs'] == {'text': 'shared'}


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_algorithm_route_can_use_real_graph_and_precedes_editor_catchall(tmp_path, isolation):
    from fastapi.responses import PlainTextResponse
    from test_server_api import text_graph
    server = Server(data_dir=tmp_path)

    async def frontend(scope, receive, send):
        await PlainTextResponse('editor')(scope, receive, send)

    server.app.mount('/', frontend)
    # Generate OpenAPI before adding routes to check schema cache invalidation.
    server.app.openapi()
    server.add(name='echo', workflow=text_graph(), router=algorithm_router(), config={
        'inputs': {'text': {'type': 'string', 'node': ['source']}},
        'outputs': {'text': {'type': 'string', 'edge': 'text'}},
        'execution': {'isolation': isolation}})
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/messages', json={'text': 'graph'})
        assert response.status_code == 200 and response.json()['text'] == 'graph', response.text
        assert client.get('/design').text == 'editor'
        assert '/v1/server/echo/messages' in client.get('/openapi.json').json()['paths']

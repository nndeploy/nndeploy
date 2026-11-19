"""Real MCP JSON-RPC requests crossing the Server's ASGI interface."""
import time
import threading

import pytest
pytest.importorskip('mcp')
from fastapi.testclient import TestClient
from pydantic import BaseModel
from nndeploy.server import Server

HEADERS = {'Accept': 'application/json, text/event-stream', 'MCP-Protocol-Version': '2025-11-25'}
CONFIG = {'enabled': True, 'allowed_hosts': ['testserver']}


def rpc(client, method, params=None, token=None):
    headers = dict(HEADERS)
    if token:
        headers['Authorization'] = 'Bearer ' + token
    response = client.post('/mcp', headers=headers,
        json={'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params or {}})
    assert response.status_code == 200, response.text
    return response.json()['result']


def call(client, name, arguments=None, token=None):
    return rpc(client, 'tools/call', {'name': name, 'arguments': arguments or {}}, token)


def finished(client, identity, token=None):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = call(client, 'get_task', {'task_id': identity}, token)
        assert not result.get('isError'), result
        record = result['structuredContent']
        if record['state'] in ('succeeded', 'failed', 'cancelled', 'expired'):
            return record
        time.sleep(.01)
    pytest.fail('Task did not finish')


def add(server, name='echo', **kwargs):
    return server.add(name=name, deploy=lambda inputs: inputs,
        config={'inputs': {'text': {'type': 'string'}}},
        mcp={'name': name, 'description': 'Echo text'}, **kwargs)


def test_multiple_tools_typed_inputs_idempotency_and_http(tmp_path):
    class Nested(BaseModel):
        number: int
    class Input(BaseModel):
        item: Nested
    server = Server(resources=tmp_path, data_dir=tmp_path, mcp_config=CONFIG)
    add(server)
    server.add(name='double', deploy=lambda values: {'number': values['item']['number'] * 2},
        input_model=Input, mcp={'name': 'double_number', 'description': 'Double a number'})
    server.add(name='private', deploy=lambda inputs: inputs)
    with TestClient(server.app) as client:
        init = rpc(client, 'initialize', {'protocolVersion': '2025-11-25', 'capabilities': {},
                   'clientInfo': {'name': 'test', 'version': '1'}})
        assert init['serverInfo']['name'] == 'nndeploy'
        tools = rpc(client, 'tools/list')['tools']
        assert {tool['name'] for tool in tools} == {'echo', 'double_number', 'get_task', 'cancel_task'}
        arguments = {'inputs': {'text': 'hello'}, 'idempotency_key': 'repeat'}
        first = call(client, 'echo', arguments)['structuredContent']['task_id']
        assert call(client, 'echo', arguments)['structuredContent']['task_id'] == first
        assert finished(client, first)['outputs'] == {'text': 'hello'}
        assert client.get('/v1/tasks/' + first).json()['outputs'] == {'text': 'hello'}
        second = call(client, 'double_number', {'inputs': {'item': {'number': 3}}})
        assert finished(client, second['structuredContent']['task_id'])['outputs'] == {'number': 6}
        assert call(client, 'echo', {'inputs': {'text': 4}})['isError']
        assert call(client, 'double_number', {'inputs': {'item': {'number': '3'}}})['isError']
        assert call(client, 'echo', {'inputs': {'text': 'a'}, 'unknown': 1})['isError']
        assert call(client, 'private')['isError']


def test_identity_discovery_and_task_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv('MCP_ALICE', 'alice-secret' * 4)
    monkeypatch.setenv('MCP_BOB', 'bob-secret' * 4)
    auth = {'keys': [dict(subject=name, token_env='MCP_' + name.upper(), entries=entries,
                         permissions=['deploy', 'tasks:read', 'tasks:cancel'])
                     for name, entries in [('alice', ['echo']), ('bob', ['other'])]]}
    server = Server(resources=tmp_path, data_dir=tmp_path, mcp_config=CONFIG, auth_config=auth)
    add(server)
    add(server, 'other')
    with TestClient(server.app) as client:
        assert client.post('/mcp', headers=HEADERS, json={}).status_code == 401
        assert {t['name'] for t in rpc(client, 'tools/list', token='alice-secret' * 4)['tools']} == {'echo', 'get_task', 'cancel_task'}
        assert {t['name'] for t in rpc(client, 'tools/list', token='bob-secret' * 4)['tools']} == {'other', 'get_task', 'cancel_task'}
        assert call(client, 'echo', {'inputs': {'text': 'x'}}, 'bob-secret' * 4)['isError']
        task = call(client, 'echo', {'inputs': {'text': 'x'}}, 'alice-secret' * 4)['structuredContent']['task_id']
        assert finished(client, task, 'alice-secret' * 4)['state'] == 'succeeded'
        for tool in ('get_task', 'cancel_task'):
            assert call(client, tool, {'task_id': task}, 'bob-secret' * 4)['isError']


def test_pending_cancel_and_backpressure(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def blocked(inputs):
        entered.set()
        assert release.wait(10)
        return inputs
    server = Server(resources=tmp_path, data_dir=tmp_path, mcp_config=CONFIG)
    server.add(name='slow', deploy=blocked, config={'inputs': {}, 'execution': {'max_queue_size': 1}},
               mcp={'name': 'slow', 'description': 'Controlled task'})
    with TestClient(server.app) as client:
        try:
            first = call(client, 'slow')['structuredContent']['task_id']
            assert entered.wait(5)
            second = call(client, 'slow')['structuredContent']['task_id']
            full = call(client, 'slow')
            assert full['isError'] and full['structuredContent']['error']['status_code'] == 429
            cancelled = call(client, 'cancel_task', {'task_id': second})
            assert not cancelled['isError']
            assert finished(client, second)['state'] == 'cancelled'
        finally:
            release.set()
        assert finished(client, first)['state'] == 'succeeded'


def test_configuration_and_optional_loading(tmp_path):
    server = Server(resources=tmp_path / 'off', data_dir=tmp_path / 'off')
    assert not any(route.path == '/mcp' for route in server.app.routes)
    with pytest.raises(ValueError, match='Enable'):
        add(server)
    server = Server(resources=tmp_path / 'on', data_dir=tmp_path / 'on', mcp_config=CONFIG)
    add(server)
    with pytest.raises(ValueError, match='Duplicate MCP'):
        server.add(name='another', deploy=lambda v: v,
            config={}, mcp={'name': 'echo', 'description': 'duplicate'})
    with pytest.raises(ValueError, match='reserved'):
        add(server, 'get_task')
    with pytest.raises(ValueError, match='input_model'):
        server.add(name='untyped', deploy=lambda v: v, mcp={'name': 'untyped', 'description': 'No schema'})
    assert set(server.entries) == {'echo'}
    with TestClient(server.app) as client:
        assert client.post('/mcp', headers={**HEADERS, 'Origin': 'https://evil.example'}, json={}).status_code == 403
        assert client.post('/mcp', headers={**HEADERS, 'Host': 'evil.example'}, json={}).status_code == 421


def test_composite_native_workflow(tmp_path):
    from test_server_api import text_graph
    server = Server(resources=tmp_path, data_dir=tmp_path, mcp_config=CONFIG)
    server.add(name='pipeline', workflow=text_graph(),
        config={'inputs': {'text': {'node': ['source'], 'type': 'string'}},
                'outputs': {'text': {'edge': 'text', 'type': 'string'}}},
        mcp={'name': 'text_pipeline', 'description': 'Run InputStr and ConsolePrint as a single workflow'})
    with TestClient(server.app) as client:
        identity = call(client, 'text_pipeline', {'inputs': {'text': 'workflow'}})['structuredContent']['task_id']
        assert finished(client, identity)['outputs'] == {'text': 'workflow'}


def test_official_sdk_client_and_audit(tmp_path):
    import asyncio
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    server = Server(resources=tmp_path, data_dir=tmp_path, mcp_config=CONFIG, audit_config={})
    add(server)

    async def exercise():
        async with server.app.router.lifespan_context(server.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
                async with streamable_http_client('http://testserver/mcp', http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        assert 'echo' in [tool.name for tool in (await client.list_tools()).tools]
                        result = await client.call_tool('echo', {'inputs': {'text': 'SDK'}})
                        assert not result.isError
                        task = result.structuredContent['task_id']
                        async with asyncio.timeout(10):
                            while True:
                                result = await client.call_tool('get_task', {'task_id': task})
                                assert not result.isError
                                if result.structuredContent['state'] == 'succeeded':
                                    assert result.structuredContent['outputs'] == {'text': 'SDK'}
                                    break
                                await asyncio.sleep(.01)
            from sqlalchemy import select
            from nndeploy.server.audit import records
            with server.db.engine.connect() as connection:
                rows = connection.execute(select(records).where(records.c.route == '/mcp')).mappings().all()
            assert any(row['entry'] == 'echo' and row['resource_id'] == task and row['state'] == 'complete' for row in rows)
    asyncio.run(exercise())


def test_disabled_does_not_import_sdk(tmp_path):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', '''
import sys
from nndeploy.server import Server
server = Server(resources=sys.argv[1], data_dir=sys.argv[1], mcp_config={'enabled': False})
assert 'mcp' not in sys.modules
assert server.mcp is None
''', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_file_schema_and_native_media_roundtrip(tmp_path):
    import io
    from PIL import Image
    from test_server_api import image_graph
    server = Server(resources=tmp_path, data_dir=tmp_path, mcp_config=CONFIG)
    server.add(name='image', workflow=image_graph(),
        mcp={'name': 'process_image', 'description': 'Decode and encode an image'})
    with TestClient(server.app) as client:
        schema = next(tool['inputSchema'] for tool in rpc(client, 'tools/list')['tools'] if tool['name'] == 'process_image')
        assert schema['properties']['inputs']['properties']['decode']['required'] == ['file_id']
        picture = io.BytesIO()
        Image.new('RGB', (5, 7), (20, 40, 60)).save(picture, format='PNG')
        uploaded = client.post('/v1/files', files={'file': ('input.png', picture.getvalue(), 'image/png')})
        identity = call(client, 'process_image', {'inputs': {'decode': {'file_id': uploaded.json()['file_id']}}})['structuredContent']['task_id']
        result = finished(client, identity)
        assert result['state'] == 'succeeded', result
        content = client.get('/v1/files/' + result['outputs']['encode']['file_id'] + '/content')
        assert Image.open(io.BytesIO(content.content)).getpixel((0, 0)) == (20, 40, 60)
        assert call(client, 'process_image', {'inputs': {'decode': {'path': '/tmp/local.png'}}})['isError']


def test_algorithm_registry_and_spa_order(tmp_path):
    from starlette.routing import Mount
    from starlette.responses import PlainTextResponse
    server = Server(resources=tmp_path, data_dir=tmp_path, mcp_config=CONFIG)
    server.register('echo_factory', lambda: dict(deploy=lambda v: v, config={'inputs': {'text': {'type': 'string'}}}))
    server.app.mount('/', PlainTextResponse('editor'))
    server.add(name='registered', algorithm='echo_factory',
        mcp={'name': 'registered_tool', 'description': 'Registered algorithm'})
    routes = server.app.routes
    assert next(i for i, route in enumerate(routes) if route.path == '/mcp') < next(i for i, route in enumerate(routes) if isinstance(route, Mount))
    with TestClient(server.app) as client:
        task = call(client, 'registered_tool', {'inputs': {'text': 'registry'}})['structuredContent']['task_id']
        assert finished(client, task)['outputs'] == {'text': 'registry'}
        assert client.get('/editor').text == 'editor'

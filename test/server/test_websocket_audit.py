"""WebSocket connection audit uses the same durable, metadata-only records."""
import asyncio
import json
import queue
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from nndeploy.server import Server
from nndeploy.server.audit import records
from test_auth import credentials
from test_db import db
from test_duplex import config, echo, wait
from test_audit import make


def connections(server):
    with server.db.engine.connect() as connection:
        return [dict(row) for row in connection.execute(select(records).where(
            records.c.scope == server.audit.scope, records.c.method == 'WEBSOCKET'
        ).order_by(records.c.id)).mappings()]


@pytest.mark.parametrize('method', ['WEBSOCKET', 'PYTHON', 'SYSTEM'])
def test_http_method_cannot_impersonate_an_internal_audit_record(db, tmp_path, method):
    server = make(db, tmp_path)
    with TestClient(server.app) as client:
        response = client.request(method, '/v1/models')
        assert response.status_code == 405
        with db.engine.connect() as connection:
            row = connection.execute(select(records).where(
                records.c.request_id == response.headers['x-request-id'])).mappings().one()
        assert row['method'] == 'OTHER' and row['status_code'] == 405
        assert connections(server) == []


def test_duplex_connection_identity_precedes_accept_and_disconnect_is_durable(db, tmp_path, credentials):
    auth, headers = credentials
    server = make(db, tmp_path, name='echo', deploy=echo, config=config(), auth_config=auth)
    with TestClient(server.app) as client:
        task = client.post('/v1/server/echo/deploy', json={'mode': 'async'}, headers=headers['alice']).json()['task_id']
        # A ready stream means it can observe the Task, not that the independent
        # Worker has already started. Establish the state before disconnecting.
        wait(lambda: server.db.get_task(task)['state'] == 'running')
        with client.websocket_connect(f'/v1/tasks/{task}/stream?private=do-not-record',
                                      headers=dict(headers['alice'], **{'X-Request-ID': 'untrusted-id'})) as ws:
            assert ws.receive_json()['event'] == 'ready'
            row, = connections(server)
            assert row['state'] == 'authorized' and row['subject'] == 'alice'
            assert row['route'] == '/v1/tasks/{task_id}/stream'
            assert row['entry'] == 'echo' and row['resource_id'] == task
            assert row['request_id'].encode() == dict(ws.extra_headers)[b'x-request-id']
            assert 'do-not-record' not in json.dumps(row) and 'untrusted-id' not in json.dumps(row)
        wait(lambda: connections(server)[0]['finished_at'] is not None)
        row, = connections(server)
        assert row['state'] == 'complete' and row['status_code'] == 1000
        assert row['duration_seconds'] >= 0 and row['expires_at'] > row['finished_at']
        assert client.get(f'/v1/tasks/{task}', headers=headers['alice']).json()['state'] == 'running'
        client.post(f'/v1/tasks/{task}/cancel', headers=headers['alice'])


@pytest.mark.parametrize('denied', ['anonymous', 'wrong_owner', 'origin'])
def test_denied_connections_are_audited_without_accepting(db, tmp_path, credentials, denied):
    auth, headers = credentials
    server = make(db, tmp_path, name='echo', deploy=echo, config=config(), auth_config=auth)
    with TestClient(server.app) as client:
        task = client.post('/v1/server/echo/deploy', json={'mode': 'async'}, headers=headers['alice']).json()['task_id']
        request_headers = ({} if denied == 'anonymous' else headers['bob'] if denied == 'wrong_owner'
                           else dict(headers['alice'], Origin='https://untrusted.example'))
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(f'/v1/tasks/{task}/stream', headers=request_headers):
                pytest.fail('Unauthorized connection was accepted')
        assert error.value.code == 1008
        wait(lambda: connections(server) and connections(server)[0]['finished_at'] is not None)
        row, = connections(server)
        assert row['status_code'] == 1008 and row['state'] == 'complete'
        assert row['subject'] == {'anonymous': None, 'wrong_owner': 'bob', 'origin': 'alice'}[denied]
        assert not row['administrator']
        client.post(f'/v1/tasks/{task}/cancel', headers=headers['alice'])


@pytest.mark.parametrize('phase', ['_begin', '_identify'])
def test_audit_failure_refuses_handshake_without_consuming_connection_capacity(db, tmp_path, credentials, monkeypatch, phase):
    auth, headers = credentials
    server = make(db, tmp_path, name='echo', deploy=echo, config=config(), auth_config=auth)
    with TestClient(server.app) as client:
        task = client.post('/v1/server/echo/deploy', json={'mode': 'async'}, headers=headers['alice']).json()['task_id']
        def fail(*args):
            raise RuntimeError('private-database-error')
        with monkeypatch.context() as patch:
            patch.setattr(server.audit, phase, fail)
            with pytest.raises(WebSocketDisconnect) as error:
                with client.websocket_connect(f'/v1/tasks/{task}/stream', headers=headers['alice']):
                    pytest.fail('Connection accepted without durable audit')
            assert error.value.code == 1013
            wait(lambda: server.stream_connections == 0)
        assert 'private-database-error' not in json.dumps(connections(server))
        client.post(f'/v1/tasks/{task}/cancel', headers=headers['alice'])


@pytest.mark.parametrize('code', [1000, 1006])
def test_editor_read_only_connection_records_identity_and_peer_close(tmp_path, monkeypatch, code):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
                    admin_token='audit-admin', audit_config={})
    with TestClient(server.app) as client:
        assert client.post('/api/auth/session', headers={'Authorization': 'Bearer audit-admin'}).status_code == 200
        with client.websocket_connect('/api/ws/progress', headers={'Origin': 'http://testserver'}) as ws:
            row, = connections(server)
            assert row['administrator'] and row['subject'] == '@administrator'
            assert row['state'] == 'authorized' and row['route'] == '/api/ws/progress'
            ws.close(code=code)
        wait(lambda: connections(server)[0]['finished_at'] is not None)
        row, = connections(server)
        assert row['status_code'] == code
        assert row['state'] == ('complete' if code == 1000 else 'interrupted')
        assert not server.sockets and not server.ws_task_map


def test_connection_completion_failure_keeps_unconfirmed_identity_and_task(db, tmp_path, credentials, monkeypatch):
    auth, headers = credentials
    server = make(db, tmp_path, name='echo', deploy=echo, config=config(), auth_config=auth)
    with TestClient(server.app) as client:
        task = client.post('/v1/server/echo/deploy', json={'mode': 'async'}, headers=headers['alice']).json()['task_id']
        wait(lambda: server.db.get_task(task)['state'] == 'running')
        attempted = []
        finish = server.audit._finish
        def fail(request_id, context, status, complete, duration):
            if status == 1000:
                attempted.append(request_id)
                raise RuntimeError('private-final-write-error')
            return finish(request_id, context, status, complete, duration)
        with monkeypatch.context() as patch:
            patch.setattr(server.audit, '_finish', fail)
            with client.websocket_connect(f'/v1/tasks/{task}/stream', headers=headers['alice']) as ws:
                assert ws.receive_json()['event'] == 'ready'
            wait(lambda: attempted and not server.audit.pending)
        row, = connections(server)
        assert row['request_id'] == attempted[0] and row['state'] == 'authorized'
        assert row['finished_at'] is None and row['expires_at'] is None
        assert client.get(f'/v1/tasks/{task}', headers=headers['alice']).json()['state'] == 'running'
        client.post(f'/v1/tasks/{task}/cancel', headers=headers['alice'])


def test_cancelled_connection_joins_final_audit_write_before_server_close(db, tmp_path, credentials):
    """An ASGI disconnect/cancellation must not dispose DB underneath audit."""
    import threading
    from fastapi import Depends, WebSocket
    from nndeploy.server.dependencies import current_identity

    auth, headers = credentials
    server = make(db, tmp_path, auth_config=auth)
    entered, release = threading.Event(), threading.Event()
    finish = server.audit._finish
    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return finish(*args)
    server.audit._finish = blocked

    @server.app.websocket('/v1/observe')
    async def observe(ws: WebSocket, caller=Depends(current_identity)):
        await ws.accept()
        await asyncio.Event().wait()

    async def scenario():
        accepted = asyncio.Event()
        async def receive():
            return {'type': 'websocket.connect'}
        async def send(message):
            if message['type'] == 'websocket.accept':
                accepted.set()
        scope = dict(type='websocket', asgi={'version': '3.0'}, scheme='ws', path='/v1/observe',
                     raw_path=b'/v1/observe', root_path='', query_string=b'',
                     headers=[(b'authorization', headers['alice']['Authorization'].encode())],
                     client=('127.0.0.1', 1234), server=('test', 80), subprotocols=[])
        request = asyncio.create_task(server.app(scope, receive, send))
        closing = None
        try:
            await asyncio.wait_for(accepted.wait(), 5)
            request.cancel()
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(server.close())
            await asyncio.sleep(.05)
            assert not closing.done() and server.audit.pending
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await request
            await asyncio.wait_for(closing, 5)
            row, = connections(server)
            assert row['state'] == 'interrupted' and row['status_code'] == 101
            assert not server.audit.pending and server.audit.closed
        finally:
            release.set()
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            if closing is not None:
                await closing
            await server.close()
    asyncio.run(scenario())

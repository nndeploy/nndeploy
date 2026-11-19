"""One Task receives ordered input while producing durable incremental output."""
import time
import asyncio
import json
import uuid

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect
from sqlalchemy import select

from nndeploy.server import Server
from nndeploy.server.config import StreamingConfig
from nndeploy.server.errors import ServerError
from nndeploy.server.db import task_inputs
from test_db import db
from test_auth import credentials
from test_redis_queue import redis_service, backend


async def echo(inputs, *, input_stream):
    total = inputs.get('prefix', '')
    async for value in input_stream:
        total += value['text']
        yield {'text': total}
    yield {'text': total}


def config(**stream):
    return {'inputs': {'prefix': {'type': 'string', 'required': False, 'default': ''}},
            'streaming': {'enabled': True, 'duplex': True, 'max_duration_seconds': 10, **stream},
            'execution': {'scheduling': 'engine', 'cooperative_cancel': True}}


def wait(check, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('Expected state did not arrive')


@pytest.mark.parametrize('mode', ['engine', 'thread', 'process'])
def test_duplex_http_produces_output_before_end_and_retries_are_idempotent(db, tmp_path, mode):
    settings = config()
    if mode != 'engine':
        settings['execution'] = {'isolation': mode}
    server = Server(name='echo', deploy=echo, config=settings, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        accepted = client.post('/v1/server/echo/deploy', json={'mode': 'async', 'inputs': {'prefix': '>'}})
        assert accepted.status_code == 202, accepted.text
        identity = accepted.json()['task_id']
        path = f'/v1/tasks/{identity}/inputs'
        first = {'sequence': 1, 'data': {'text': 'one'}}
        assert client.post(path, json=first).status_code == 200
        assert client.post(path, json=first).status_code == 200
        events = wait(lambda: [event for event in server.db.read_task_events(identity) if event['kind'] == 'delta'])
        assert events[0]['data']['data'] == {'text': '>one'}
        assert client.get(f'/v1/tasks/{identity}').json()['state'] == 'running'
        assert client.post(path, json={'sequence': 1, 'data': {'text': 'different'}}).status_code == 409
        assert client.post(path, json={'sequence': 3, 'data': {'text': 'gap'}}).status_code == 409
        assert client.post(path, json={'sequence': 2, 'data': {'text': 'two'}}).status_code == 200
        end = {'sequence': 3, 'end': True}
        assert client.post(path, json=end).status_code == 200
        done = wait(lambda: (record if (record := client.get(f'/v1/tasks/{identity}').json())['state'] == 'succeeded' else None))
        assert done['outputs'] == {'text': '>onetwo'}
        assert client.post(path, json=end).status_code == 200
        assert client.post(path, json={'sequence': 4, 'data': 'late'}).status_code == 409


def test_websocket_reconnect_replays_output_and_deduplicates_input(db, tmp_path, credentials):
    auth, headers = credentials
    server = Server(name='echo', deploy=echo, config=config(), resources=tmp_path, data_dir=tmp_path, auth_config=auth,
                    audit_config={}, cors_origins=['https://trusted.example'],
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        identity = client.post('/v1/server/echo/deploy', json={'mode': 'async'}, headers=headers['alice']).json()['task_id']
        path = f'/v1/tasks/{identity}/stream'
        for denied in ({}, headers['bob'], headers['reader'], dict(headers['alice'], Origin='https://untrusted.example')):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(path, headers=denied) as ws:
                    ws.receive_json()
        first = {'sequence': 1, 'data': {'text': 'one'}}
        cursor = 0
        with client.websocket_connect(path, headers=dict(headers['alice'], Origin='https://trusted.example')) as ws:
            assert ws.receive_json()['event'] == 'ready'
            ws.send_json(first)
            ack, delta = False, False
            while not (ack and delta):
                event = ws.receive_json()
                cursor = max(cursor, event.get('sequence', 0))
                ack |= event['event'] == 'input_ack'
                delta |= event['event'] == 'delta'
        assert client.get(f'/v1/tasks/{identity}', headers=headers['alice']).json()['state'] == 'running'
        with client.websocket_connect(path + f'?after={cursor}', headers=headers['alice']) as ws:
            ready = ws.receive_json()
            assert ready['data']['input_sequence'] == 1
            ws.send_json(first)
            assert ws.receive_json()['event'] == 'input_ack'
            ws.send_json({'sequence': 2, 'data': {'text': 'two'}})
            ws.send_json({'sequence': 3, 'end': True})
            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event['event'] == 'done':
                    break
            assert events[-1]['data']['state'] == 'succeeded'
            assert all(item['sequence'] > cursor for item in events if 'sequence' in item)
        assert client.get(f'/v1/tasks/{identity}', headers=headers['alice']).json()['outputs'] == {'text': 'onetwo'}
        wait(lambda: server.stream_connections == 0)
        # Only metadata goes into audit, including mutations over WebSocket.
        from nndeploy.server.audit import records
        with server.db.engine.connect() as connection:
            rows = connection.execute(select(records).where(records.c.resource_id == identity)).mappings().all()
        ws_records = [row for row in rows if row['method'] == 'OTHER']
        assert len(ws_records) == 4
        assert all(row['subject'] == 'alice' and row['state'] == 'complete' for row in ws_records)


def test_input_buffer_backpressure_fencing_and_task_retention(db):
    identity = 'task_' + uuid.uuid4().hex
    limits = StreamingConfig(enabled=True, duplex=True, max_pending_inputs=1, max_events=3, max_event_bytes=128)
    db.create_task(identity, 'echo', time.time(), {}, {}, owner='alice', retention_seconds=60, input_state=limits.input_state())
    first = dict(sequence=1, data={'text': 'one'})
    with pytest.raises(ServerError, match='Task does not exist'):
        db.append_task_input(identity, first, 'bob')
    db.append_task_input(identity, first, 'alice')
    with pytest.raises(ServerError) as error:
        db.append_task_input(identity, dict(sequence=2, data='next'), 'alice')
    assert error.value.code == 'input_queue_full'
    db.start_task(identity, time.time())
    with pytest.raises(ServerError):
        db.read_task_input(identity, 0, worker_token='stale')
    assert db.read_task_input(identity, 0)['data'] == {'text': 'one'}
    assert db.read_task_input(identity, 1) is None
    db.append_task_input(identity, dict(sequence=2, data='next'), 'alice')
    # End can always follow a full buffer; at most one extra end marker exists.
    db.append_task_input(identity, dict(sequence=3, end=True), 'alice')
    with pytest.raises(ServerError):
        db.append_task_input(identity, dict(sequence=4, data='late'), 'alice')
    db.finish_task(identity, 'succeeded', time.time(), outputs={'text': 'done'})
    assert db.append_task_input(identity, first, 'alice')['sequence'] == 1
    from nndeploy.server.db import tasks
    with db.engine.begin() as connection:
        connection.execute(tasks.update().where(tasks.c.task_id == identity).values(purge_after=0))
    db.cleanup_tasks()
    with db.engine.connect() as connection:
        assert connection.execute(select(task_inputs).where(task_inputs.c.task_id == identity)).first() is None


def test_duplex_timeout_and_explicit_cancel_stop_waiting_for_input(tmp_path):
    for cancel in (False, True):
        server = Server(name='echo', deploy=echo, config=config(max_duration_seconds=.2), resources=tmp_path / str(cancel), data_dir=tmp_path / str(cancel))
        with TestClient(server.app) as client:
            identity = client.post('/v1/server/echo/deploy', json={'mode': 'async'}).json()['task_id']
            if cancel:
                assert client.post(f'/v1/tasks/{identity}/cancel').status_code == 200
            result = wait(lambda: (r if (r := client.get(f'/v1/tasks/{identity}').json())['state'] in ('expired', 'cancelled') else None))
            assert result['state'] == ('cancelled' if cancel else 'expired')
            assert client.post(f'/v1/tasks/{identity}/inputs', json={'sequence': 1, 'data': 'late'}).status_code == 409


def test_websocket_limits_invalid_frames_and_disconnect_do_not_cancel_task(tmp_path, monkeypatch):
    from nndeploy.server.api import stream
    monkeypatch.setattr(stream, 'RECEIVE_SECONDS', .1)
    server = Server(name='echo', deploy=echo, config=config(max_event_bytes=128), resources=tmp_path, data_dir=tmp_path, max_inflight_tasks=1)
    with TestClient(server.app) as client:
        identity = client.post('/v1/server/echo/deploy', json={'mode': 'async'}).json()['task_id']
        path = f'/v1/tasks/{identity}/stream'
        with client.websocket_connect(path) as ws:
            assert ws.receive_json()['event'] == 'ready'
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(path) as other:
                    other.receive_json()
            ws.send_json({'sequence': '1', 'data': 'wrong-type'})
            while (error := ws.receive_json())['event'] != 'error':
                pass
            assert error['error']['code'] == 'invalid_input'
            ws.send_json({'sequence': 1, 'data': 'x' * 200})
            while (error := ws.receive_json())['event'] != 'error':
                pass
            assert error['error']['code'] == 'input_too_large'
            with pytest.raises(WebSocketDisconnect):
                while True:
                    ws.receive_json()
        wait(lambda: server.stream_connections == 0)
        assert client.get(f'/v1/tasks/{identity}').json()['state'] == 'running'
        for data, expected in ((b'binary', 1003), ('x' * 5000, 1009)):
            with client.websocket_connect(path) as ws:
                ws.receive_json()
                if isinstance(data, bytes):
                    ws.send_bytes(data)
                else:
                    ws.send_text(data)
                with pytest.raises(WebSocketDisconnect) as failure:
                    while True:
                        ws.receive_json()
                assert failure.value.code == expected
        client.post(f'/v1/tasks/{identity}/cancel')


def test_input_write_races_and_commit_receipt_loss_are_safe(db, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    identity = 'task_' + uuid.uuid4().hex
    db.create_task(identity, 'echo', time.time(), {}, {}, input_state=StreamingConfig(enabled=True, duplex=True).input_state())
    message = dict(sequence=1, data={'text': 'one'})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: db.append_task_input(identity, message), range(16)))
    assert all(result['sequence'] == 1 for result in results)
    with db.engine.connect() as connection:
        assert len(connection.execute(select(task_inputs).where(task_inputs.c.task_id == identity)).all()) == 1
    # Simulate a lost commit receipt, then retry the same message.
    from contextlib import contextmanager
    original = db.engine.begin
    @contextmanager
    def lost_receipt():
        with original() as connection:
            yield connection
        raise OSError('Commit receipt lost')
    with monkeypatch.context() as patch:
        patch.setattr(db.engine, 'begin', lost_receipt)
        with pytest.raises(OSError):
            db.append_task_input(identity, dict(sequence=2, data='two'))
    assert db.append_task_input(identity, dict(sequence=2, data='two'))['sequence'] == 2
    db.request_cancel(identity)
    with pytest.raises(ServerError) as failure:
        db.append_task_input(identity, dict(sequence=3, data='late'))
    assert failure.value.code == 'input_closed'


def test_duplex_file_inputs_follow_task_ownership_and_retention(db, tmp_path):
    import io
    from nndeploy.server.storage.local import LocalStorage
    from nndeploy.server.db import file_refs
    storage = LocalStorage(tmp_path / 'files', db)
    own = storage.save(io.BytesIO(b'input audio bytes'), owner='alice', media_type='application/octet-stream')
    other = storage.save(io.BytesIO(b'other bytes'), owner='bob', media_type='application/octet-stream')
    identity = 'task_' + uuid.uuid4().hex
    db.create_task(identity, 'echo', time.time(), {}, {}, owner='alice',
                   input_state=StreamingConfig(enabled=True, duplex=True).input_state())
    with pytest.raises(ServerError) as failure:
        db.append_task_input(identity, dict(sequence=1, data={'file_id': other['file_id']}), 'alice')
    assert failure.value.code == 'invalid_file'
    db.append_task_input(identity, dict(sequence=1, data={'file_id': own['file_id']}), 'alice')
    with db.engine.connect() as connection:
        refs = connection.execute(select(file_refs.c.file_id).where(file_refs.c.task_id == identity)).scalars().all()
    assert refs == [own['file_id']]
    storage.close()


def test_shared_duplex_inputs_cross_servers_and_run_in_an_isolated_worker(backend, tmp_path):
    import httpx
    database, redis_config = backend
    settings = config()
    settings['execution'] = {'isolation': 'process'}
    servers = [Server(name='echo', deploy=echo, config=settings, resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                      database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config) for _ in range(2)]

    async def scenario():
        held = None
        try:
            for server in servers:
                await server.start()
            # Occupy a real Worker with an open input stream, then submit to
            # that Server so the other process must execute the next Task.
            held, observer = await servers[0].submit('echo', {})
            observer.cancel()
            for _ in range(250):
                busy = next((index for index, server in enumerate(servers) if held in server.running_workers), None)
                if busy is not None:
                    break
                await asyncio.sleep(.02)
            assert busy is not None
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[busy].app), base_url='http://test') as client:
                response = await client.post('/v1/server/echo/deploy', json={'mode': 'async'})
                assert response.status_code == 202, response.text
                identity = response.json()['task_id']
                for sequence, value in enumerate(('one', 'two'), 1):
                    response = await client.post(f'/v1/tasks/{identity}/inputs', json={'sequence': sequence, 'data': {'text': value}})
                    assert response.status_code == 200, response.text
                response = await client.post(f'/v1/tasks/{identity}/inputs', json={'sequence': 3, 'end': True})
                assert response.status_code == 200, response.text
                for _ in range(250):
                    record = await asyncio.to_thread(database.get_task, identity)
                    if record['state'] in ('succeeded', 'failed', 'expired'):
                        break
                    await asyncio.sleep(.02)
                assert record['state'] == 'succeeded', record
                assert record['outputs'] == {'text': 'onetwo'}
                assert identity not in servers[busy].running_workers
        finally:
            if held is not None:
                await servers[0].cancel_task(held)
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_duplex_over_real_http_and_websocket_connections(tmp_path):
    import requests
    from websockets.sync.client import connect
    from test_streaming import live_server
    server = Server(name='echo', deploy=echo, config=config(), resources=tmp_path, data_dir=tmp_path, audit_config={})
    with live_server(server) as address:
        response = requests.post(address + '/v1/server/echo/deploy', json={'mode': 'async'}, timeout=3)
        assert response.status_code == 202, response.text
        identity = response.json()['task_id']
        with connect(address.replace('http://', 'ws://') + f'/v1/tasks/{identity}/stream', proxy=None) as ws:
            request_id = ws.response.headers['X-Request-ID']
            assert json.loads(ws.recv(timeout=3))['event'] == 'ready'
            for sequence, value in enumerate(('first', 'second'), 1):
                ws.send(json.dumps({'sequence': sequence, 'data': {'text': value}}))
                ack, delta = False, False
                while not (ack and delta):
                    event = json.loads(ws.recv(timeout=3))
                    ack |= event['event'] == 'input_ack' and event['data']['sequence'] == sequence
                    delta |= event['event'] == 'delta'
            ws.send(json.dumps({'sequence': 3, 'end': True}))
            while (event := json.loads(ws.recv(timeout=3)))['event'] != 'done':
                pass
            assert event['data']['state'] == 'succeeded'
        assert requests.get(address + f'/v1/tasks/{identity}', timeout=3).json()['outputs'] == {'text': 'firstsecond'}
        from nndeploy.server.audit import records
        def completed():
            with server.db.engine.connect() as connection:
                return connection.execute(select(records).where(records.c.request_id == request_id,
                    records.c.finished_at.is_not(None))).mappings().first()
        row = wait(completed)
        assert row['request_id'] == request_id and row['state'] == 'complete'
        assert row['status_code'] == 1000 and row['resource_id'] == identity


def test_stream_capacity_is_reserved_before_database_lookup(tmp_path, monkeypatch):
    import threading
    from starlette.datastructures import URL
    from nndeploy.server.dependencies import Identity
    entered, release = threading.Event(), threading.Event()
    reads = []
    server = Server(resources=tmp_path, data_dir=tmp_path, max_inflight_tasks=1)
    endpoint = next(route.endpoint for route in server.app.routes if route.path == '/v1/tasks/{task_id}/stream')

    def blocked(identity):
        reads.append(identity)
        entered.set()
        assert release.wait(5)
        return None

    class Socket:
        def __init__(self):
            self.scope = {}
        headers = {}
        url = URL('ws://localhost/v1/tasks/missing/stream')
        closed = False
        async def close(self, code):
            self.closed = True

    monkeypatch.setattr(server.db, 'get_task', blocked)

    async def scenario():
        await server.start()
        first, second = Socket(), Socket()
        caller = Identity(None, frozenset(('deploy', 'tasks:read')), frozenset(('*',)))
        running = asyncio.create_task(endpoint(first, 'first', caller))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            assert server.stream_connections == 1
            await endpoint(second, 'second', caller)
            assert second.closed and reads == ['first']
            release.set()
            await running
            assert first.closed and server.stream_connections == 0
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)
            await server.close()

    asyncio.run(scenario())

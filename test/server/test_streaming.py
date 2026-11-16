"""Incremental output, durable replay, isolation, disconnect and backpressure."""
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import threading
import time

from fastapi.testclient import TestClient
import pytest
import requests
import uvicorn

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.streaming import TaskEventResponse, accepts_events
from test_auth import credentials
from test_db import db


def output_stream(values):
    try:
        yield {'text': 'first\n中文'}
        if values.get('release'):
            deadline = time.monotonic() + 10
            while not Path(values['release']).exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Test stream was not released')
                time.sleep(.01)
        yield {'text': values.get('text', 'last')}
    finally:
        if values.get('closed'):
            Path(values['closed']).touch()


async def async_output_stream(values):
    yield {'text': 'first\n中文'}
    await asyncio.sleep(.01)
    yield {'text': values.get('text', 'last')}


async def waiting_stream(values):
    try:
        yield {'text': 'started'}
        await asyncio.sleep(10)
        yield {'text': 'must not arrive'}
    finally:
        if values.get('closed'):
            Path(values['closed']).touch()


def invalid_stream(values):
    yield {'text': 'accepted'}
    if values.get('oversize'):
        yield {'text': 'x' * 200}
    else:
        for _ in range(3):
            yield {'text': 'too many'}


def parse_events(text):
    result = []
    for block in text.split('\n\n'):
        lines = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line)
        if 'data' in lines:
            result.append(dict(kind=lines['event'], sequence=int(lines['id']) if 'id' in lines else None,
                               data=json.loads(lines['data'])))
    return result


def stream_config(isolation='thread', **limits):
    return {'streaming': dict(enabled=True, **limits), 'execution': {'isolation': isolation},
            'inputs': {name: {'type': 'string', 'required': False, 'default': default}
                       for name, default in [('release', ''), ('closed', ''), ('text', 'last')]}}


@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('handler', [output_stream, async_output_stream])
def test_stream_replay_and_final_state_are_durable(db, tmp_path, credentials, isolation, handler):
    auth_config, headers = credentials
    server = Server(name='echo', deploy=handler, config=stream_config(isolation),
                    data_dir=tmp_path, database_url=db.url.render_as_string(hide_password=False), auth_config=auth_config)
    with TestClient(server.app) as client:
        result = client.post('/v1/server/echo/deploy', json={'mode': 'stream'}, headers=headers['alice'])
        assert result.status_code == 200, result.text
        assert result.headers['content-type'].startswith('text/event-stream')
        events = parse_events(result.text)
        assert [event['kind'] for event in events] == ['start', 'delta', 'delta', 'result', 'done']
        assert [event['sequence'] for event in events] == [1, 2, 3, 4, 5]
        assert events[1]['data']['data'] == {'text': 'first\n中文'}
        identity = result.headers['x-task-id']
        assert events[-1]['data']['state'] == 'succeeded'
        assert db.get_task(identity)['outputs'] == {'text': 'last'}
        assert client.get(f'/v1/tasks/{identity}/events', headers=headers['bob']).status_code == 404
        for invalid in ('-1', 'bad', '6'):
            assert client.get(f'/v1/tasks/{identity}/events', headers=dict(headers['alice'], **{'Last-Event-ID': invalid})).status_code in (409, 422)
        assert client.post('/v1/server/echo/deploy', json={'mode': 'stream'}, headers=dict(headers['alice'], Accept='application/json')).status_code == 406
    with TestClient(Server(data_dir=tmp_path, database_url=db.url.render_as_string(hide_password=False), auth_config=auth_config).app) as client:
        replay = client.get(f'/v1/tasks/{identity}/events', headers=dict(headers['alice'], **{'Last-Event-ID': '2'}))
        assert [event['sequence'] for event in parse_events(replay.text)] == [3, 4, 5]


@contextmanager
def live_server(server):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    address = f'http://127.0.0.1:{listener.getsockname()[1]}'
    http = uvicorn.Server(uvicorn.Config(server.app, log_level='error', lifespan='on'))
    thread = threading.Thread(target=http.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if http.started:
                break
            if not thread.is_alive():
                raise AssertionError('HTTP Server failed to start')
            time.sleep(.02)
        else:
            raise AssertionError('HTTP Server startup timed out')
        yield address
    finally:
        http.should_exit = True
        thread.join(timeout=15)
        listener.close()
        assert not thread.is_alive()


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_real_http_disconnect_does_not_cancel_stream_and_can_resume(tmp_path, isolation):
    release = tmp_path / 'release'
    server = Server(name='echo', deploy=output_stream, data_dir=tmp_path,
                    config=stream_config(isolation))
    with live_server(server) as address:
        try:
            with requests.post(address + '/v1/server/echo/deploy', json={'mode': 'stream', 'inputs': {'release': str(release)}},
                               stream=True, timeout=5) as response:
                assert response.status_code == 200
                identity = response.headers['x-task-id']
                lines = response.iter_lines(chunk_size=1)
                while next(lines) != b'id: 2':
                    pass
                # Actual bytes arrived before releasing the next generator yield.
                assert server.db.get_task(identity)['state'] == 'running'
            assert not server.db.get_task(identity)['cancel_requested']
            release.touch()
            with requests.get(address + f'/v1/tasks/{identity}/events', headers={'Last-Event-ID': '2'}, timeout=5) as replay:
                events = parse_events(replay.text)
                assert [event['sequence'] for event in events] == [3, 4, 5]
                assert events[-1]['data']['state'] == 'succeeded'
        finally:
            release.touch()


@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('oversize', [True, False])
def test_stream_limits_fail_without_more_output_or_reexecution(tmp_path, isolation, oversize):
    config = stream_config(isolation, max_events=2, max_event_bytes=100)
    config['inputs']['oversize'] = {'type': 'boolean'}
    server = Server(name='echo', deploy=invalid_stream, data_dir=tmp_path, config=config)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/deploy', json={'mode': 'stream', 'inputs': {'oversize': oversize}})
        events = parse_events(response.text)
        assert events[-1]['kind'] == 'done' and events[-1]['data']['state'] == 'failed'
        assert len([event for event in events if event['kind'] == 'delta']) == (1 if oversize else 2)
        task = client.get('/v1/tasks/' + response.headers['x-task-id']).json()
        assert task['error']['code'] == ('stream_event_too_large' if oversize else 'stream_event_limit')


@pytest.mark.parametrize('spec_version', ['2.0', '2.4'])
def test_slow_transport_is_bounded_and_keeps_task_ownership(tmp_path, spec_version):
    server = Server(name='echo', deploy=output_stream, data_dir=tmp_path,
                    config=stream_config(), max_inflight_tasks=1)
    release = tmp_path / 'release'
    async def scenario():
        await server.start()
        try:
            identity, future = await server.submit('echo', {'release': str(release)})
            response = TaskEventResponse(server, identity, send_seconds=.01)
            with pytest.raises(ServerError) as full:
                TaskEventResponse(server, identity)
            assert full.value.status_code == 429
            async def blocked_send(message):
                await asyncio.Event().wait()
            async def receive():
                await asyncio.Event().wait()
            await asyncio.wait_for(response({'type': 'http', 'asgi': {'spec_version': spec_version}}, receive, blocked_send), 1)
            assert server.stream_connections == 0
            assert not future.done()
            release.touch()
            assert (await asyncio.wait_for(future, 5))['outputs'] == {'text': 'last'}
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_accept_header_honors_explicit_exclusion_and_wildcards():
    assert accepts_events('text/*')
    assert accepts_events('application/json, text/event-stream;q=0.5')
    assert not accepts_events('text/event-stream;q=0, */*')
    assert not accepts_events('text/event-stream;q=nan')
    assert not accepts_events('application/json')


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_stream_cancel_stops_iteration_and_next_request_remains_usable(tmp_path, isolation):
    release, closed = tmp_path / 'release', tmp_path / 'closed'
    server = Server(name='echo', deploy=output_stream, data_dir=tmp_path, config=stream_config(isolation))
    with live_server(server) as address:
        try:
            response = requests.post(address + '/v1/server/echo/deploy', json={'mode': 'async',
                'inputs': {'release': str(release), 'closed': str(closed)}}, timeout=5)
            identity = response.json()['task_id']
            for _ in range(500):
                if any(event['kind'] == 'delta' for event in server.db.read_task_events(identity)):
                    break
                time.sleep(.01)
            else:
                raise AssertionError('Stream did not emit its first event')
            assert requests.post(address + f'/v1/tasks/{identity}/cancel', timeout=5).status_code == 200
            release.touch()
            events = parse_events(requests.get(address + f'/v1/tasks/{identity}/events', timeout=5).text)
            assert events[-1]['data']['state'] == 'cancelled'
            assert len([event for event in events if event['kind'] == 'delta']) == 1
            if isolation == 'thread':
                assert closed.exists()  # cooperative generator close, not only a status update
            following = requests.post(address + '/v1/server/echo/deploy', json={}, timeout=5)
            assert following.status_code == 200 and following.json()['outputs'] == {'text': 'last'}
        finally:
            release.touch()


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_async_stream_deadline_closes_waiting_iterator(tmp_path, isolation):
    closed = tmp_path / 'closed'
    server = Server(name='echo', deploy=waiting_stream, data_dir=tmp_path,
                    config=stream_config(isolation, max_duration_seconds=.1))
    with TestClient(server.app) as client:
        result = client.post('/v1/server/echo/deploy', json={'mode': 'stream', 'inputs': {'closed': str(closed)}})
        events = parse_events(result.text)
        assert events[-1]['data']['state'] == 'expired'
        assert [event['kind'] for event in events] == ['start', 'delta', 'error', 'done']
        if isolation == 'thread':
            assert closed.exists()


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_event_recording_failure_stops_generator_and_does_not_publish_success(tmp_path, monkeypatch, isolation):
    server = Server(name='echo', deploy=output_stream, data_dir=tmp_path, config=stream_config(isolation))
    append = server.db.append_task_event
    def fail(*args, **kwargs):
        raise RuntimeError('Event database temporarily unavailable')
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.db, 'append_task_event', fail)
        result = client.post('/v1/server/echo/deploy', json={'mode': 'stream'})
        events = parse_events(result.text)
        assert [event['kind'] for event in events] == ['start', 'error', 'done']
        assert events[-1]['data']['state'] == 'failed'
        monkeypatch.setattr(server.db, 'append_task_event', append)
        assert client.post('/v1/server/echo/deploy', json={}).json()['outputs'] == {'text': 'last'}


@pytest.mark.parametrize('stop', ['cancel', 'deadline'])
def test_process_control_remains_live_during_blocked_event_write(tmp_path, monkeypatch, stop):
    entered, release = threading.Event(), threading.Event()
    config = stream_config('process', max_duration_seconds=.5 if stop == 'deadline' else 5.0)
    server = Server(name='echo', deploy=output_stream, data_dir=tmp_path, config=config)
    append = server.db.append_task_event
    def blocked(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError('Test event writer was not released')
        return append(*args, **kwargs)
    monkeypatch.setattr(server.db, 'append_task_event', blocked)
    async def scenario():
        await server.start()
        try:
            identity, future = await server.submit('echo', {})
            assert await asyncio.to_thread(entered.wait, 5)
            if stop == 'cancel':
                await server.cancel_task(identity)
            with pytest.raises(ServerError) as stopped:
                await asyncio.wait_for(asyncio.shield(future), 3)
            assert stopped.value.code == ('task_cancelled' if stop == 'cancel' else 'execution_timeout')
            worker = server.entries['echo']['workers'][0]
            assert worker.pid is None  # confirmed termination while the writer is still blocked
            assert not worker.pending_event.done()
            assert [event['kind'] for event in server.db.read_task_events(identity)] == ['start', 'error', 'done']
            release.set()
            for _ in range(200):
                if worker.pending_event.done():
                    break
                await asyncio.sleep(.01)
            else:
                raise AssertionError('Event writer did not finish')
            assert isinstance(worker.pending_event.exception(), ServerError)
            assert [event['kind'] for event in server.db.read_task_events(identity)] == ['start', 'error', 'done']
            assert (await server.deploy('echo', {}))['outputs'] == {'text': 'last'}
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())

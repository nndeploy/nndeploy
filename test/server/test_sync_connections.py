"""HTTP reader capacity is independent of task ownership and model capacity."""
import asyncio
import gc
import json
import socket
import threading
import time
import uuid
import weakref

import httpx
import pytest
import requests
from fastapi import APIRouter, Depends

from nndeploy.server import Server
from nndeploy.server.dependencies import get_deploy
from nndeploy.server.protocol import DeployRequest
from test_auth import credentials
from test_wait_timeout import blocked, blocked_engine, config, waiting_stream
from test_redis_queue import backend, redis_service
from test_streaming import live_server


async def until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Expected state was not reached')
        await asyncio.sleep(.005)


async def exchange(server, body, *, path='/v1/server/echo/deploy', headers=None, send=None, disconnect=None):
    payload = json.dumps(body).encode()
    incoming = [(b'content-type', b'application/json')]
    incoming += [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]
    scope = dict(type='http', asgi={'version': '3.0', 'spec_version': '2.4'}, http_version='1.1',
                 method='POST', scheme='http', path=path, raw_path=path.encode(), root_path='', query_string=b'',
                 headers=incoming, server=('test', 80), client=('127.0.0.1', 12345))
    delivered = False
    disconnect = disconnect or asyncio.Event()
    messages = []

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return dict(type='http.request', body=payload, more_body=False)
        await disconnect.wait()
        return dict(type='http.disconnect')

    async def output(message):
        messages.append(message)
        if send:
            await send(message)
    await server.app(scope, receive, output)
    return messages


@pytest.mark.parametrize('kind', ['standard', 'batch', 'algorithm'])
def test_slot_covers_actual_response_delivery_without_blocking_control_or_async(tmp_path, kind):
    router = APIRouter()

    @router.post('/calculate')
    async def calculate(body: DeployRequest, deploy=Depends(get_deploy)):
        result = await deploy(body)
        # More than one use of the common callable in one HTTP request still
        # occupies one connection, while each accepted task is accounted for.
        await deploy(body)
        return {'translated': result['outputs']}

    calls = []
    def calculate(inputs):
        calls.append(inputs)
        return inputs
    server = Server(name='echo', deploy=calculate, router=router, data_dir=tmp_path,
                    max_sync_connections=1, admin_token='admin')
    path = '/v1/server/echo/' + {'standard': 'deploy', 'batch': 'deploy/batch', 'algorithm': 'calculate'}[kind]
    body = {'items': [{}, {}]} if kind == 'batch' else {}

    async def scenario():
        await server.start()
        sending, release = asyncio.Event(), asyncio.Event()

        async def slow(message):
            if message['type'] == 'http.response.body':
                sending.set()
                await release.wait()
        held = asyncio.create_task(exchange(server, body, path=path, send=slow))
        try:
            await asyncio.wait_for(sending.wait(), 5)
            assert server.sync_connections == 1 and server.inflight == 0
            initial_calls = len(calls)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                for target, content in [('/v1/server/echo/deploy', {}), ('/v1/server/echo/deploy/batch', {'items': [{}]}),
                                        ('/v1/server/echo/calculate', {})]:
                    result = await client.post(target, json=content)
                    assert result.status_code == 429 and result.json()['error']['code'] == 'sync_connections_full'
                assert server.inflight == 0
                assert len(calls) == initial_calls
                assert (await client.get('/health/live')).status_code == 200
                async_result = await client.post('/v1/server/echo/deploy', json={'mode': 'async'})
                assert async_result.status_code == 202
                status_url = async_result.json()['status_url']
                assert (await client.get(status_url)).status_code == 200
                assert (await client.post(status_url + '/cancel')).status_code == 200
                metrics = await client.get('/metrics', headers={'Authorization': 'Bearer admin'})
                assert 'nndeploy_sync_connections 1.0' in metrics.text
                assert 'nndeploy_sync_connections_limit 1.0' in metrics.text
                release.set()
                await held
                assert server.sync_connections == 0 and server.sync_connections_by_subject == {}
                assert (await client.post('/v1/server/echo/deploy', json={})).status_code == 200
        finally:
            release.set()
            await held
            await server.close()
    asyncio.run(scenario())


def test_subject_limit_groups_multiple_keys_and_preserves_other_callers(tmp_path, credentials, monkeypatch):
    auth, headers = credentials
    token = uuid.uuid4().hex
    monkeypatch.setenv('TEST_ALICE_SECOND_KEY', token)
    auth['keys'].append(dict(auth['keys'][0], token_env='TEST_ALICE_SECOND_KEY'))
    auth['max_sync_connections_per_subject'] = 1
    server = Server(name='echo', deploy=lambda inputs: inputs, data_dir=tmp_path, auth_config=auth, max_sync_connections=3)

    async def scenario():
        await server.start()
        sending, release = asyncio.Event(), asyncio.Event()

        async def slow(message):
            if message['type'] == 'http.response.body':
                sending.set()
                await release.wait()
        held = asyncio.create_task(exchange(server, {}, headers=headers['alice'], send=slow))
        try:
            await asyncio.wait_for(sending.wait(), 5)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                result = await client.post('/v1/server/echo/deploy', json={}, headers={'Authorization': 'Bearer ' + token})
                assert result.status_code == 429
                assert (await client.post('/v1/server/echo/deploy', json={}, headers=headers['bob'])).status_code == 200
                assert server.sync_connections_by_subject == {'alice': 1}
                assert (await client.post('/v1/server/echo/deploy', json={}, headers=headers['reader'])).status_code == 403
                assert server.sync_connections_by_subject == {'alice': 1}
        finally:
            release.set()
            await held
            assert server.sync_connections_by_subject == {}
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('scheduling,isolation', [('serial', 'thread'), ('serial', 'process'), ('concurrent', 'thread'), ('engine', 'thread')])
@pytest.mark.parametrize('cancel_http', [False, True])
def test_disconnect_releases_reader_but_keeps_task_and_idempotency(tmp_path, scheduling, isolation, cancel_http):
    release = tmp_path / 'release'
    server = Server(name='echo', deploy=blocked_engine if scheduling == 'engine' else blocked, data_dir=tmp_path,
                    config=config(scheduling=scheduling, isolation=isolation, http_wait_timeout_seconds=10), max_sync_connections=1)
    body = {'inputs': {'release': str(release)}}
    headers = {'Idempotency-Key': 'same'}

    async def scenario():
        await server.start()
        disconnected = asyncio.Event()
        request = asyncio.create_task(exchange(server, body, headers=headers, disconnect=disconnected))
        try:
            await until(lambda: bool(server.task_futures))
            identity = next(iter(server.task_futures))
            await until(lambda: server.db.get_task(identity) is not None)
            assert server.sync_connections == 1
            if cancel_http:
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
            else:
                disconnected.set()
                assert await asyncio.wait_for(request, 1) == []
            assert server.sync_connections == 0 and not server.sync_connections_by_subject
            assert server.inflight == 1 and not server.db.get_task(identity)['cancel_requested']
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/v1/server/echo/deploy', json=dict(body, wait_timeout_seconds=.02), headers=headers)
                assert response.status_code == 504 and response.json()['error']['task_id'] == identity
                assert server.sync_connections == 0 and server.inflight == 1
                release.touch()
                await until(lambda: server.db.get_task(identity)['state'] == 'succeeded')
                replay = await client.post('/v1/server/echo/deploy', json=body, headers=headers)
                assert replay.status_code == 200 and replay.json()['task_id'] == identity
        finally:
            release.touch()
            await asyncio.gather(request, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['timeout', 'socket', 'cancel'])
def test_slow_or_failed_response_send_releases_slot(tmp_path, monkeypatch, failure):
    import nndeploy.server.request as request_module
    monkeypatch.setattr(request_module, 'SYNC_SEND_SECONDS', .05)
    server = Server(name='echo', deploy=lambda inputs: inputs, data_dir=tmp_path, max_sync_connections=1)

    async def scenario():
        await server.start()
        sending = asyncio.Event()
        identity = None

        async def send(message):
            nonlocal identity
            if message['type'] == 'http.response.body':
                identity = json.loads(message['body'])['task_id']
                sending.set()
                if failure == 'socket':
                    raise OSError('Socket disconnected')
                await asyncio.Event().wait()
        request = asyncio.create_task(exchange(server, {}, send=send))
        try:
            await asyncio.wait_for(sending.wait(), 5)
            if failure == 'cancel':
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
            else:
                await asyncio.wait_for(request, 1)
            assert server.sync_connections == 0 and not server.sync_connections_by_subject
            assert server.db.get_task(identity)['state'] == 'succeeded'
        finally:
            await asyncio.gather(request, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())


def test_shared_replicas_limit_their_own_connections_not_the_shared_task(backend, tmp_path):
    db, redis_config = backend
    options = dict(name='echo', deploy=blocked, data_dir=tmp_path, config=config(http_wait_timeout_seconds=10),
                   database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config, max_sync_connections=1)
    first, second = Server(**options), Server(**options)
    release = tmp_path / 'release'
    body = {'inputs': {'release': str(release)}}
    headers = {'Idempotency-Key': 'shared'}

    async def scenario():
        await first.start()
        await second.start()
        disconnect = asyncio.Event()
        held = asyncio.create_task(exchange(first, body, headers=headers, disconnect=disconnect))
        try:
            await until(lambda: bool(first.task_futures))
            identity = next(iter(first.task_futures))
            await until(lambda: db.get_task(identity) is not None)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=second.app), base_url='http://test') as client:
                response = await client.post('/v1/server/echo/deploy', json=dict(body, wait_timeout_seconds=.02), headers=headers)
                assert response.status_code == 504 and response.json()['error']['task_id'] == identity
                assert first.sync_connections == 1 and second.sync_connections == 0
                assert db.queue_usage(first.entries['echo']['queue'].key)['inflight'] == 1
            disconnect.set()
            await asyncio.wait_for(held, 1)
            release.touch()
            await until(lambda: db.get_task(identity)['state'] == 'succeeded')
        finally:
            release.touch()
            disconnect.set()
            await held
            await first.close()
            await second.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('batch', [False, True])
def test_real_http_disconnect_keeps_compute_and_releases_connection(tmp_path, batch):
    release = tmp_path / 'release'
    server = Server(name='echo', deploy=blocked, data_dir=tmp_path, max_sync_connections=1,
                    config=config(http_wait_timeout_seconds=10))
    item = {'inputs': {'release': str(release)}}
    path = '/v1/server/echo/deploy' + ('/batch' if batch else '')
    body = json.dumps({'items': [dict(item, idempotency_key='first'), {}]} if batch else item).encode()
    with live_server(server) as address:
        host, port = address.removeprefix('http://').split(':')
        connection = socket.create_connection((host, int(port)), timeout=5)
        try:
            lines = [f'POST {path} HTTP/1.1', 'Host: localhost', 'Content-Type: application/json',
                     f'Content-Length: {len(body)}']
            if not batch:
                lines.append('Idempotency-Key: first')
            connection.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode() + body)
            deadline = time.monotonic() + 5
            while server.inflight != (2 if batch else 1):
                assert time.monotonic() < deadline
                time.sleep(.005)
            assert server.sync_connections == 1
            connection.shutdown(socket.SHUT_RDWR)
            connection.close()
            deadline = time.monotonic() + 2
            while server.sync_connections:
                assert time.monotonic() < deadline
                time.sleep(.005)
            assert not server.sync_connections_by_subject
            assert server.inflight == (2 if batch else 1)
            retry = requests.post(address + '/v1/server/echo/deploy', json=dict(item, wait_timeout_seconds=.02),
                                  headers={'Idempotency-Key': 'first'}, timeout=3)
            assert retry.status_code == 504, retry.text
            status_url = retry.json()['error']['status_url']
            release.touch()
            deadline = time.monotonic() + 5
            while True:
                record = requests.get(address + status_url, timeout=2).json()
                if record['state'] == 'succeeded':
                    break
                assert time.monotonic() < deadline
                time.sleep(.005)
            assert record['cancel_requested'] is False
        finally:
            release.touch()
            connection.close()


def test_rejections_execution_failure_and_response_conversion_failure_do_not_leak_slots(tmp_path):
    router = APIRouter()

    @router.post('/broken-response')
    async def broken(body: DeployRequest, deploy=Depends(get_deploy)):
        await deploy(body)
        raise RuntimeError('Response conversion failed')

    def calculate(inputs):
        if inputs['text'] == 'fail':
            raise RuntimeError('Algorithm failed')
        return inputs

    server = Server(name='echo', deploy=calculate, router=router, data_dir=tmp_path, max_sync_connections=1,
                    config={'inputs': {'text': {'type': 'string'}}})

    async def scenario():
        await server.start()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False), base_url='http://test') as client:
                for path, body, expected in [('/deploy', {}, 422), ('/deploy', {'inputs': {'text': 'fail'}}, 500),
                                             ('/broken-response', {'inputs': {'text': 'ok'}}, 500),
                                             ('/deploy', {'inputs': {'text': 'ok'}}, 200)]:
                    response = await client.post('/v1/server/echo' + path, json=body)
                    assert response.status_code == expected, response.text
                    assert server.sync_connections == 0 and not server.sync_connections_by_subject
        finally:
            await server.close()
    asyncio.run(scenario())


def test_disconnected_batch_does_not_leave_an_owned_result_waiter(tmp_path):
    release = tmp_path / 'release'
    server = Server(name='echo', deploy=blocked, data_dir=tmp_path, config=config(http_wait_timeout_seconds=10))

    async def scenario():
        await server.start()
        disconnect = asyncio.Event()
        body = {'items': [{'inputs': {'release': str(release)}}, {}]}
        request = asyncio.create_task(exchange(server, body, path='/v1/server/echo/deploy/batch', disconnect=disconnect))
        try:
            await until(lambda: len(server.task_futures) == 2 and all(server.db.get_task(key) for key in server.task_futures))
            disconnect.set()
            await asyncio.wait_for(request, 1)
            if server.control_tasks:
                _, pending = await asyncio.wait(list(server.control_tasks), timeout=.1)
                assert not pending, 'Disconnected result waiters must not remain owned Server operations'
            assert server.inflight == 2
        finally:
            release.touch()
            await request
            await server.close()
    asyncio.run(scenario())


def test_disconnect_during_batch_database_admission_finishes_admission_without_result_waiter(tmp_path, monkeypatch):
    release = tmp_path / 'release'
    server = Server(name='echo', deploy=blocked, data_dir=tmp_path, config=config(http_wait_timeout_seconds=10))
    entered, release_database = threading.Event(), threading.Event()
    create = server.db.create_task
    writes = 0

    def delayed(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            entered.set()
            assert release_database.wait(5)
        return create(*args, **kwargs)
    monkeypatch.setattr(server.db, 'create_task', delayed)

    async def scenario():
        await server.start()
        disconnect = asyncio.Event()
        request = asyncio.create_task(exchange(server, {'items': [{'inputs': {'release': str(release)}}, {}, {}]},
                                               path='/v1/server/echo/deploy/batch', disconnect=disconnect))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            disconnect.set()
            await asyncio.wait_for(request, 1)
            assert server.sync_connections == 0
            assert server.control_tasks  # The pending durable write remains owned.
            release_database.set()
            await until(lambda: len(server.task_futures) == 3 and not server.control_tasks)
            identities = list(server.task_futures)
            assert all(server.db.get_task(key) is not None for key in identities)
            assert server.inflight == 3
            release.touch()
            await until(lambda: all(server.db.get_task(key)['state'] == 'succeeded' for key in identities))
        finally:
            release_database.set()
            release.touch()
            await request
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['sync', 'async', 'stream'])
def test_finished_http_readers_do_not_accumulate_observers_on_a_running_task(tmp_path, monkeypatch, mode):
    settings = config()
    if mode == 'stream':
        settings['streaming'] = {'enabled': True}
    server = Server(name='echo', deploy=waiting_stream if mode == 'stream' else blocked, data_dir=tmp_path, config=settings)
    release = tmp_path / 'release'
    observers = []
    submit = server.submit

    async def observe(*args, **kwargs):
        identity, observer = await submit(*args, **kwargs)
        observers.append(weakref.ref(observer))
        return identity, observer
    monkeypatch.setattr(server, 'submit', observe)

    async def scenario():
        await server.start()
        body = {'inputs': {'release': str(release)}, 'mode': mode}
        if mode == 'sync':
            body['wait_timeout_seconds'] = .01
        try:
            for _ in range(4):
                async def send(message):
                    if mode == 'stream' and message['type'] == 'http.response.body':
                        raise OSError('Client closed the stream')
                messages = await exchange(server, body, headers={'Idempotency-Key': 'same'}, send=send)
                assert messages[0]['status'] == {'sync': 504, 'async': 202, 'stream': 200}[mode]
            # Future callbacks run on the next loop turn; collect unreachable
            # exception tracebacks too, while the accepted computation stays live.
            await asyncio.sleep(.01)
            gc.collect()
            assert len(observers) == 4 and all(ref() is None for ref in observers)
            assert server.inflight == 1 and server.sync_connections == server.stream_connections == 0
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_timed_out_batch_waiters_are_reclaimed_while_computation_runs(tmp_path, monkeypatch):
    server = Server(name='echo', deploy=blocked, data_dir=tmp_path, config=config())
    release = tmp_path / 'release'
    readers = []
    deploy = server.deploy_batch

    async def observe(*args, **kwargs):
        readers.append(weakref.ref(asyncio.current_task()))
        return await deploy(*args, **kwargs)
    monkeypatch.setattr(server, 'deploy_batch', observe)

    async def scenario():
        await server.start()
        try:
            for _ in range(4):
                await exchange(server, {'items': [{'inputs': {'release': str(release)}, 'idempotency_key': 'same'}],
                                        'wait_timeout_seconds': .01}, path='/v1/server/echo/deploy/batch')
            await asyncio.sleep(.01)
            gc.collect()
            assert len(readers) == 4 and all(ref() is None for ref in readers)
            assert server.inflight == 1
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())

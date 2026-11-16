"""Result wait deadlines never surrender task ownership or execution capacity."""
import asyncio
from pathlib import Path
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_auth import credentials
from test_db import db
from test_redis_queue import redis_service, backend


def blocked(inputs):
    release = inputs['release']
    deadline = time.monotonic() + 10
    while release and not Path(release).exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Test did not release worker')
        time.sleep(.005)
    return {'done': True}


async def blocked_engine(inputs):
    release = inputs['release']
    deadline = time.monotonic() + 10
    while release and not Path(release).exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Test did not release engine')
        await asyncio.sleep(.005)
    return {'done': True}


def config(**execution):
    return {'inputs': {'release': {'type': 'string', 'required': False, 'default': ''}},
            'outputs': {'done': {'type': 'boolean'}},
            'execution': {'http_wait_timeout_seconds': .1, **execution}}


def completed(client, path, **options):
    for _ in range(200):
        response = client.get(path, **options)
        assert response.status_code == 200, response.text
        value = response.json()
        if value['state'] in ('succeeded', 'failed', 'cancelled', 'expired'):
            return value
        time.sleep(.01)
    raise AssertionError('Task did not reach terminal state')


@pytest.mark.parametrize('scheduling,isolation', [('serial', 'thread'), ('serial', 'process'), ('concurrent', 'thread'), ('engine', 'thread')])
def test_http_timeout_keeps_task_and_capacity_and_allows_idempotent_replay(tmp_path, scheduling, isolation):
    server = Server(name='echo', deploy=blocked_engine if scheduling == 'engine' else blocked, data_dir=tmp_path,
                    config=config(scheduling=scheduling, isolation=isolation, max_queue_size=0))
    release = tmp_path / 'release'
    body = {'inputs': {'release': str(release)}, 'wait_timeout_seconds': .02}
    headers = {'Idempotency-Key': uuid.uuid4().hex}
    with TestClient(server.app) as client:
        try:
            response = client.post('/v1/server/echo/deploy', json=body, headers=headers)
            assert response.status_code == 504, response.text
            error = response.json()['error']
            assert error['code'] == 'wait_timeout' and error['accepted'] is True
            assert error['retryable'] is True and response.headers['location'] == error['status_url']
            assert error['status_url'] == '/v1/tasks/' + error['task_id']
            record = client.get(error['status_url']).json()
            assert record['state'] in ('pending', 'running') and not record['cancel_requested']
            assert server.inflight == 1
            assert client.post('/v1/server/echo/deploy', json={}).status_code == 429
            retry = client.post('/v1/server/echo/deploy', json=body, headers=headers)
            assert retry.status_code == 504 and retry.json()['error']['task_id'] == error['task_id']
            release.touch()
            assert completed(client, error['status_url'])['state'] == 'succeeded'
            replay = client.post('/v1/server/echo/deploy', json=dict(body, wait_timeout_seconds=.05), headers=headers)
            assert replay.status_code == 200 and replay.json()['task_id'] == error['task_id']
            assert replay.json()['outputs'] == {'done': True}
        finally:
            release.touch()


@pytest.mark.parametrize('body', [
    {'wait_timeout_seconds': 0}, {'wait_timeout_seconds': -1}, {'wait_timeout_seconds': True},
    {'wait_timeout_seconds': '1'}, {'wait_timeout_seconds': 3601}, {'wait_timeout_seconds': .2},
    {'mode': 'async', 'wait_timeout_seconds': .01}, {'mode': 'stream', 'wait_timeout_seconds': .01},
])
def test_invalid_or_excessive_wait_is_rejected_before_admission(tmp_path, body):
    calls = []
    server = Server(name='echo', deploy=lambda inputs: calls.append(inputs), config=config(), data_dir=tmp_path)
    with TestClient(server.app) as client:
        assert client.post('/v1/server/echo/deploy', json=body).status_code == 422
        assert not calls and not server.task_futures


def test_entry_default_wait_limit_and_python_wait_policy(tmp_path):
    server = Server(name='echo', deploy=blocked, config=config(http_wait_timeout_seconds=.02), data_dir=tmp_path)
    release = tmp_path / 'release'
    with TestClient(server.app) as client:
        try:
            assert client.get('/v1/server/echo/schema').json()['http_wait_timeout_seconds'] == .02
            response = client.post('/v1/server/echo/deploy', json={'inputs': {'release': str(release)}})
            assert response.status_code == 504
            release.touch()
            assert completed(client, response.json()['error']['status_url'])['state'] == 'succeeded'
        finally:
            release.touch()
    other = Server(name='echo', deploy=blocked_engine, config=config(scheduling='engine'), data_dir=tmp_path / 'python')
    gate = tmp_path / 'python-release'
    async def scenario():
        await other.start()
        try:
            with pytest.raises(ServerError) as timeout:
                await other.deploy('echo', {'release': str(gate)}, wait_timeout_seconds=.01)
            assert timeout.value.code == 'wait_timeout' and timeout.value.accepted is True
            assert not other.db.get_task(timeout.value.task_id)['cancel_requested']
            # Python's omitted deadline retains the original unbounded wait behavior.
            waiting = asyncio.create_task(other.deploy('echo', {'release': str(gate)}))
            await asyncio.sleep(.12)
            assert not waiting.done()
            gate.touch()
            assert (await waiting)['outputs'] == {'done': True}
        finally:
            gate.touch()
            await other.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '-Infinity'])
def test_nonfinite_wait_values_return_validation_errors_without_admission(tmp_path, number):
    calls = []
    server = Server(name='echo', deploy=lambda inputs: calls.append(inputs), config=config(), data_dir=tmp_path)
    with TestClient(server.app, raise_server_exceptions=False) as client:
        response = client.post('/v1/server/echo/deploy', content='{"wait_timeout_seconds":' + number + '}',
                               headers={'Content-Type': 'application/json'})
        assert response.status_code == 422, response.text
        assert not calls and not server.task_futures


def test_batch_wait_uses_one_window_and_retains_completed_and_pending_items(tmp_path):
    server = Server(name='echo', deploy=blocked, data_dir=tmp_path, config=config())
    release = tmp_path / 'release'
    body = {'items': [{}, {'inputs': {'unknown': 1}}, {'inputs': {'release': str(release)}, 'idempotency_key': 'one'}],
            'wait_timeout_seconds': .05}
    with TestClient(server.app) as client:
        try:
            response = client.post('/v1/server/echo/deploy/batch', json=body)
            assert response.status_code == 200, response.text
            rows = response.json()['items']
            assert [row['status_code'] for row in rows] == [200, 422, 504]
            assert [row['accepted'] for row in rows] == [True, False, True]
            assert rows[2]['error']['code'] == 'wait_timeout'
            assert not server.db.get_task(rows[2]['task_id'])['cancel_requested']
            assert client.post('/v1/server/echo/deploy/batch', json=dict(body, mode='async')).status_code == 422
            release.touch()
            assert completed(client, rows[2]['status_url'])['state'] == 'succeeded'
            retry = client.post('/v1/server/echo/deploy/batch', json={'items': [body['items'][2]]})
            assert retry.json()['items'][0]['task_id'] == rows[2]['task_id']
            assert retry.json()['items'][0]['outputs'] == {'done': True}
        finally:
            release.touch()


def test_timed_out_task_records_keep_owner_and_survive_server_restart(db, tmp_path, credentials):
    auth, headers = credentials
    options = dict(data_dir=tmp_path, database_url=db.url.render_as_string(hide_password=False), auth_config=auth)
    release = tmp_path / 'release'
    server = Server(name='echo', deploy=blocked, config=config(), **options)
    with TestClient(server.app) as client:
        try:
            response = client.post('/v1/server/echo/deploy', json={'inputs': {'release': str(release)}}, headers=headers['alice'])
            assert response.status_code == 504
            error = response.json()['error']
            assert client.get(error['status_url'], headers=headers['bob']).status_code == 404
            release.touch()
            assert completed(client, error['status_url'], headers=headers['alice'])['state'] == 'succeeded'
        finally:
            release.touch()
    with TestClient(Server(**options).app) as client:
        assert client.get(error['status_url'], headers=headers['alice']).json()['outputs'] == {'done': True}
        assert client.get(error['status_url'], headers=headers['bob']).status_code == 404


def test_wait_timeout_does_not_replace_queue_expiry_or_explicit_process_cancel(tmp_path):
    release = tmp_path / 'release'
    server = Server(name='echo', deploy=blocked, config=config(isolation='process', queue_timeout_seconds=.05), data_dir=tmp_path)
    with TestClient(server.app) as client:
        try:
            first = client.post('/v1/server/echo/deploy', json={'inputs': {'release': str(release)}, 'wait_timeout_seconds': .02})
            first_task = first.json()['error']
            second = client.post('/v1/server/echo/deploy', json={'wait_timeout_seconds': .02}).json()['error']
            time.sleep(.07)
            assert client.post(first_task['status_url'] + '/cancel').status_code == 200
            assert completed(client, first_task['status_url'])['state'] == 'cancelled'
            expired = completed(client, second['status_url'])
            assert expired['state'] == 'expired' and expired['error']['code'] == 'queue_timeout'
            recovered = client.post('/v1/server/echo/deploy', json={'mode': 'async'})
            assert recovered.status_code == 202
            assert completed(client, recovered.json()['status_url'])['state'] == 'succeeded'
        finally:
            release.touch()


def test_shared_task_survives_http_timeout_and_wait_policy_does_not_split_queue(backend, tmp_path):
    db, redis_config = backend
    options = dict(name='echo', deploy=blocked, data_dir=tmp_path, redis_config=redis_config,
                   database_url=db.url.render_as_string(hide_password=False))
    server = Server(**options, config=config())
    other = Server(**options, config=config(http_wait_timeout_seconds=.5))
    entry = server.entries['echo']
    release = tmp_path / 'release'
    with TestClient(server.app) as client, TestClient(other.app):
        try:
            assert entry['queue'].key == other.entries['echo']['queue'].key
            response = client.post('/v1/server/echo/deploy', json={'inputs': {'release': str(release)}, 'wait_timeout_seconds': .02})
            assert response.status_code == 504, response.text
            error = response.json()['error']
            assert db.get_task(error['task_id'])['state'] in ('pending', 'running')
            assert db.queue_usage(entry['queue'].key)['inflight'] == 1
            release.touch()
            assert completed(client, error['status_url'])['state'] == 'succeeded'
        finally:
            release.touch()


class WaitingBatch:
    calls = []

    def batch_key(self, inputs):
        return 'same'

    def deploy_batch(self, items):
        self.calls.append(len(items))
        return [{'done': True} for _ in items]


def test_timeout_during_dynamic_batch_collection_keeps_item_for_actual_batch(tmp_path):
    WaitingBatch.calls = []
    server = Server(name='echo', handler_factory=WaitingBatch, data_dir=tmp_path,
                    config=config(batching={'enabled': True, 'max_size': 2, 'max_wait_ms': 150}))
    with TestClient(server.app) as client:
        first = client.post('/v1/server/echo/deploy', json={'wait_timeout_seconds': .02})
        assert first.status_code == 504
        second = client.post('/v1/server/echo/deploy', json={'mode': 'async'})
        assert second.status_code == 202
        assert completed(client, first.json()['error']['status_url'])['state'] == 'succeeded'
        assert completed(client, second.json()['status_url'])['state'] == 'succeeded'
        assert WaitingBatch.calls == [2]


def waiting_stream(inputs):
    yield {'done': False}
    yield blocked(inputs)


def test_sync_stream_can_finish_and_replay_events_after_http_wait_timeout(tmp_path):
    from test_streaming import parse_events
    settings = config()
    settings['streaming'] = {'enabled': True}
    server = Server(name='echo', deploy=waiting_stream, data_dir=tmp_path, config=settings)
    release = tmp_path / 'release'
    with TestClient(server.app) as client:
        try:
            response = client.post('/v1/server/echo/deploy', json={'inputs': {'release': str(release)}, 'wait_timeout_seconds': .02})
            assert response.status_code == 504
            path = response.json()['error']['status_url']
            release.touch()
            assert completed(client, path)['outputs'] == {'done': True}
            events = parse_events(client.get(path + '/events', headers={'Accept': 'text/event-stream'}).text)
            assert events[-1]['kind'] == 'done' and events[-1]['data']['state'] == 'succeeded'
            assert [item['data']['data'] for item in events if item['kind'] == 'delta'] == [{'done': False}, {'done': True}]
        finally:
            release.touch()


def test_algorithm_route_uses_same_wait_limit_and_task_status(tmp_path):
    from fastapi import APIRouter, Depends
    from nndeploy.server.dependencies import get_deploy
    from nndeploy.server.protocol import DeployRequest
    router = APIRouter()

    @router.post('/calculate')
    async def calculate(body: DeployRequest, deploy=Depends(get_deploy)):
        return await deploy(body)

    server = Server(name='echo', deploy=blocked, router=router, data_dir=tmp_path, config=config())
    release = tmp_path / 'release'
    with TestClient(server.app) as client:
        try:
            response = client.post('/v1/server/echo/calculate', json={'inputs': {'release': str(release)}, 'wait_timeout_seconds': .02})
            assert response.status_code == 504
            release.touch()
            assert completed(client, response.json()['error']['status_url'])['state'] == 'succeeded'
        finally:
            release.touch()

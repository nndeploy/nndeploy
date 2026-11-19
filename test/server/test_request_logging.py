"""Request correlation survives execution boundaries and never logs HTTP content."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import re
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from nndeploy.server import Server
from nndeploy.server.logging_taskid import get_request_id, get_task_id
from test_db import db
from test_redis_queue import backend, redis_service


def correlated_echo(inputs):
    identity = get_request_id()
    time.sleep(.01)
    assert get_request_id() == identity
    return {'request_id': identity, 'task_id': get_task_id(), 'pid': os.getpid()}


async def correlated_engine(inputs):
    identity = get_request_id()
    await asyncio.sleep(.02)
    assert get_request_id() == identity
    return {'request_id': identity, 'task_id': get_task_id(), 'pid': os.getpid()}


def events(caplog, kind=None):
    values = [record.server_event for record in caplog.records if hasattr(record, 'server_event')]
    return [value for value in values if kind is None or value['event'] == kind]


@pytest.mark.parametrize('scheduling,isolation', [('serial', 'thread'), ('serial', 'process'), ('concurrent', 'thread'), ('engine', 'thread')])
def test_http_and_worker_ids_are_isolated_and_headers_cannot_spoof_them(tmp_path, caplog, scheduling, isolation):
    caplog.set_level(logging.INFO, logger='nndeploy.server.events')
    server = Server(name='echo', deploy=correlated_engine if scheduling == 'engine' else correlated_echo, resources=tmp_path, data_dir=tmp_path,
                    config={'execution': {'scheduling': scheduling, 'isolation': isolation,
                                          'max_concurrency_per_worker': 1 if scheduling == 'serial' else 2}})
    with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: client.post('/v1/server/echo/deploy', json={}, headers={'X-Request-ID': 'attacker-secret'}), range(2)))
        identities = set()
        for result in results:
            assert result.status_code == 200, result.text
            identity = result.headers['x-request-id']
            assert re.fullmatch(r'req_[0-9a-f]{32}', identity)
            identities.add(identity)
            value = result.json()
            assert value['outputs']['request_id'] == identity
            assert value['outputs']['task_id'] == value['task_id']
            stored = client.get('/v1/tasks/' + value['task_id'])
            assert stored.json()['request_id'] == identity
            assert stored.headers['x-request-id'] != identity
            started = next(event for event in events(caplog, 'worker_request_started') if event['task_id'] == value['task_id'])
            assert started['request_id'] == identity and started['worker_pid'] == value['outputs']['pid']
        assert len(identities) == 2
        direct = client.portal.call(server.deploy, 'echo', {})
        assert direct['outputs']['request_id'] is None
        assert get_request_id() is None
        assert 'attacker-secret' not in json.dumps(events(caplog))


def test_idempotent_replay_keeps_original_id_and_survives_restart(db, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger='nndeploy.server.events')
    options = dict(resources=tmp_path, data_dir=tmp_path, database_url=db.url.render_as_string(hide_password=False))
    server = Server(name='echo', deploy=correlated_echo, **options)
    with TestClient(server.app) as client:
        headers = {'Idempotency-Key': uuid.uuid4().hex}
        first = client.post('/v1/server/echo/deploy', json={}, headers=headers)
        second = client.post('/v1/server/echo/deploy', json={}, headers=headers)
        assert first.json() == second.json()
        assert first.headers['x-request-id'] != second.headers['x-request-id']
        task_id = first.json()['task_id']
        selected = [event for event in events(caplog, 'task_selected') if event['task_id'] == task_id]
        assert {event['request_id'] for event in selected} == {first.headers['x-request-id'], second.headers['x-request-id']}
        finished = [event for event in events(caplog, 'task_finished') if event['task_id'] == task_id]
        assert len(finished) == 1 and finished[0]['request_id'] == first.headers['x-request-id']
    with TestClient(Server(**options).app) as client:
        result = client.get('/v1/tasks/' + task_id)
        assert result.json()['request_id'] == first.headers['x-request-id']


def test_errors_unknown_routes_and_secrets_only_emit_safe_metadata(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger='nndeploy.server.events')
    server = Server(name='echo', deploy=correlated_echo, resources=tmp_path, data_dir=tmp_path, config={}, cors_origins=['http://client.test'])

    @server.app.get('/fail')
    async def fail():
        raise RuntimeError('private-exception-content')

    with TestClient(server.app, raise_server_exceptions=False) as client:
        results = [
            client.get('/private-path?password=private-query'),
            client.post('/v1/server/echo/deploy', json={'inputs': {'private-field': 'private-value'}}),
            client.get('/fail'),
            client.post('/v1/server/echo/deploy', content='private-invalid-json', headers={'Content-Type': 'application/json'}),
            client.post('/v1/server/echo/deploy', json={}, headers={'Authorization': 'Bearer private-token', 'Cookie': 'secret=private-cookie'}),
        ]
        assert [response.status_code for response in results] == [404, 422, 500, 422, 401]
        for response in results:
            identity = response.headers['x-request-id']
            recorded = [value for value in events(caplog, 'http_request') if value['request_id'] == identity]
            assert len(recorded) == 1 and recorded[0]['status'] == response.status_code
        body = json.dumps(events(caplog))
        assert 'private-' not in body and 'password' not in body and 'secret=' not in body
        assert 'private-exception-content' not in results[2].text
        response = client.post('/v1/server/echo/deploy', json={}, headers={'Origin': 'http://client.test'})
        assert 'X-Request-ID' in response.headers['access-control-expose-headers']


def correlated_stream(inputs):
    for index in range(2):
        yield {'request_id': get_request_id(), 'index': index}


def test_stream_has_one_http_id_and_records_original_id_in_events(tmp_path, caplog):
    from test_streaming import parse_events
    caplog.set_level(logging.INFO, logger='nndeploy.server.events')
    server = Server(name='echo', deploy=correlated_stream, config={'streaming': {'enabled': True}}, resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/deploy', json={'mode': 'stream'})
        assert response.status_code == 200
        identity = response.headers['x-request-id']
        deltas = [item for item in parse_events(response.text) if item['kind'] == 'delta']
        assert len(deltas) == 2
        assert all(item['data']['data']['request_id'] == identity for item in deltas)
        assert len([item for item in events(caplog, 'http_request') if item['request_id'] == identity]) == 1


class CorrelatedBatch:
    def batch_key(self, inputs):
        assert get_request_id() is not None and get_task_id() is not None
        return 'same'

    def deploy_batch(self, items):
        assert get_request_id() is None and get_task_id() is None
        return [{'request_id': item['request_id'], 'task_id': item['task_id']} for item in items]


def test_dynamic_batch_keeps_per_item_ids_without_inventing_single_request_owner(tmp_path):
    server = Server(name='echo', handler_factory=CorrelatedBatch, resources=tmp_path, data_dir=tmp_path,
                    config={'execution': {'batching': {'enabled': True, 'max_size': 2, 'max_wait_ms': 50}}})
    with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: client.post('/v1/server/echo/deploy', json={}), range(2)))
        for response in results:
            assert response.status_code == 200, response.text
            assert response.json()['outputs']['request_id'] == response.headers['x-request-id']
        assert results[0].headers['x-request-id'] != results[1].headers['x-request-id']


def test_shared_process_worker_gets_original_request_id_from_database(backend, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger='nndeploy.server.events')
    db, redis_config = backend
    server = Server(name='echo', deploy=correlated_echo, resources=tmp_path, data_dir=tmp_path, redis_config=redis_config,
                    database_url=db.url.render_as_string(hide_password=False), config={'execution': {'isolation': 'process'}})
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/deploy', json={})
        assert response.status_code == 200, response.text
        identity = response.headers['x-request-id']
        value = response.json()
        assert value['outputs']['request_id'] == identity and value['outputs']['pid'] != os.getpid()
        assert db.get_task(value['task_id'])['request_id'] == identity
        assert any(item.get('request_id') == identity for item in events(caplog, 'task_finished'))


def test_recreated_worker_logs_new_process_id_with_same_request_correlation(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger='nndeploy.server.events')
    server = Server(name='echo', deploy=correlated_echo, resources=tmp_path, data_dir=tmp_path, config={'execution': {'isolation': 'process'}})
    with TestClient(server.app) as client:
        worker = server.entries['echo']['workers'][0]
        previous = worker.pid
        # Fault injection: kill this test's idle worker, then let normal deploy recover it.
        worker.process.terminate()
        worker.process.join(timeout=5)
        assert not worker.process.is_alive()
        response = client.post('/v1/server/echo/deploy', json={})
        assert response.status_code == 200, response.text
        value = response.json()
        assert value['outputs']['pid'] != previous
        finished = next(item for item in events(caplog, 'worker_request_finished') if item['task_id'] == value['task_id'])
        assert finished['worker_pid'] == value['outputs']['pid']
        assert finished['request_id'] == value['outputs']['request_id'] == response.headers['x-request-id']


def test_request_id_migration_preserves_existing_records(db, tmp_path):
    identity = 'task_' + uuid.uuid4().hex
    db.create_task(identity, 'legacy', time.time(), {'text': 'keep me'})
    with db.engine.begin() as connection:
        connection.execute(text('ALTER TABLE server_tasks DROP COLUMN request_id'))
        connection.execute(text("UPDATE alembic_version SET version_num = '0013_task_payloads'"))
    server = Server(resources=tmp_path, data_dir=tmp_path, database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        record = client.get('/v1/tasks/' + identity).json()
        assert record['inputs'] == {'text': 'keep me'} and record['request_id'] is None
        column = next(item for item in inspect(server.db.engine).get_columns('server_tasks') if item['name'] == 'request_id')
        assert column['nullable'] and column['type'].length == 36


def test_structured_formatter_keeps_json_even_with_legacy_task_prefix_factory(caplog):
    from nndeploy.server.logging_taskid import install_taskid_logrecord_factory, task_log_context, set_request_id, reset_request_id
    from nndeploy.server.request import event, StructuredFormatter
    previous = logging.getLogRecordFactory()
    caplog.set_level(logging.INFO, logger='nndeploy.server.events')
    token = set_request_id('req_' + '1' * 32)
    try:
        install_taskid_logrecord_factory()
        installed = logging.getLogRecordFactory()
        install_taskid_logrecord_factory()
        assert logging.getLogRecordFactory() is installed
        with task_log_context('task_example'):
            event('test', task_id='task_example', request_id=get_request_id())
        record = caplog.records[-1]
        assert json.loads(record.getMessage())['task_id'] == 'task_example'
        assert json.loads(StructuredFormatter('%(levelname)s %(message)s').format(record))['request_id'] == 'req_' + '1' * 32
    finally:
        logging.setLogRecordFactory(previous)
        reset_request_id(token)

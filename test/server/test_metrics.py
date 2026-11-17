"""Protected scrape interface across actual execution, queues and dependencies."""
from concurrent.futures import ThreadPoolExecutor
import io
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_auth import credentials
from test_db import db
from test_redis_queue import backend, redis_service
from test_server_api import text_graph


ADMIN = {'Authorization': 'Bearer metrics-admin-secret'}


def samples(response):
    assert response.status_code == 200, response.text
    return [sample for family in text_string_to_metric_families(response.text) for sample in family.samples]


def value(records, metric, **labels):
    matches = [record.value for record in records if record.name == metric and record.labels == labels]
    assert len(matches) == 1, (metric, labels, matches)
    return matches[0]


def scrape(client):
    return samples(client.get('/metrics', headers=ADMIN))


def test_metrics_authentication_is_administrator_only_and_no_cache(tmp_path, credentials):
    auth, headers = credentials
    server = Server(data_dir=tmp_path, admin_token='metrics-admin-secret', auth_config=auth)
    with TestClient(server.app) as client:
        assert client.get('/metrics').status_code == 401
        assert client.get('/metrics', headers=headers['alice']).status_code == 403
        assert client.get('/metrics', headers={'Authorization': 'Bearer invalid'}).status_code == 401
        result = client.get('/metrics', headers=ADMIN)
        assert result.status_code == 200
        assert result.headers['content-type'].startswith('text/plain')
        assert result.headers['cache-control'] == 'no-store'
        assert '/metrics' not in client.get('/openapi.json').json()['paths']
    with TestClient(Server(data_dir=tmp_path / 'disabled').app) as client:
        assert client.get('/metrics', headers=ADMIN).status_code == 503


def test_metrics_are_per_server_have_bounded_labels_and_do_not_contain_payloads(tmp_path):
    options = dict(deploy=lambda inputs: inputs, name='echo', admin_token='metrics-admin-secret')
    first = Server(data_dir=tmp_path / 'one', **options)
    second = Server(data_dir=tmp_path / 'two', **options)
    with TestClient(first.app) as a, TestClient(second.app) as b:
        response = a.post('/v1/server/echo/deploy', json={'inputs': {'text': 'private-request-value'}})
        assert response.status_code == 200
        task_id = response.json()['task_id']
        for index in range(20):
            # Both modes resolve the entry schema before admission; rejected
            # names still retain bounded HTTP labels.
            assert a.post(f'/v1/server/unknown-{index}/deploy', json={}).status_code == 404
            assert a.post(f'/v1/server/unknown-{index}/deploy', json={'mode': 'async'}).status_code == 404
            assert a.get(f'/no-such-route-{index}').status_code == 404
            assert a.get(f'/v1/tasks/missing-{index}').status_code == 404
        result = a.get('/metrics', headers=ADMIN)
        records = samples(result)
        assert value(records, 'nndeploy_operation_seconds_count', operation='admission', entry='echo', outcome='success') == 1
        assert not any(row.name == 'nndeploy_operation_seconds_count'
                       and row.labels.get('operation') == 'admission'
                       and row.labels.get('entry') == '__unknown__' for row in records)
        assert value(records, 'nndeploy_http_request_seconds_count', method='POST', route='/v1/server/{name}/deploy', entry='__unknown__', status='404') == 40
        assert value(records, 'nndeploy_queue_wait_seconds_count', entry='echo') == 1
        for secret in ('private-request-value', task_id, 'metrics-admin-secret', 'unknown-19', 'missing-19', 'no-such-route-19', str(tmp_path)):
            assert secret not in result.text
        assert not any(row.name == 'nndeploy_operation_seconds_count' and row.labels.get('operation') == 'admission' for row in scrape(b))
        assert value(scrape(b), 'nndeploy_entry_accepting', entry='echo') == 1


def test_queue_gauges_show_slow_entry_without_blocking_other_entry(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked(inputs):
        entered.set()
        assert release.wait(10)
        return inputs

    server = Server(data_dir=tmp_path, admin_token='metrics-admin-secret')
    server.add(name='slow', deploy=blocked, config={'execution': {'max_queue_size': 1}})
    server.add(name='fast', deploy=lambda inputs: inputs)
    with TestClient(server.app) as client, ThreadPoolExecutor() as pool:
        running = pool.submit(client.post, '/v1/server/slow/deploy', json={})
        try:
            assert entered.wait(5)
            pending = client.post('/v1/server/slow/deploy', json={'mode': 'async'})
            assert pending.status_code == 202
            assert client.post('/v1/server/slow/deploy', json={}).status_code == 429
            assert client.post('/v1/server/fast/deploy', json={}).status_code == 200
            records = scrape(client)
            assert value(records, 'nndeploy_queue_tasks', entry='slow', state='running') == 1
            assert value(records, 'nndeploy_queue_tasks', entry='slow', state='pending') == 1
            assert value(records, 'nndeploy_inflight') == 2
            assert value(records, 'nndeploy_operation_seconds_count', operation='admission', entry='slow', outcome='queue_full') == 1
            assert value(records, 'nndeploy_entry_capacity', entry='slow') == 2
        finally:
            release.set()
        assert running.result().status_code == 200
        for _ in range(100):
            if client.get(pending.json()['status_url']).json()['state'] == 'succeeded':
                break
            time.sleep(.01)
        records = scrape(client)
        assert value(records, 'nndeploy_queue_tasks', entry='slow', state='running') == 0
        assert value(records, 'nndeploy_operation_seconds_count', operation='worker_request', entry='slow', outcome='success') == 2
        assert client.post('/api/server/slow/stop', headers=ADMIN).status_code == 200
        records = scrape(client)
        assert value(records, 'nndeploy_entry_accepting', entry='slow') == 0
        assert value(records, 'nndeploy_workers', entry='slow', state='current') == 0
        assert value(records, 'nndeploy_entry_accepting', entry='fast') == 1


def test_database_and_file_metrics_use_real_operations_and_sanitized_errors(db, tmp_path):
    server = Server(data_dir=tmp_path, admin_token='metrics-admin-secret', database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('private-filename.txt', b'private-content', 'text/plain')})
        identity = uploaded.json()['file_id']
        assert client.get(f'/v1/files/{identity}/content').content == b'private-content'
        assert client.delete('/v1/files/' + identity, headers=ADMIN).status_code == 200
        assert client.get(f'/v1/files/{identity}/content').status_code == 404
        with pytest.raises(DBAPIError), server.db.engine.connect() as connection:
            connection.execute(text('SELECT secret_column FROM nndeploy_no_such_private_table'))
        records = scrape(client)
        for operation in ('save', 'resolve', 'remove', 'recover_pending'):
            assert value(records, 'nndeploy_storage_operation_seconds_count', operation=operation, outcome='success') >= 1
        assert value(records, 'nndeploy_database_statement_seconds_count', outcome='success') > 0
        assert value(records, 'nndeploy_database_statement_seconds_count', outcome='error') == 1
        output = client.get('/metrics', headers=ADMIN).text
        for secret in ('private-filename', 'private-content', 'secret_column', 'nndeploy_no_such_private_table', identity):
            assert secret not in output


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_native_graph_and_failed_initialization_have_worker_metrics(tmp_path, isolation):
    def fail():
        raise ValueError('private-model-path')
    server = Server(name='echo', workflow=text_graph(), data_dir=tmp_path, admin_token='metrics-admin-secret',
                    config={'inputs': {'text': {'type': 'string', 'node': ['source'], 'required': False, 'default': 'hello'}},
                            'outputs': {'text': {'type': 'string', 'edge': 'text'}}, 'execution': {'isolation': isolation}})
    server.add(name='broken', handler_factory=fail)
    with TestClient(server.app) as client:
        assert client.post('/v1/server/echo/deploy', json={}).status_code == 200
        records = scrape(client)
        assert value(records, 'nndeploy_operation_seconds_count', entry='echo', operation='worker_initialize', outcome='success') == 1
        assert value(records, 'nndeploy_operation_seconds_count', entry='broken', operation='worker_initialize', outcome='error') == 1
        assert value(records, 'nndeploy_entry_accepting', entry='broken') == 0
        assert value(records, 'nndeploy_operation_seconds_count', entry='echo', operation='worker_request', outcome='success') == 1
        assert 'private-model-path' not in client.get('/metrics', headers=ADMIN).text


def shared_echo(inputs):
    return inputs


def test_shared_queue_metrics_use_existing_heartbeat_samples_without_scrape_io(backend, tmp_path, monkeypatch):
    db, redis_config = backend
    server = Server(name='echo', deploy=shared_echo, data_dir=tmp_path, admin_token='metrics-admin-secret',
                    redis_config=redis_config, database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        assert client.post('/v1/server/echo/deploy', json={}).status_code == 200
        for _ in range(100):
            records = scrape(client)
            if any(row.name == 'nndeploy_shared_inflight' and row.value == 0 for row in records):
                break
            time.sleep(.02)
        assert value(records, 'nndeploy_shared_inflight', entry='echo') == 0
        assert value(records, 'nndeploy_shared_usage_age_seconds', entry='echo') < 5
        assert not any(row.name == 'nndeploy_queue_tasks' for row in records)
        assert value(records, 'nndeploy_queue_wait_seconds_count', entry='echo') == 1
        assert value(records, 'nndeploy_redis_command_seconds_count', entry='echo', outcome='success') > 0
        # Scraping is still available with a failing dependency; the sample ages.
        def unavailable(*args, **kwargs):
            raise OSError('private-db-address')
        with monkeypatch.context() as patch:
            patch.setattr(server.db, 'queue_usage', unavailable)
            time.sleep(.25)
            response = client.get('/metrics', headers=ADMIN)
            assert response.status_code == 200 and 'private-db-address' not in response.text
            assert value(samples(response), 'nndeploy_shared_usage_age_seconds', entry='echo') >= .2


def test_cleanup_metrics_count_retries_without_exposing_file_identifiers(tmp_path, monkeypatch):
    from test_file_cleanup import expire
    server = Server(data_dir=tmp_path, admin_token='metrics-admin-secret')
    with TestClient(server.app) as client:
        identity = server.storage.save(io.BytesIO(b'expire me'), 'text/plain')['file_id']
        expire(server.db, identity)
        def unavailable(*args):
            raise OSError('private-storage-location')
        with monkeypatch.context() as patch:
            patch.setattr(server.storage, '_delete_content', unavailable)
            result = client.post('/api/server/cleanup-files', headers=ADMIN)
            assert result.status_code == 200 and result.json()['failed'] == 1
        records = scrape(client)
        assert value(records, 'nndeploy_file_cleanup_total', outcome='failed') == 1
        expire(server.db, identity)
        assert client.post('/api/server/cleanup-files', headers=ADMIN).json()['deleted'] == 1
        records = scrape(client)
        assert value(records, 'nndeploy_file_cleanup_total', outcome='deleted') == 1
        assert value(records, 'nndeploy_file_cleanup_total', outcome='failed') == 1
        assert identity not in client.get('/metrics', headers=ADMIN).text


def test_real_stream_reports_open_connections_and_separate_first_byte_duration(tmp_path):
    import httpx
    from test_streaming import live_server
    release = threading.Event()

    def stream(inputs):
        yield {'text': 'first'}
        assert release.wait(10)
        yield {'text': 'last'}

    server = Server(name='echo', deploy=stream, data_dir=tmp_path, admin_token='metrics-admin-secret',
                    config={'streaming': {'enabled': True}})
    with live_server(server) as address, httpx.Client(base_url=address, timeout=5) as client:
        try:
            with client.stream('POST', '/v1/server/echo/deploy', json={'mode': 'stream'}, headers={'Accept': 'text/event-stream'}) as response:
                assert response.status_code == 200
                lines = response.iter_lines()
                for line in lines:
                    if 'event: delta' in line:
                        break
                records = scrape(client)
                assert value(records, 'nndeploy_stream_connections') == 1
                assert value(records, 'nndeploy_queue_tasks', entry='echo', state='running') == 1
                # Closing the HTTP reader does not cancel actual computation.
            release.set()
            for _ in range(100):
                records = scrape(client)
                if value(records, 'nndeploy_stream_connections') == 0 and value(records, 'nndeploy_inflight') == 0:
                    break
                time.sleep(.02)
            assert value(records, 'nndeploy_stream_connections') == 0
            assert value(records, 'nndeploy_operation_seconds_count', operation='worker_request', entry='echo', outcome='success') == 1
            labels = dict(method='POST', route='/v1/server/{name}/deploy', entry='echo')
            assert value(records, 'nndeploy_http_first_byte_seconds_count', **labels) == 1
            assert value(records, 'nndeploy_http_request_seconds_count', status='200', **labels) == 1
        finally:
            release.set()


def test_terminal_counters_follow_acknowledged_commits_not_replays_or_rollbacks(db, tmp_path):
    server = Server(name='echo', deploy=lambda inputs: inputs, data_dir=tmp_path, admin_token='metrics-admin-secret',
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        key = {'Idempotency-Key': uuid.uuid4().hex}
        result = client.post('/v1/server/echo/deploy', json={}, headers=key)
        assert result.status_code == 200
        assert client.post('/v1/server/echo/deploy', json={}, headers=key).json() == result.json()
        assert value(scrape(client), 'nndeploy_tasks_terminal_total', entry='echo', state='succeeded') == 1
        identity = 'task_' + uuid.uuid4().hex
        server.db.create_task(identity, 'echo', time.time())
        with pytest.raises(ServerError):
            server.db.finish_task(identity, 'succeeded', time.time(), outputs={'file_id': 'file_missing'})
        assert server.db.get_task(identity)['state'] == 'pending'
        assert value(scrape(client), 'nndeploy_tasks_terminal_total', entry='echo', state='succeeded') == 1
        server.db.finish_task(identity, 'failed', time.time())
        with pytest.raises(RuntimeError):
            server.db.finish_task(identity, 'failed', time.time())
        assert value(scrape(client), 'nndeploy_tasks_terminal_total', entry='echo', state='failed') == 1


def test_pending_cancel_and_queue_expiry_are_counted_without_execution(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(inputs):
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return {}

    server = Server(name='echo', deploy=blocked, data_dir=tmp_path, admin_token='metrics-admin-secret',
                    config={'execution': {'queue_timeout_seconds': .2}})
    with TestClient(server.app) as client:
        try:
            first = client.post('/v1/server/echo/deploy', json={'mode': 'async'}).json()
            assert entered.wait(5)
            second = client.post('/v1/server/echo/deploy', json={'mode': 'async'}).json()
            third = client.post('/v1/server/echo/deploy', json={'mode': 'async'}).json()
            assert client.post(third['status_url'] + '/cancel').json()['state'] == 'cancelled'
            time.sleep(.25)
            release.set()
            for _ in range(100):
                if client.get(second['status_url']).json()['state'] == 'expired':
                    break
                time.sleep(.01)
            records = scrape(client)
            for state in ('succeeded', 'expired', 'cancelled'):
                assert value(records, 'nndeploy_tasks_terminal_total', entry='echo', state=state) == 1
            assert calls == [1]
        finally:
            release.set()

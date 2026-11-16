import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from nndeploy.server.errors import ServerError


@pytest.fixture
def receiver():
    messages = queue.Queue()
    statuses = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            messages.put((payload, self.headers.get('X-NNDeploy-Task-ID')))
            try:
                status = statuses.get_nowait()
            except queue.Empty:
                status = 204
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{http.server_port}/callback', messages, statuses
    http.shutdown()
    http.server_close()
    thread.join()


def wait_callback(client, identity, state='delivered'):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        result = client.get('/v1/tasks/' + identity).json()
        if result['callback']['state'] == state:
            return result
        time.sleep(.03)
    pytest.fail(f'Callback did not reach {state}: {result}')


def test_http_callback_retries_and_idempotency(tmp_path, receiver):
    url, messages, statuses = receiver
    statuses.put(503)
    server = Server(name='echo', deploy=lambda values: values, data_dir=tmp_path)
    with TestClient(server.app) as client:
        body = {'text': 'hello', 'mode': 'async', 'callback_url': url}
        response = client.post('/v1/server/echo/deploy', json=body, headers={'Idempotency-Key': 'callback'})
        assert response.status_code == 202, response.text
        identity = response.json()['task_id']
        record = wait_callback(client, identity)
        assert record['state'] == 'succeeded' and record['callback']['attempts'] == 2
        first, header = messages.get(timeout=1)
        second, _ = messages.get(timeout=1)
        assert first == second and header == identity
        assert first['event'] == 'task.finished' and first['outputs'] == {'text': 'hello'}
        assert 'inputs' not in first and 'callback_url' not in first
        same = client.post('/v1/server/echo/deploy', json=body, headers={'Idempotency-Key': 'callback'})
        assert same.json()['task_id'] == identity
        changed = client.post('/v1/server/echo/deploy', json=dict(body, callback_url=url + '/other'), headers={'Idempotency-Key': 'callback'})
        assert changed.status_code == 409
        assert messages.empty()


def test_failed_task_callback_and_invalid_url(tmp_path, receiver):
    url, messages, _ = receiver
    def fail(values):
        raise ServerError(422, 'bad_input', 'Invalid input')
    server = Server(name='fail', deploy=fail, data_dir=tmp_path)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/fail/deploy', json={'callback_url': url})
        assert response.status_code == 422
        identity = response.json()['error']['task_id']
        record = wait_callback(client, identity)
        assert record['state'] == 'failed'
        message, _ = messages.get(timeout=1)
        assert message['state'] == 'failed' and message['error']['code'] == 'bad_input'
        for invalid in ['file:///tmp/callback', 'http://user:password@127.0.0.1/', 'http://host/#fragment']:
            response = client.post('/v1/server/fail/deploy', json={'callback_url': invalid})
            assert response.status_code == 422


def test_completed_callback_survives_server_recreation(tmp_path, receiver):
    url, messages, _ = receiver
    first = Server(data_dir=tmp_path)
    first.db.create_task('task_callback_restart', 'echo', time.time(), {}, {},
                         callback_url=url, callback_scope=first.config_scope, retention_seconds=60)
    first.db.finish_task('task_callback_restart', 'cancelled', time.time(), error={'code': 'task_cancelled'})
    from sqlalchemy import update
    from nndeploy.server.db import tasks
    with first.db.engine.begin() as connection:
        connection.execute(update(tasks).where(tasks.c.task_id == 'task_callback_restart').values(purge_after=0))
    assert first.db.cleanup_tasks() == 0  # Keep result/files while delivery is pending.
    first.db.close()
    second = Server(data_dir=tmp_path)
    with TestClient(second.app) as client:
        record = wait_callback(client, 'task_callback_restart')
        assert record['state'] == 'cancelled'
        message, _ = messages.get(timeout=1)
        assert message['state'] == 'cancelled'


def test_batch_items_accept_callbacks(tmp_path, receiver):
    url, messages, _ = receiver
    server = Server(name='echo', deploy=lambda values: values, data_dir=tmp_path)
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/deploy/batch', json={'items': [{'text': 'batch', 'callback_url': url}]})
        assert response.status_code == 200, response.text
        message, _ = messages.get(timeout=5)
        assert message['outputs'] == {'text': 'batch'}


def test_callback_attempt_limit_does_not_change_task_result(tmp_path, monkeypatch):
    from sqlalchemy import update
    from nndeploy.server import callbacks
    from nndeploy.server.db import tasks
    server = Server(data_dir=tmp_path)
    server.db.create_task('task_callback_limit', 'echo', time.time(), {}, {},
                         callback_url='http://127.0.0.1/callback', callback_scope=server.config_scope)
    server.db.finish_task('task_callback_limit', 'succeeded', time.time(), outputs={'text': 'ok'})
    monkeypatch.setattr(callbacks, 'send', lambda record: 503)
    for attempt in range(1, 6):
        record = callbacks.claim(server.db, server.config_scope)
        assert record['callback_attempts'] == attempt
        assert callbacks.claim(server.db, server.config_scope) is None
        callbacks.deliver(server.db, record)
        with server.db.engine.begin() as connection:
            connection.execute(update(tasks).where(tasks.c.task_id == record['task_id']).values(callback_next_at=0))
    record = server.db.get_task('task_callback_limit')
    assert record['callback']['state'] == 'failed' and record['callback']['attempts'] == 5
    assert record['state'] == 'succeeded' and record['outputs'] == {'text': 'ok'}
    assert callbacks.claim(server.db, server.config_scope) is None
    server.db.close()

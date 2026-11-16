"""Existing editor history survives Server recreation in the same workspace."""
import json
import os
import queue
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from nndeploy.server.task_queue import ExecutionStatus
from nndeploy.server.errors import ServerError
from test_server_api import text_graph
from test_editor_queue_recovery import wait


URLS = ['sqlite'] + json.loads(os.environ.get('NNDEPLOY_TEST_DATABASE_URLS', '[]'))


@pytest.fixture(params=URLS, ids=lambda url: url.split(':')[0])
def location(request, tmp_path, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    return None if request.param == 'sqlite' else request.param


def editor(path, location):
    return Server(SimpleNamespace(resources=str(path), debug=True), database_url=location)


def test_editor_output_preview_is_durable_and_returned_by_exact_task_query(tmp_path, location):
    path = tmp_path / 'editor'
    server = editor(path, location)
    with TestClient(server.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        idx, _ = server.queue.get(timeout=1)
        results = {'sink': {'String': ['first', '中文'], 'Bool': [True], 'Num': [3.5]},
                   'another': {'String': ['second']}}
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), results, {'run_time': 1})
        expected = {'sink': 'first 中文 True 3.5', 'another': 'second'}
        results['sink']['String'][0] = 'mutated'
        assert client.get('/api/queue/task/' + identity).json()['result']['outputs'] == expected
    with TestClient(editor(path, location).app) as client:
        restored = client.get('/api/queue/task/' + identity)
        assert restored.status_code == 200
        assert restored.json()['result']['outputs'] == expected
    with TestClient(editor(tmp_path / 'other', location).app) as client:
        assert client.get('/api/queue/task/' + identity).status_code == 404


def test_worker_preserves_safe_preview_when_another_native_output_cannot_be_pickled():
    import pickle
    from nndeploy.server.worker import ensure_picklable
    from nndeploy.server.editor_outputs import preview
    class Native:
        def __str__(self):
            raise AssertionError('Native objects must not be stringified for preview')
        def __reduce__(self):
            raise TypeError('Native output cannot cross process IPC')
    value = ensure_picklable({'sink': {'String': ['kept', Native()], 'Tensor': Native()}})
    received = pickle.loads(pickle.dumps(value))
    assert preview(received) == ({'sink': 'kept'}, False)


def test_preview_limits_values_nodes_and_utf8_bytes_without_encoding_native_objects():
    from nndeploy.server.editor_outputs import memory_outputs, MAX_OUTPUT_BYTES, MAX_NODE_BYTES
    outputs, truncated = memory_outputs({f'node_{index}': {'String': ['中文' * 100000]} for index in range(300)})
    assert truncated and outputs
    assert sum(len(key.encode()) + len(value.encode()) for key, value in outputs.items()) <= MAX_OUTPUT_BYTES
    assert all(len(value.encode()) <= MAX_NODE_BYTES for value in outputs.values())
    outputs, truncated = memory_outputs({'sink': {'Num': list(range(2000))}})
    assert truncated and outputs['sink'].endswith('1023')


def test_truncation_flag_and_preview_survive_repeated_history_write(tmp_path, location, monkeypatch):
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        idx, _ = server.queue.get(timeout=1)
        original = server.queue.history_store.save
        def lost_receipt(*args):
            original(*args)
            raise OSError('Controlled lost history receipt')
        monkeypatch.setattr(server.queue.history_store, 'save', lost_receipt)
        results = {'sink': {'String': ['中文' * 100000]}}
        status = ExecutionStatus(True, 'Completed')
        with pytest.raises(OSError, match='Controlled lost'):
            server.queue.task_done(idx, status, results, {})
        expected = server.queue.history_store.get(identity)
        assert expected['outputs_truncated'] is True
        results['sink']['String'] = ['changed after receipt loss']
        monkeypatch.setattr(server.queue.history_store, 'save', original)
        server.queue.task_done(idx, status, results, {})
        actual = client.get('/api/queue/task/' + identity).json()['result']
        assert actual['outputs'] == expected['outputs']
        assert actual['outputs_truncated'] is True


def test_editor_terminal_history_survives_restart_and_is_workspace_scoped(tmp_path, location):
    path = tmp_path / 'editor'
    server = editor(path, location)
    with TestClient(server.app) as client:
        graph = text_graph()
        graph['node_repository_'][0]['str_'] = '中文' * 25000
        success = client.post('/api/queue', json=graph).json()['result']['task_id']
        idx, _ = server.queue.get(timeout=1)
        server.queue.mark_dispatched(idx)
        server.queue.mark_started(success, worker_pid=123)
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {'native': object()}, {'run_time': 2.5})
        cancelled = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        assert client.post('/api/queue/cancel/' + cancelled).json()['result']['state'] == 'CANCELLED'
        expected = client.get('/api/queue/history').json()['result']
        assert expected['total'] == 2
    recreated = editor(path, location)
    with TestClient(recreated.app) as client:
        assert client.get('/api/queue/history').json()['result'] == expected
        response = client.post('/api/queue/cancel/' + success)
        assert response.status_code == 200, response.text
        assert response.json()['result']['state'] == 'SUCCEEDED'
        assert client.get('/api/queue/history?max_items=1').json()['result']['items'][0]['task_id'] == cancelled
    other = editor(tmp_path / 'other', location)
    with TestClient(other.app) as client:
        assert client.get('/api/queue/history').json()['result'] == {'items': [], 'total': 0}


@pytest.mark.parametrize('committed', [False, True], ids=['uncommitted', 'receipt-lost'])
def test_editor_completion_history_failure_keeps_result_retryable(tmp_path, location, monkeypatch, committed):
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        idx, _ = server.queue.get(timeout=1)
        server.queue.mark_started(identity, worker_pid=123)
        store = server.queue.history_store
        original = store.save
        calls = []
        def fail(task_id, record):
            if committed:
                original(task_id, record)
            raise OSError('Controlled history commit failure')
        monkeypatch.setattr(store, 'save', fail)
        monkeypatch.setattr(server, 'notify_task_done', lambda *args: calls.append(args))
        status = ExecutionStatus(True, 'Completed')
        results, profile = {'value': 'kept'}, {'run_time': 2.5}
        with pytest.raises(OSError, match='Controlled history'):
            server.queue.task_done(idx, status, results, profile)
        assert server.queue.get_task_by_id(identity)['state'] == 'RUNNING'
        assert not calls
        first = store.get(identity)
        assert (first is not None) == committed
        # Callers may reuse these containers after a failed notification.
        status.completed = False
        results['value'] = 'changed'
        profile['run_time'] = 99
        monkeypatch.setattr(store, 'save', original)
        server.queue.task_done(idx, status, {'value': 'kept'}, {'run_time': 2.5})
        server.queue.task_done(idx, status, {'value': 'kept'}, {'run_time': 2.5})
        assert len(calls) == 1
        assert calls[0][1].completed is True
        assert calls[0][2:] == ({'value': 'kept'}, {'run_time': 2.5})
        record = store.get(identity)
        assert record['state'] == 'SUCCEEDED'
        if committed:
            assert record == first
        different = dict(record, state='FAILED')
        with pytest.raises(ServerError, match='different recorded result'):
            store.save(identity, different)
        assert store.get(identity) == record


@pytest.mark.parametrize('committed', [False, True], ids=['uncommitted', 'receipt-lost'])
def test_editor_pending_cancellation_history_failure_keeps_task_retryable(tmp_path, location, monkeypatch, committed):
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        store = server.queue.history_store
        original = store.save
        def fail(task_id, record):
            if committed:
                original(task_id, record)
            raise OSError('Controlled cancellation history failure')
        monkeypatch.setattr(store, 'save', fail)
        with pytest.raises(OSError, match='Controlled cancellation'):
            server.queue.cancel_editor_task(identity)
        assert server.queue.get_task_by_id(identity)['state'] == 'PENDING'
        assert server.queue.counts()['pending'] == 1
        assert server.queue.get(timeout=.01) is None, 'An unconfirmed cancellation must not run'
        first = store.get(identity)
        monkeypatch.setattr(store, 'save', original)
        assert server.queue.cancel_editor_task(identity)['state'] == 'CANCELLED'
        assert server.queue.counts()['pending'] == 0
        if committed:
            assert store.get(identity) == first


def test_editor_result_collector_retries_database_failure_without_result_queue_replay(tmp_path, location, monkeypatch):
    from nndeploy.server.app import start_finisher
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        idx, _ = server.queue.get(timeout=1)
        original = server.queue.history_store.save
        attempted, release, stop = threading.Event(), threading.Event(), threading.Event()
        def save(task_id, record):
            if not release.is_set():
                attempted.set()
                raise OSError('History temporarily unavailable')
            return original(task_id, record)
        monkeypatch.setattr(server.queue.history_store, 'save', save)
        results = queue.Queue()
        results.put((idx, ExecutionStatus(True, 'Completed'), {'value': 'one'}, {'run_time': 1.0}))
        thread = start_finisher(server.queue, results, stop)
        try:
            assert attempted.wait(3)
            assert results.empty()
            # A later Worker exit cannot replace the already-known result.
            with pytest.raises(OSError):
                server.queue.task_done(idx, ExecutionStatus(False, 'Worker exited'), {}, {})
            release.set()
            wait(lambda: server.queue.get_task_by_id(identity)['state'] == 'SUCCEEDED')
            assert server.queue.get_history()[identity]['results'] == {'value': 'one'}
            assert server.queue._pending_completions == {}
        finally:
            release.set()
            stop.set()
            thread.join(timeout=3)
            assert not thread.is_alive()


def test_editor_legacy_flush_persists_pending_and_dispatched_cancellations(tmp_path, location):
    jobs = queue.Queue()
    server = editor(tmp_path / 'editor', location)
    server.queue._job_q = jobs
    with TestClient(server.app) as client:
        identities = [client.post('/api/queue', json=text_graph()).json()['result']['task_id'] for _ in range(2)]
        idx, payload = server.queue.get(timeout=1)
        server.queue.dispatch(idx, payload, jobs)
        assert server.queue.flush() == {'cleared_pending': 1, 'drained_job_q': 1}
    restored = editor(tmp_path / 'editor', location)
    with TestClient(restored.app) as client:
        records = client.get('/api/queue/history').json()['result']['items']
        assert {record['task_id'] for record in records} == set(identities)
        assert all(record['state'] == 'CANCELLED' for record in records)


def test_editor_status_query_recovers_exact_task_without_native_results(tmp_path, location):
    path = tmp_path / 'editor'
    server = editor(path, location)
    class NativeResult:
        def __deepcopy__(self, memo):
            raise TypeError('Native outputs cannot be copied')
    with TestClient(server.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        url = '/api/queue/task/' + identity
        response = client.get(url)
        assert response.status_code == 200, response.text
        assert response.json()['result']['state'] == 'PENDING'
        idx, _ = server.queue.get(timeout=1)
        server.queue.mark_dispatched(idx)
        assert client.get(url).json()['result']['state'] == 'DISPATCHED'
        server.queue.mark_started(identity, worker_pid=123)
        assert client.get(url).json()['result']['state'] == 'RUNNING'
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {'run_time': 2.5})
        # Metadata lookup must never serialize/copy a live native output.
        server.queue._hist[identity]['results'] = {'native': NativeResult()}
        expected = client.get(url).json()['result']
        assert expected['state'] == 'SUCCEEDED'
        assert expected['task_id'] == identity
        assert expected['time_profile'] == {'run_time': 2.5}
        assert 'results' not in expected and 'task' not in expected
        assert client.get('/api/queue/task/missing').status_code == 404
    with TestClient(editor(path, location).app) as client:
        assert client.get(url).json()['result'] == expected
    with TestClient(editor(tmp_path / 'other', location).app) as client:
        assert client.get(url).status_code == 404


def test_history_summary_pages_are_stable_and_do_not_load_native_results(tmp_path, location, monkeypatch):
    server = editor(tmp_path / 'editor', location)
    identities = []
    with TestClient(server.app) as client:
        clock = server.db.clock
        monkeypatch.setattr(server.db, 'clock', lambda connection=None: 1234.5)
        for index in range(5):
            graph = text_graph()
            graph['name_'] = 'History example'
            graph['node_repository_'][0]['str_'] = 'PRIVATE INPUT NOT NEEDED FOR LIST'
            identity = client.post('/api/queue', json=graph).json()['result']['task_id']
            idx, _ = server.queue.get(timeout=1)
            server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {'sink': {'String': [str(index)]}}, {})
            identities.append(identity)
        class Native:
            def __deepcopy__(self, memo):
                raise AssertionError('History listing must not copy native results')
        for value in server.queue._hist.values():
            value['results'] = Native()
        page = client.get('/api/queue/history?summary=true&max_items=2')
        assert page.status_code == 200, page.text
        first = page.json()['result']['items']
        expected = sorted(identities, reverse=True)
        assert [item['task_id'] for item in first] == expected[:2]
        assert all(item['name'] == 'History example' and 'task' not in item and 'outputs' not in item for item in first)
        assert 'PRIVATE INPUT' not in page.text
        # A new completion cannot shift the continuation boundary like offsets do.
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        idx, _ = server.queue.get(timeout=1)
        monkeypatch.setattr(server.db, 'clock', clock)
        server.queue.task_done(idx, ExecutionStatus(True, 'New completion'), {}, {})
        second = client.get('/api/queue/history', params={'summary': True, 'max_items': 2, 'before': first[-1]['task_id']})
        assert [item['task_id'] for item in second.json()['result']['items']] == expected[2:4]
        # Legacy clients still receive their existing workflow payload shape.
        old = client.get('/api/queue/history?max_items=2').json()['result']['items']
        assert 'task' in old[0] and 'name' not in old[0]
        missing = client.get('/api/queue/history?summary=true&before=unknown')
        assert missing.status_code == 404
    with TestClient(editor(tmp_path / 'editor', location).app) as client:
        last = client.get('/api/queue/history', params={'summary': True, 'max_items': 2, 'before': expected[3]})
        assert [item['task_id'] for item in last.json()['result']['items']] == expected[4:]
        detail = client.get('/api/queue/task/' + expected[-1]).json()['result']
        assert detail['state'] == 'SUCCEEDED' and detail['outputs']['sink']
    with TestClient(editor(tmp_path / 'other', location).app) as client:
        assert client.get('/api/queue/history?summary=true').json()['result']['items'] == []
        assert client.get('/api/queue/history', params={'before': expected[0]}).status_code == 404

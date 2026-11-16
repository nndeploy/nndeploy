"""The editor's existing cancel route must reflect actual task execution."""
import queue
from contextlib import contextmanager
import logging
from logging.handlers import QueueListener
import multiprocessing
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from nndeploy.server import app
from nndeploy.server.app import _EditorJobQueue
from nndeploy.server.task_queue import ExecutionStatus
from test_server_api import text_graph


@pytest.fixture
def editor(tmp_path, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    jobs = _EditorJobQueue()
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), jobs, queue.Queue(), queue.Queue())
    with TestClient(server.app) as client:
        yield server, jobs, client


@pytest.mark.parametrize('stage', ['pending', 'dispatching', 'dispatched', 'delivered'])
def test_editor_cancel_before_execution_never_reaches_a_worker(editor, stage):
    server, jobs, client = editor
    reader = jobs.new_reader()
    response = client.post('/api/queue', json=text_graph())
    task_id = response.json()['result']['task_id']
    if stage != 'pending':
        idx, payload = server.queue.get()
        if stage != 'dispatching':
            jobs.put((idx, payload))
            server.queue.mark_dispatched(idx)
        if stage == 'delivered':
            assert jobs.get(reader=reader, timeout=0)[0] == idx
    response = client.post('/api/queue/cancel/' + task_id)
    assert response.status_code == 200, response.text
    assert server.queue.get_task_by_id(task_id)['state'] == 'CANCELLED'
    assert response.json()['result'] == dict(task_id=task_id, state='CANCELLED', cancel_requested=True)
    with pytest.raises(queue.Empty):
        jobs.get(reader=reader, timeout=0)
    if stage == 'delivered':
        assert not jobs.claim(idx, reader)
    assert client.post('/api/queue/cancel/' + task_id).json() == response.json()


def test_editor_running_cancel_is_a_request_until_computation_finishes(editor):
    server, jobs, client = editor
    task_id = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
    idx, payload = server.queue.get()
    reader = jobs.new_reader()
    jobs.put((idx, payload))
    jobs.get(reader=reader, timeout=0)
    assert jobs.claim(idx, reader)
    server.queue.mark_started(task_id, worker_pid=123)
    response = client.post('/api/queue/cancel/' + task_id)
    assert response.status_code == 200, response.text
    assert response.json()['result'] == dict(task_id=task_id, state='RUNNING', cancel_requested=True)
    assert task_id not in server.queue.get_history()
    assert server.queue.get_task_by_id(task_id)['state'] == 'RUNNING'
    assert jobs.cancellation_requested(idx, reader)


def test_editor_cancel_preserves_completed_results_and_rejects_unknown_tasks(editor):
    server, jobs, client = editor
    assert client.post('/api/queue/cancel/missing').status_code == 404
    task_id = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
    idx, _ = server.queue.get()
    server.queue.task_done(idx, ExecutionStatus(True), {'result': 'kept'}, {})
    response = client.post('/api/queue/cancel/' + task_id)
    assert response.status_code == 200, response.text
    assert response.json()['result'] == dict(task_id=task_id, state='SUCCEEDED', cancel_requested=False)
    assert server.queue.get_task_by_id(task_id)['results'] == {'result': 'kept'}


def test_editor_cancel_after_computation_preserves_result_waiting_for_notification(editor):
    server, jobs, client = editor
    identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
    idx, payload = server.queue.get()
    reader = jobs.new_reader()
    jobs.put((idx, payload))
    jobs.get(reader=reader, timeout=0)
    assert jobs.claim(idx, reader)
    assert jobs.finish(idx, reader, ExecutionStatus(True), {'value': 'complete'}, {})
    response = client.post('/api/queue/cancel/' + identity)
    assert response.status_code == 200, response.text
    assert response.json()['result'] == dict(task_id=identity, state='SUCCEEDED', cancel_requested=False)
    assert server.queue.get_task_by_id(identity)['results'] == {'value': 'complete'}
    jobs.retire_reader(reader)
    assert jobs.outcomes() == []


def test_editor_cancel_reads_its_completed_result_even_with_a_notification_backlog(editor):
    server, jobs, client = editor
    reader = jobs.new_reader()
    for index in range(20):
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        idx, payload = server.queue.get()
        jobs.put((idx, payload))
        server.queue.mark_dispatched(idx)
        jobs.get(reader=reader, timeout=0)
        assert jobs.claim(idx, reader)
        assert jobs.finish(idx, reader, ExecutionStatus(True), {'value': index}, {})
    response = client.post('/api/queue/cancel/' + identity)
    assert response.status_code == 200, response.text
    assert response.json()['result'] == dict(task_id=identity, state='SUCCEEDED', cancel_requested=False)
    assert server.queue.get_task_by_id(identity)['results'] == {'value': 19}


def test_editor_cancel_while_dispatch_waits_for_capacity_cannot_publish_later(editor):
    server, jobs, client = editor
    jobs.maxsize = 1
    jobs.put((-1, {'id': 'occupying-capacity'}))
    identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
    stop = threading.Event()
    thread = app.start_scheduler(server.queue, jobs, stop)
    try:
        eventually(lambda: any('idx' in row for row in server.queue.get_current_queue()['PENDING']))
        assert client.post('/api/queue/cancel/' + identity).json()['result']['state'] == 'CANCELLED'
        assert jobs.get_nowait()[0] == -1
        next_id = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        assert jobs.get(timeout=3)[1]['id'] == next_id
        assert jobs.empty()
    finally:
        stop.set()
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_editor_flush_cancels_pending_deliveries_but_keeps_claimed_execution(editor):
    server, jobs, client = editor
    reader = jobs.new_reader()
    identities, indices = [], []
    for stage in ('running', 'delivered', 'queued', 'pending'):
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        identities.append(identity)
        if stage != 'pending':
            idx, payload = server.queue.get()
            indices.append(idx)
            jobs.put((idx, payload))
            server.queue.mark_dispatched(idx)
            if stage in ('running', 'delivered'):
                jobs.get(reader=reader, timeout=0)
            if stage == 'running':
                assert jobs.claim(idx, reader)
    # Deliberately omit the running progress notification: claim is authoritative.
    response = client.post('/api/queue/flush')
    assert response.status_code == 200, response.text
    assert response.json()['result'] == dict(cleared_pending=1, drained_job_q=2)
    assert identities[0] not in server.queue.get_history()
    assert not jobs.cancellation_requested(indices[0], reader)
    assert not jobs.claim(indices[1], reader)
    for identity in identities[1:]:
        assert server.queue.get_task_by_id(identity)['state'] == 'CANCELLED'
    assert jobs.empty()


@pytest.mark.parametrize('dispatched', [False, True])
def test_editor_notification_failure_does_not_change_confirmed_cancellation(editor, monkeypatch, dispatched):
    server, jobs, client = editor
    def broken(*args):
        raise OSError('Controlled WebSocket notification failure')
    monkeypatch.setattr(server, 'notify_task_done', broken)
    identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
    if dispatched:
        idx, payload = server.queue.get()
        jobs.put((idx, payload))
        server.queue.mark_dispatched(idx)
    response = client.post('/api/queue/cancel/' + identity)
    assert response.status_code == 200, response.text
    assert response.json()['result']['state'] == 'CANCELLED'
    assert jobs.outcomes() == []


HOLD_PLUGIN = '''
import time
from pathlib import Path
from nndeploy.dag import Node, NodeCreator, register_node
from nndeploy.base import Status
class Hold(Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('test.EditorHold')
        self.set_input_type(str)
    def run(self):
        root = Path(self.get_input(0).get(self))
        with (root / 'entered').open('a') as stream:
            stream.write('run\\n')
        deadline = time.monotonic() + 20
        while not (root / 'release').exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Test release was not supplied')
            time.sleep(.01)
        (root / 'exited').touch()
        return Status.ok()
class Creator(NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = Hold(name, inputs, outputs)
        return self.node
creator = Creator()
register_node('test.EditorHold', creator)
'''


def eventually(check, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError('Expected editor state was not observed')


@contextmanager
def running_editor(tmp_path, monkeypatch, *, collect=True, managed=True):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', .02)
    monkeypatch.setattr(app, 'WORKER_MONITOR_INTERVAL_SECONDS', .02)
    context = multiprocessing.get_context('spawn')
    monkeypatch.setattr(app.mp, 'Process', context.Process)
    plugin_dir = tmp_path / 'plugin'
    plugin_dir.mkdir()
    (plugin_dir / 'editor_hold.py').write_text(HOLD_PLUGIN)
    cancelled = threading.Event()
    class Capture(logging.Handler):
        def emit(self, record):
            if 'cancel requested' in record.getMessage():
                cancelled.set()
    with app._EditorManager(ctx=context) as manager:
        jobs = manager.JobQueue() if managed else manager.Queue()
        results, progress, logs, plugins, cancels = manager.Queue(1), manager.Queue(), manager.Queue(), manager.Queue(), manager.Queue()
        if not collect:
            results.put((-1, ExecutionStatus(True), {}, {}))
        server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), jobs, plugins, cancels)
        stop = threading.Event()
        file_log = logging.FileHandler(tmp_path / 'worker.log')
        listener = QueueListener(logs, Capture(), file_log)
        listener.start()
        threads = []
        worker = app.start_worker(jobs, results, progress, logs, plugins, cancels, str(tmp_path))
        monitor = threading.Thread(target=app.monitor_worker,
            args=(worker, jobs, results, progress, logs, plugins, cancels, str(tmp_path), server, stop))
        monitor.start()
        with TestClient(server.app) as client:
            try:
                threads.append(app.start_scheduler(server.queue, jobs, stop))
                threads.append(app.start_progress_listener(server, progress, stop))
                if collect:
                    threads.append(app.start_finisher(server.queue, results, stop))
                yield SimpleNamespace(server=server, client=client, jobs=jobs, worker=worker,
                                      results=results, cancels=cancels, stop=stop, threads=threads, cancelled=cancelled)
            finally:
                stop.set()
                monitor.join(timeout=8)
                for thread in threads:
                    thread.join(timeout=3)
                assert not monitor.is_alive()
                assert not any(thread.is_alive() for thread in threads)
                listener.stop()
                file_log.close()


def submit(runtime, graph):
    response = runtime.client.post('/api/queue', json=graph)
    assert response.status_code == 202, response.text
    return response.json()['result']['task_id']


def held_graph(directory):
    directory.mkdir()
    graph = text_graph()
    graph['node_repository_'][0]['str_'] = str(directory)
    graph['node_repository_'][1]['key_'] = 'test.EditorHold'
    return graph


def terminal(runtime, task_id):
    return eventually(lambda: runtime.server.queue.get_history().get(task_id))


def entered(runtime, task_id, control):
    eventually(lambda: (control / 'entered').exists() or runtime.server.queue.get_history().get(task_id))
    assert (control / 'entered').exists(), (runtime.server.queue.get_history(), (control.parent / 'worker.log').read_text())


@pytest.mark.parametrize('target', ['queued', 'running'])
@pytest.mark.parametrize('managed', [True, False], ids=['editor', 'legacy-queues'])
def test_editor_real_graph_cancellation_keeps_execution_owned_until_return(tmp_path, monkeypatch, target, managed):
    with running_editor(tmp_path, monkeypatch, managed=managed) as runtime:
        control = tmp_path / 'control'
        held = submit(runtime, held_graph(control))
        entered(runtime, held, control)
        eventually(lambda: runtime.server.queue.get_task_by_id(held)['state'] == 'RUNNING')
        fast = text_graph()
        fast['node_repository_'][1]['frontend_params'].update(save_to_file=True, file_path=str(tmp_path / 'fast.txt'))
        queued = submit(runtime, fast)
        identity = queued if target == 'queued' else held
        response = runtime.client.post('/api/queue/cancel/' + identity)
        assert response.status_code == 200, response.text
        if target == 'queued':
            if managed:
                assert terminal(runtime, queued)['state'] == 'CANCELLED'
            else:
                assert response.json()['result']['cancel_requested']
                # The busy Worker consumes this other task's signal first; it
                # must retain it and skip computation when that task arrives.
                eventually(lambda: runtime.cancels.empty())
                assert queued not in runtime.server.queue.get_history()
        else:
            assert response.json()['result']['cancel_requested']
            assert runtime.cancelled.wait(5)
            assert held not in runtime.server.queue.get_history()
            assert runtime.server.queue.get_task_by_id(held)['state'] == 'RUNNING'
            assert not (control / 'exited').exists()
        (control / 'release').touch()
        assert terminal(runtime, held)['state'] == ('CANCELLED' if target == 'running' else 'SUCCEEDED')
        assert terminal(runtime, queued)['state'] == ('CANCELLED' if target == 'queued' else 'SUCCEEDED')
        assert (tmp_path / 'fast.txt').exists() == (target == 'running')
        assert (control / 'entered').read_text() == 'run\n'
        followup = submit(runtime, text_graph())
        assert terminal(runtime, followup)['state'] == 'SUCCEEDED'


@pytest.mark.parametrize('cancel_first', [False, True])
def test_editor_worker_crash_finishes_owned_task_once_and_preserves_pending_work(tmp_path, monkeypatch, cancel_first):
    with running_editor(tmp_path, monkeypatch) as runtime:
        control = tmp_path / 'control'
        held = submit(runtime, held_graph(control))
        entered(runtime, held, control)
        queued = submit(runtime, text_graph())
        if cancel_first:
            assert runtime.client.post('/api/queue/cancel/' + held).status_code == 200
        runtime.worker.kill()
        record = terminal(runtime, held)
        assert record['state'] == ('CANCELLED' if cancel_first else 'FAILED')
        assert record['results'] == {}
        assert terminal(runtime, queued)['state'] == 'SUCCEEDED'
        assert (control / 'entered').read_text() == 'run\n'
        assert not (control / 'exited').exists()
        assert runtime.jobs.outcomes() == []


def test_editor_completed_result_survives_worker_crash_before_parent_notification(tmp_path, monkeypatch):
    with running_editor(tmp_path, monkeypatch, collect=False) as runtime:
        graph = text_graph()
        output = tmp_path / 'once.txt'
        graph['node_repository_'][0]['str_'] = 'once'
        graph['node_repository_'][1]['frontend_params'].update(save_to_file=True, file_path=str(output))
        identity = submit(runtime, graph)
        eventually(lambda: runtime.jobs.outcomes())
        runtime.worker.kill()
        runtime.threads.append(app.start_finisher(runtime.server.queue, runtime.results, runtime.stop))
        assert terminal(runtime, identity)['state'] == 'SUCCEEDED'
        assert terminal(runtime, submit(runtime, text_graph()))['state'] == 'SUCCEEDED'
        assert output.read_text() == 'once\n'
        eventually(lambda: not runtime.jobs.outcomes())

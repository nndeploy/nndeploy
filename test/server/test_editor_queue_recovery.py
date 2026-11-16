"""Recover editor IPC and helper failures without losing control of computation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import requests

from test_cli import unused_port
from test_editor_task_control import HOLD_PLUGIN, held_graph
from test_server_api import text_graph


LAUNCH = '''
import json, os, threading, time
from pathlib import Path
from nndeploy.server import app
from nndeploy.server.template import WorkflowTemplateManager
root = Path(os.environ['NNDEPLOY_EDITOR_RECOVERY_TEST'])
WorkflowTemplateManager.init_templates = staticmethod(lambda: root)
# Observe successful application cleanup directly. A diagnostic queued just
# before Uvicorn re-raises SIGTERM is not a durable shutdown acknowledgement.
original_server_close = app.NnDeployServer.close
async def close_server(self):
    await original_server_close(self)
    (root / 'server-closed').touch()
app.NnDeployServer.close = close_server
records = {'managers': [], 'workers': []}
lock = threading.Lock()
def record(kind, pid):
    with lock:
        records[kind].append(pid)
        temporary = root / 'pids.tmp'
        temporary.write_text(json.dumps(records))
        temporary.replace(root / 'pids.json')
# Hold the test inspection window before the actual IPC startup deadline.
original_channels = app._EditorRuntime._channels
def channels(self):
    if records['managers']:
        (root / 'restarting').touch()
        deadline = time.monotonic() + 30
        while not (root / 'resume').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Test restart release missing')
            time.sleep(.02)
    return original_channels(self)
app._EditorRuntime._channels = channels
original_manager = app._EditorManager.start
failed = False
def start_manager(self, *args, **kwargs):
    global failed
    if records['managers'] and os.environ.get('NNDEPLOY_EDITOR_RESTART_FAILURE') == 'manager' and not failed:
        failed = True
        raise OSError('Controlled replacement manager startup failure')
    original_manager(self, *args, **kwargs)
    record('managers', self._process.pid)
app._EditorManager.start = start_manager
original_jobs = app._EditorManager.JobQueue
def jobs(self, *args, **kwargs):
    return original_jobs(self, maxsize=1)
app._EditorManager.JobQueue = jobs
original_worker = app.start_worker
def worker(*args, **kwargs):
    global failed
    if records['workers'] and os.environ.get('NNDEPLOY_EDITOR_RESTART_FAILURE') == 'worker' and not failed:
        failed = True
        raise OSError('Controlled replacement Worker startup failure')
    process = original_worker(*args, **kwargs)
    record('workers', process.pid)
    return process
app.start_worker = worker
original_close = app._close_worker
def close(worker):
    if (worker is not None and not getattr(worker, '_closed', False)
            and records['workers'] and worker.pid == records['workers'][0]
            and os.environ.get('NNDEPLOY_EDITOR_HOLD_EXIT') == '1'):
        (root / 'closing').touch()
        deadline = time.monotonic() + 30
        while not (root / 'close-resume').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Test close release missing')
            time.sleep(.02)
    original_close(worker)
app._close_worker = close
# End exactly one owned helper without stopping the runtime or its manager.
# A private stop view drives the real helper loop through an exit or exception.
fault = os.environ.get('NNDEPLOY_EDITOR_HELPER_EXIT', 'manager')
if os.environ.get('NNDEPLOY_EDITOR_HOLD_RESULT') == '1':
    original_finish = app.TaskQueue.finish_result
    def finish_result(self, result):
        if len(records['managers']) == 1 and threading.current_thread().name == 'FinisherThread':
            (root / 'result-waiting').touch()
            raise RuntimeError('Controlled result notification failure')
        return original_finish(self, result)
    app.TaskQueue.finish_result = finish_result
class HelperStop:
    def __init__(self, stop):
        self.stop = stop
    def is_set(self):
        if fault == 'scheduler' and (root / 'exit-helper').exists():
            raise RuntimeError('Controlled scheduler failure')
        return self.stop.is_set() or (root / 'exit-helper').exists()
    def wait(self, timeout=None):
        return self.is_set() or self.stop.wait(timeout)
helper_names = {'scheduler': 'start_scheduler', 'finisher': 'start_finisher',
                'progress': 'start_progress_listener', 'monitor': 'monitor_worker'}
if fault in helper_names:
    original_helper = getattr(app, helper_names[fault])
    helper_started = False
    def helper(*args, **kwargs):
        global helper_started
        if helper_started:
            return original_helper(*args, **kwargs)
        helper_started = True
        args = (*args[:-1], HelperStop(args[-1]))
        result = original_helper(*args, **kwargs)
        if fault == 'monitor':
            (root / 'helper-exited').touch()
        else:
            def observe():
                result.join()
                (root / 'helper-exited').touch()
            threading.Thread(target=observe, daemon=True).start()
        return result
    setattr(app, helper_names[fault], helper)
elif fault == 'logger':
    original_logger = app.configure_root_logger
    def logger(*args):
        listener = original_logger(*args)
        if len(records['managers']) == 1:
            def end_listener():
                thread = listener._thread
                while thread.is_alive() and not (root / 'exit-helper').exists():
                    time.sleep(.02)
                if thread.is_alive():
                    listener.enqueue_sentinel()
                    thread.join()
                    (root / 'helper-exited').touch()
            threading.Thread(target=end_listener, daemon=True).start()
        return listener
    app.configure_root_logger = logger
if os.environ.get('NNDEPLOY_EDITOR_RESTART_FAILURE') == 'history':
    from nndeploy.server.db import _EditorHistory
    from sqlalchemy.exc import SQLAlchemyError
    original_save = _EditorHistory.save
    def save(self, *args, **kwargs):
        if threading.current_thread().name == 'EditorQueueMonitorThread' and not (root / 'database-resume').exists():
            with (root / 'database-failed').open('a') as attempts:
                attempts.write('attempt\\n')
            raise SQLAlchemyError('Controlled history write failure during recovery')
        return original_save(self, *args, **kwargs)
    _EditorHistory.save = save
app.EDITOR_IPC_TIMEOUT_SECONDS = 2
app.WORKER_MONITOR_INTERVAL_SECONDS = .02
app.WORKER_RESTART_DELAY_SECONDS = .02
app.main()
'''


def wait(check, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.03)
    raise AssertionError('Editor did not reach the expected recovery state')


def live(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX process signals')
@pytest.mark.parametrize('fault,cancel,restart_failure,known_result',
    [('manager', cancel, failure, False) for cancel in (False, True) for failure in ('none', 'manager', 'worker')]
    + [(helper, False, 'none', False) for helper in ('scheduler', 'finisher', 'progress', 'monitor', 'logger')]
    + [('finisher', False, 'none', True), ('finisher', False, 'history', True),
       ('stalled-manager', False, 'none', False)])
def test_editor_recovers_after_runtime_failure_without_reexecuting_uncertain_tasks(tmp_path, fault, cancel, restart_failure, known_result):
    resources = tmp_path / 'resources'
    plugin = resources / 'plugin'
    plugin.mkdir(parents=True)
    (plugin / 'editor_hold.py').write_text(HOLD_PLUGIN)
    address = f'http://127.0.0.1:{unused_port()}'
    logs = tmp_path / 'console.log'
    def pids():
        path = tmp_path / 'pids.json'
        return json.loads(path.read_text()) if path.exists() else dict(managers=[], workers=[])
    with logs.open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', LAUNCH, '--host', '127.0.0.1',
            '--port', address.rsplit(':', 1)[1], '--debug', '--resources', str(resources),
            '--log', str(tmp_path / 'editor.log')],
            env=dict(os.environ, NNDEPLOY_EDITOR_RECOVERY_TEST=str(tmp_path),
                     NNDEPLOY_EDITOR_RESTART_FAILURE=restart_failure,
                     NNDEPLOY_EDITOR_HELPER_EXIT=fault,
                     NNDEPLOY_EDITOR_HOLD_RESULT='1' if known_result else '0',
                     NNDEPLOY_EDITOR_HOLD_EXIT='1' if cancel else '0'), stdout=log, stderr=log)
        def ready():
            assert process.poll() is None, logs.read_text()
            try:
                return requests.get(address + '/health/ready', timeout=.5).status_code == 200
            except requests.RequestException:
                return False
        def submit(graph):
            response = requests.post(address + '/api/queue', json=graph, timeout=3)
            assert response.status_code == 202, response.text
            return response.json()['result']['task_id']
        def history():
            response = requests.get(address + '/api/queue/history', timeout=3)
            assert response.status_code == 200, response.text
            return {record['task_id']: record for record in response.json()['result']['items']}
        try:
            wait(ready, 30)
            initial = pids()
            completed = []
            if known_result:
                for index in range(18 if restart_failure == 'history' else 1):
                    control = tmp_path / f'completed-{index}'
                    completed_graph = held_graph(control)
                    (control / 'release').touch()
                    completed.append(submit(completed_graph))
                    wait(lambda: (control / 'exited').exists())
                wait(lambda: (tmp_path / 'result-waiting').exists())
                assert not set(completed).intersection(history())
            held = submit(held_graph(tmp_path / 'control'))
            wait(lambda: (tmp_path / 'control' / 'entered').exists())
            identities = [held]
            for index in range(3):
                graph = text_graph()
                graph['node_repository_'][0]['str_'] = f'queued-{index}'
                graph['node_repository_'][1]['frontend_params'].update(
                    save_to_file=True, file_path=str(tmp_path / f'queued-{index}.txt'))
                identities.append(submit(graph))
            def has_unpublished_task():
                rows = requests.get(address + '/api/queue/info', timeout=3).json()['result']['pending']
                return any(row['task']['id'] == identities[-1] and 'idx' not in row for row in rows)
            wait(has_unpublished_task)
            if fault in ('manager', 'stalled-manager'):
                os.kill(initial['managers'][0], signal.SIGSTOP if fault == 'stalled-manager' else signal.SIGKILL)
            else:
                assert live(initial['managers'][0])
                (tmp_path / 'exit-helper').touch()
                wait(lambda: (tmp_path / 'helper-exited').exists())
                wait(lambda: requests.get(address + '/health/ready', timeout=2).status_code == 503, 2)
            if cancel:
                wait(lambda: (tmp_path / 'closing').exists())
                assert live(initial['workers'][0])
                response = requests.post(address + '/api/queue/cancel/' + held, timeout=2)
                assert response.status_code == 200, response.text
                assert response.json()['result']['state'] in ('DISPATCHED', 'RUNNING')
                assert response.json()['result']['cancel_requested'] is True
                assert held not in history()
                (tmp_path / 'close-resume').touch()
            if restart_failure == 'history':
                wait(lambda: (tmp_path / 'database-failed').exists()
                     and len((tmp_path / 'database-failed').read_text().splitlines()) >= 2)
                assert not live(initial['workers'][0])
                assert live(initial['managers'][0]), 'Retain cached results until history commits'
                assert not (tmp_path / 'restarting').exists()
                (tmp_path / 'database-resume').touch()
            wait(lambda: (tmp_path / 'restarting').exists(), 8)
            assert requests.get(address + '/health/live', timeout=2).status_code == 200
            assert requests.get(address + '/health/ready', timeout=2).status_code == 503
            response = requests.post(address + '/api/queue', json=text_graph(), timeout=2)
            assert response.status_code == 503, response.text
            wait(lambda: all(identity in history() for identity in identities[:-1]))
            assert history()[held]['state'] == ('CANCELLED' if cancel else 'FAILED')
            assert all(history()[identity]['state'] == 'FAILED' for identity in identities[1:-1])
            assert identities[-1] not in history()
            for index, completed_id in enumerate(completed):
                assert history()[completed_id]['state'] == 'SUCCEEDED'
                assert (tmp_path / f'completed-{index}' / 'entered').read_text() == 'run\n'
            assert not live(initial['workers'][0])
            assert (tmp_path / 'control' / 'entered').read_text() == 'run\n'
            assert not (tmp_path / 'control' / 'exited').exists()
            if not cancel and restart_failure == 'none':
                uploaded = HOLD_PLUGIN.replace('test.EditorHold', 'test.UploadedHold')
                response = requests.post(address + '/api/nodes/upload',
                    files={'file': ('uploaded_hold.py', uploaded, 'text/x-python')}, timeout=3)
                assert response.status_code == 201, response.text
                assert 'Worker notification deferred' in response.json()['message']
            (tmp_path / 'resume').touch()
            wait(ready)
            record = wait(lambda: history().get(identities[-1]))
            assert record['state'] == 'SUCCEEDED', record
            assert (tmp_path / 'queued-2.txt').read_text() == 'queued-2\n'
            assert not (tmp_path / 'queued-0.txt').exists()
            assert not (tmp_path / 'queued-1.txt').exists()
            latest = submit(text_graph())
            assert wait(lambda: history().get(latest))['state'] == 'SUCCEEDED'
            if not cancel and restart_failure == 'none':
                graph = held_graph(tmp_path / 'uploaded-control')
                graph = json.loads(json.dumps(graph).replace('test.EditorHold', 'test.UploadedHold'))
                (tmp_path / 'uploaded-control' / 'release').touch()
                uploaded_task = submit(graph)
                assert wait(lambda: history().get(uploaded_task))['state'] == 'SUCCEEDED'
                assert (tmp_path / 'uploaded-control' / 'entered').read_text() == 'run\n'
            final = pids()
            assert len(final['managers']) == (3 if restart_failure == 'worker' else 2)
            assert len(final['workers']) == 2
            process.terminate()
            process.wait(timeout=15)
            assert not any(live(pid) for group in final.values() for pid in group)
            assert (tmp_path / 'server-closed').exists()
        finally:
            (tmp_path / 'database-resume').touch()
            (tmp_path / 'resume').touch()
            (tmp_path / 'close-resume').touch()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for group in pids().values():
                for pid in group:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


@pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX process signals')
@pytest.mark.parametrize('stage', ['backoff', 'manager', 'worker'])
def test_editor_close_during_queue_recovery_reaps_all_children(tmp_path, monkeypatch, stage):
    import multiprocessing
    from nndeploy.server import app
    from nndeploy.server.task_queue import TaskQueue
    from test_editor_worker_supervision import idle, spawn, cleanup

    reached, resume = threading.Event(), threading.Event()
    workers, managers, notifications = [], [], []
    original_start = app._EditorManager.start

    def manager_start(self, *args, **kwargs):
        original_start(self, *args, **kwargs)
        managers.append(self._process)
        if len(managers) > 1 and stage == 'manager':
            reached.set()
            assert resume.wait(10)

    def worker_start(*args, **kwargs):
        ready = multiprocessing.get_context('spawn').Event()
        worker = spawn(idle, ready)
        workers.append(worker)
        assert ready.wait(10)
        if len(workers) > 1 and stage == 'worker':
            reached.set()
            assert resume.wait(10)
        return worker

    def factory(jobs, plugins, cancellations):
        return SimpleNamespace(queue=TaskQueue(job_mp_q=jobs),
                               notify_system_event=lambda event, data: notifications.append(event))

    monkeypatch.setattr(app._EditorManager, 'start', manager_start)
    monkeypatch.setattr(app, 'start_worker', worker_start)
    monkeypatch.setattr(app, 'configure_root_logger', lambda *args: SimpleNamespace(stop=lambda: None, _thread=threading.current_thread()))
    monkeypatch.setattr(app, 'WORKER_MONITOR_INTERVAL_SECONDS', .01)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', 60 if stage == 'backoff' else .01)
    runtime = app._EditorRuntime(SimpleNamespace(resources=tmp_path, log=str(tmp_path / 'log')))
    closer = None
    try:
        runtime.start(factory)
        assert runtime.ready
        managers[0].kill()
        managers[0].join(timeout=3)
        if stage == 'backoff':
            wait(lambda: runtime.manager is None)
        else:
            assert reached.wait(10)
        assert not runtime.ready
        closer = threading.Thread(target=runtime.close)
        closer.start()
        assert runtime.stop.wait(2)
        if stage != 'backoff':
            assert closer.is_alive(), 'Close must own the in-flight process creation'
            resume.set()
        closer.join(timeout=8)
        assert not closer.is_alive()
        assert runtime.closed
        assert not runtime.ipc_watchdog.thread.is_alive()
        assert not notifications, 'Shutdown must not announce a replacement as ready'
        assert len(managers) == (1 if stage == 'backoff' else 2)
        assert len(workers) == (2 if stage == 'worker' else 1)
        assert all(not manager.is_alive() for manager in managers)
        assert all(getattr(worker, '_closed', False) for worker in workers)
        assert all(not thread.is_alive() for thread in (runtime.supervisor,))
    finally:
        resume.set()
        runtime.close()
        if closer is not None:
            closer.join(timeout=10)
        cleanup(workers)
        for manager in managers:
            if manager.is_alive():
                manager.kill()
                manager.join(timeout=5)

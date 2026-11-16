"""Editor shutdown owns the most recent real Worker, including restart races."""
import multiprocessing
import logging
from logging.handlers import QueueListener
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import requests
from fastapi import FastAPI

from nndeploy.server import app
from nndeploy.server.task_queue import ExecutionStatus, TaskQueue
from test_server_api import text_graph
from test_cli import unused_port


def idle(ready):
    ready.set()
    threading.Event().wait(15)


def dead_worker():
    pass


def ignore_terminate(ready):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    idle(ready)


def spawn(target, *args):
    worker = multiprocessing.get_context('spawn').Process(target=target, args=args, daemon=True)
    worker.start()
    return worker


def cleanup(workers):
    for worker in workers:
        try:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=5)
            worker.close()
        except ValueError:
            pass  # The supervisor already joined and closed this handle.


def supervise(worker, stop, notify):
    thread = threading.Thread(target=app.monitor_worker,
        args=(worker, None, None, None, None, None, None, '.',
              SimpleNamespace(notify_system_event=notify), stop))
    thread.start()
    return thread


def test_editor_stop_during_restart_delay_does_not_spawn_another_worker(monkeypatch):
    initial = spawn(dead_worker)
    initial.join(timeout=10)
    assert not initial.is_alive()
    stop = threading.Event()
    workers = [initial]
    def restart(*args):
        ready = multiprocessing.get_context('spawn').Event()
        child = spawn(idle, ready)
        workers.append(child)
        return child
    def notify(event, data):
        if event == 'worker_died':
            stop.set()
    monkeypatch.setattr(app, 'start_worker', restart)
    thread = supervise(initial, stop, notify)
    try:
        thread.join(timeout=.5)
        assert not thread.is_alive(), 'Shutdown must interrupt the restart delay'
        assert len(workers) == 1
    finally:
        stop.set()
        thread.join(timeout=5)
        cleanup(workers)


def test_editor_restart_retries_spawn_errors_and_closes_the_successful_child(monkeypatch):
    initial = spawn(dead_worker)
    initial.join(timeout=10)
    stop, restarted = threading.Event(), threading.Event()
    workers, pids, attempts = [initial], [], []
    def restart(*args):
        attempts.append(True)
        if len(attempts) <= 2:
            raise OSError('Controlled spawn failure')
        ready = multiprocessing.get_context('spawn').Event()
        child = spawn(idle, ready)
        workers.append(child)
        pids.append(child.pid)
        assert ready.wait(10)
        return child
    def notify(event, data):
        if event == 'worker_restarted':
            restarted.set()
            stop.set()
    monkeypatch.setattr(app, 'start_worker', restart)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', .02)
    thread = supervise(initial, stop, notify)
    try:
        assert restarted.wait(10)
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(attempts) == 3
        with pytest.raises(ProcessLookupError):
            os.kill(pids[0], 0)
    finally:
        stop.set()
        thread.join(timeout=5)
        cleanup(workers)


@pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX SIGTERM handling')
def test_editor_shutdown_kills_and_reaps_a_worker_ignoring_terminate():
    ready = multiprocessing.get_context('spawn').Event()
    child = spawn(ignore_terminate, ready)
    pid = child.pid
    stop = threading.Event()
    thread = None
    try:
        assert ready.wait(10)
        thread = supervise(child, stop, lambda *args: None)
        stop.set()
        thread.join(timeout=6)
        assert not thread.is_alive()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=5)
        cleanup([child])


def test_editor_start_failure_after_spawning_does_not_leak_a_child(monkeypatch):
    ready = multiprocessing.get_context('spawn').Event()
    child = multiprocessing.get_context('spawn').Process(target=idle, args=(ready,), daemon=True)
    pids = []
    def failed_log(*args):
        pids.append(child.pid)
        assert ready.wait(10)
        raise OSError('Controlled post-spawn failure')
    monkeypatch.setattr(app.mp, 'Process', lambda **kwargs: child)
    monkeypatch.setattr(app.logging, 'info', failed_log)
    try:
        with pytest.raises(OSError, match='post-spawn'):
            app.start_worker(None, None, None, None, None, None, '.')
        with pytest.raises(ProcessLookupError):
            os.kill(pids[0], 0)
    finally:
        cleanup([child])


def test_editor_real_graph_and_logs_work_before_and_after_worker_restart(tmp_path, monkeypatch):
    context = multiprocessing.get_context('spawn')
    manager = app._EditorManager(ctx=context)
    manager.start()
    queues = [manager.JobQueue(), *[manager.Queue() for _ in range(5)]]
    task_q, result_q, progress_q, log_q, plugin_q, cancel_q = queues
    logged = queue.Queue()
    class Capture(logging.Handler):
        def emit(self, record):
            if 'supervision-' in record.getMessage():
                logged.put((record.getMessage(), record.task_id))
    listener = QueueListener(log_q, Capture())
    listener.start()
    events = queue.Queue()
    stop = threading.Event()
    workers, pids = [], []
    original_start = app.start_worker
    def start(*args):
        child = original_start(*args)
        workers.append(child)
        pids.append(child.pid)
        return child
    monkeypatch.setattr(app.mp, 'Process', context.Process)
    monkeypatch.setattr(app, 'start_worker', start)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', .02)
    monkeypatch.setattr(app, 'WORKER_MONITOR_INTERVAL_SECONDS', .02)
    worker = start(*queues, str(tmp_path))
    thread = threading.Thread(target=app.monitor_worker,
        args=(worker, *queues, str(tmp_path),
              SimpleNamespace(notify_system_event=lambda event, data: events.put((event, data))), stop))
    thread.start()
    try:
        for index in range(2):
            value, task_id = f'supervision-{index}', f'editor-run-{index}'
            graph = text_graph()
            graph['node_repository_'][0]['str_'] = value
            graph['node_repository_'][1]['frontend_params'].update(
                print_to_logger=True, save_to_file=True, file_path=str(tmp_path / 'output.txt'))
            task_q.put((index, {'id': task_id, 'graph_json': graph}))
            idx, status, results, profile = result_q.get(timeout=15)
            assert idx == index and status.completed, status.__dict__
            assert (tmp_path / 'output.txt').read_text().splitlines() == [f'supervision-{i}' for i in range(index + 1)]
            assert logged.get(timeout=5) == (f'[task_id={task_id}] {value}', task_id)
            while True:
                _, observed_task, progress = progress_q.get(timeout=5)
                assert observed_task == task_id
                if progress['event'] == 'finished':
                    break
            if index == 0:
                worker.kill()
                event, data = events.get(timeout=10)
                assert event == 'worker_died' and data['exitcode'] != 0
                event, data = events.get(timeout=10)
                assert event == 'worker_restarted' and data['pid'] != pids[0]
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        stop.set()
        thread.join(timeout=5)
        cleanup(workers)
        listener.stop()
        manager.shutdown()


@pytest.mark.parametrize('during_spawn', [False, True], ids=['after-restart', 'during-spawn'])
def test_editor_stop_reaps_the_latest_worker_even_when_restart_races_shutdown(monkeypatch, during_spawn):
    initial = spawn(dead_worker)
    initial.join(timeout=10)
    assert not initial.is_alive()
    stop, restarted, spawning, release = [threading.Event() for _ in range(4)]
    workers, pids, events = [initial], [], []
    def restart(*args):
        ready = multiprocessing.get_context('spawn').Event()
        child = spawn(idle, ready)
        workers.append(child)
        pids.append(child.pid)
        assert ready.wait(10)
        spawning.set()
        if during_spawn:
            assert release.wait(10)
        return child
    def notify(event, data):
        events.append((event, data))
        if event == 'worker_restarted':
            restarted.set()
    monkeypatch.setattr(app, 'start_worker', restart)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', .02, raising=False)
    monkeypatch.setattr(app, 'WORKER_MONITOR_INTERVAL_SECONDS', .02, raising=False)
    thread = supervise(initial, stop, notify)
    try:
        assert (spawning if during_spawn else restarted).wait(10)
        stop.set()
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(pids) == 1
        with pytest.raises(ProcessLookupError):
            os.kill(pids[0], 0)
        if during_spawn:
            assert not restarted.is_set()
        else:
            assert ('worker_restarted', {'pid': pids[0]}) in events
    finally:
        stop.set()
        release.set()
        thread.join(timeout=10)
        cleanup(workers)


@pytest.mark.parametrize('logger_failure', [False, True], ids=['restarted-worker', 'startup-failure'])
def test_editor_main_always_joins_supervision_on_exit(tmp_path, monkeypatch, logger_failure):
    workers, pids, monitors, stopped = [], [], [], []
    helpers = []
    existing_children = {child.pid for child in multiprocessing.active_children()}
    restarted, watching = threading.Event(), threading.Event()
    original_monitor = app.monitor_worker
    def monitor(*args, **kwargs):
        monitors.append((threading.current_thread(), args[-1]))
        watching.set()
        return original_monitor(*args, **kwargs)
    def start(*args, **kwargs):
        if not workers and not logger_failure:
            child = spawn(dead_worker)
            child.join(timeout=10)
        else:
            ready = multiprocessing.get_context('spawn').Event()
            child = spawn(idle, ready)
            assert ready.wait(10)
        workers.append(child)
        pids.append(child.pid)
        return child
    class Server:
        def __init__(self, *args, **kwargs):
            self.queue = TaskQueue()
            self.app = FastAPI()
        def notify_system_event(self, event, data):
            if event == 'worker_restarted':
                restarted.set()
        def run(self, **kwargs):
            assert restarted.wait(10)
    def logger(*args):
        if logger_failure:
            assert watching.wait(3)
            raise OSError('Controlled logger startup failure')
        return SimpleNamespace(stop=lambda: stopped.append(True), _thread=threading.current_thread())
    arguments = app.cli(['--resources', str(tmp_path / 'resources'), '--log', str(tmp_path / 'server.log')])
    monkeypatch.setattr(app, 'cli', lambda: arguments)
    monkeypatch.setattr(app.mp, 'set_start_method', lambda *args, **kwargs: None)
    monkeypatch.setattr(app, 'NnDeployServer', Server)
    monkeypatch.setattr(app, 'start_worker', start)
    monkeypatch.setattr(app, 'monitor_worker', monitor)
    monkeypatch.setattr(app, 'configure_root_logger', logger)
    for name in ('start_scheduler', 'start_finisher', 'start_progress_listener'):
        original = getattr(app, name)
        def tracked(*args, _original=original):
            thread = _original(*args)
            helpers.append(thread)
            return thread
        monkeypatch.setattr(app, name, tracked)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', .02, raising=False)
    monkeypatch.setattr(app, 'WORKER_MONITOR_INTERVAL_SECONDS', .02, raising=False)
    try:
        if logger_failure:
            with pytest.raises(OSError, match='logger startup'):
                app.main()
        else:
            app.main()
            assert stopped == [True]
        assert monitors and all(not thread.is_alive() for thread, _ in monitors)
        assert helpers and all(not thread.is_alive() for thread in helpers)
        assert {child.pid for child in multiprocessing.active_children()} <= existing_children
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        for thread, stop in monitors:
            stop.set()
            thread.join(timeout=5)
        cleanup(workers)


def test_editor_scheduler_stops_while_waiting_for_a_full_worker_queue():
    attempting, stop = threading.Event(), threading.Event()
    class FullQueue(queue.Queue):
        def put(self, item, **kwargs):
            attempting.set()
            super().put(item, **kwargs)
    jobs = FullQueue(maxsize=1)
    jobs.put('already-dispatched')
    attempting.clear()
    tasks = TaskQueue()
    tasks.put({'id': 'waiting'})
    thread = app.start_scheduler(tasks, jobs, stop)
    try:
        assert attempting.wait(3)
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert tasks.get_task_by_id('waiting')['state'] == 'CANCELLED'
        assert jobs.get_nowait() == 'already-dispatched'
        assert jobs.empty()
    finally:
        stop.set()
        thread.join(timeout=2)


def test_editor_finisher_continues_after_a_notification_error():
    notified, done, stop = [], threading.Event(), threading.Event()
    def on_done(task_id, *args):
        notified.append(task_id)
        if task_id == 'first':
            raise OSError('Controlled notification failure')
        done.set()
    tasks = TaskQueue(on_done=on_done)
    results = queue.Queue()
    for name in ('first', 'second'):
        tasks.put({'id': name})
        idx, _ = tasks.get()
        results.put((idx, ExecutionStatus(True), {'value': name}, {}))
    thread = app.start_finisher(tasks, results, stop)
    try:
        assert done.wait(3)
        assert notified == ['first', 'second']
        for name in notified:
            assert tasks.get_task_by_id(name)['state'] == 'SUCCEEDED'
    finally:
        stop.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_editor_replaced_reader_cannot_take_new_tasks():
    jobs = app._EditorJobQueue()
    old_reader = jobs.new_reader()
    waiting = threading.Event()
    finished = queue.Queue()
    def read():
        waiting.set()
        try:
            finished.put(jobs.get(reader=old_reader, timeout=10))
        except queue.Empty:
            finished.put('invalidated')
    thread = threading.Thread(target=read)
    thread.start()
    try:
        assert waiting.wait(3)
        current = jobs.new_reader()
        jobs.put((1, 'new-task'))
        assert finished.get(timeout=2) == 'invalidated'
        # Also cover a request from the old connection reaching the manager late.
        with pytest.raises(queue.Empty):
            jobs.get(reader=old_reader, timeout=0)
        assert jobs.get(reader=current, timeout=0) == (1, 'new-task')
        assert jobs.claim(1, current)
        jobs.put('flush-me')
        assert jobs.get_nowait() == 'flush-me'  # Existing editor flush API.
    finally:
        jobs.retire_reader(jobs.reader)
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_editor_requeues_only_unconfirmed_deliveries_after_worker_exit():
    jobs = app._EditorJobQueue(maxsize=2)
    old = jobs.new_reader()
    jobs.put((1, 'received'))
    jobs.put((2, 'unconfirmed'))
    assert jobs.get(reader=old, timeout=0) == (1, 'received')
    with pytest.raises(queue.Full):
        jobs.put_nowait((3, 'later'))
    assert jobs.claim(1, old)
    assert jobs.get(reader=old, timeout=0) == (2, 'unconfirmed')
    jobs.put((3, 'later'))
    with pytest.raises(RuntimeError, match='confirmed exit'):
        jobs.new_reader()
    jobs.retire_reader(old)
    current = jobs.new_reader()
    assert not jobs.claim(2, old)
    assert jobs.get(reader=current, timeout=0) == (2, 'unconfirmed')
    assert jobs.claim(2, current)
    assert jobs.get(reader=current, timeout=0) == (3, 'later')
    assert jobs.claim(3, current)
    with pytest.raises(queue.Empty):
        jobs.get_nowait()  # Never re-run confirmed computation after a crash.


@pytest.mark.skipif(os.name != 'posix', reason='Uses POSIX process inspection and signals')
def test_editor_http_run_survives_idle_worker_crash_and_shutdown_reaps_children(tmp_path):
    script = '''
import os
from pathlib import Path
from nndeploy.server import Server
from nndeploy.server.template import WorkflowTemplateManager
from nndeploy.server.app import main
WorkflowTemplateManager.init_templates = staticmethod(lambda: Path.cwd())
original_close = Server.close
async def close(self):
    await original_close(self)
    Path(os.environ['NNDEPLOY_TEST_SERVER_CLOSED']).touch()
Server.close = close
main()
'''
    port = unused_port()
    address = f'http://127.0.0.1:{port}'
    log_path = tmp_path / 'editor-console.log'
    children = set()
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', script, '--host', '127.0.0.1',
            '--port', str(port), '--debug', '--resources', str(tmp_path / 'resources'),
            '--log', str(tmp_path / 'editor.log')], stdout=log, stderr=log,
            env=dict(os.environ, NNDEPLOY_TEST_SERVER_CLOSED=str(tmp_path / 'server-closed')))
        def remember_children():
            table = subprocess.check_output(['ps', '-axo', 'pid=,ppid='], text=True)
            parents = {int(pid): int(parent) for pid, parent in (line.split() for line in table.splitlines())}
            descendants = {process.pid}
            while True:
                found = {pid for pid, parent in parents.items() if parent in descendants}
                if found <= descendants:
                    break
                descendants.update(found)
            children.update(descendants - {process.pid})
        def alive(pid):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            return True
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                try:
                    if requests.get(address + '/health/ready', timeout=.5).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.05)
            else:
                pytest.fail('Editor did not start: ' + log_path.read_text())
            worker_pids = []
            for index in range(2):
                graph = text_graph()
                graph['node_repository_'][0]['str_'] = f'http-run-{index}'
                graph['node_repository_'][1]['frontend_params'].update(
                    print_to_logger=True, save_to_file=True, file_path=str(tmp_path / 'http-output.txt'))
                response = requests.post(address + '/api/queue', json=graph, timeout=5)
                assert response.status_code == 202, response.text
                task_id = response.json()['result']['task_id']
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    remember_children()
                    response = requests.get(address + '/api/queue/history', timeout=5)
                    assert response.status_code == 200, response.text
                    record = next((item for item in response.json()['result']['items'] if item['task_id'] == task_id), None)
                    if record:
                        break
                    time.sleep(.05)
                else:
                    pytest.fail('Editor task did not finish: ' + log_path.read_text())
                assert record['state'] == 'SUCCEEDED', record
                worker_pids.append(record['worker_pid'])
                assert (tmp_path / 'http-output.txt').read_text().splitlines() == [f'http-run-{i}' for i in range(index + 1)]
                if index == 0:
                    assert worker_pids[0] in children
                    os.kill(worker_pids[0], signal.SIGKILL)
            assert worker_pids[0] != worker_pids[1]
            process.terminate()
            process.wait(timeout=15)
            deadline = time.monotonic() + 5
            while any(alive(pid) for pid in children) and time.monotonic() < deadline:
                time.sleep(.05)
            assert not any(alive(pid) for pid in children), children
            assert (tmp_path / 'server-closed').exists()
            editor_log = (tmp_path / 'editor.log').read_text()
            assert 'http-run-0' in editor_log and 'http-run-1' in editor_log
            console_log = log_path.read_text()
            assert 'http-run-0' in console_log and 'http-run-1' in console_log
        finally:
            if process.poll() is None:
                remember_children()
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for pid in children:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

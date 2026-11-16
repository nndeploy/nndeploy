"""Stalled IPC is recoverable without timing out healthy idle queues or Graphs."""
import json
import multiprocessing
import os
from queue import Empty, Full
import signal
import subprocess
import sys
import threading
import time

import pytest
import requests

from nndeploy.server import app
from test_cli import unused_port
from test_editor_queue_recovery import LAUNCH, live, wait
from test_editor_task_control import HOLD_PLUGIN, held_graph


@pytest.mark.parametrize('method', ['get', 'put', 'idle-get', 'idle-put'])
def test_legitimate_queue_wait_does_not_kill_manager(monkeypatch, method):
    monkeypatch.setattr(app, 'EDITOR_IPC_TIMEOUT_SECONDS', .15)
    watchdog = app._EditorIpcWatchdog()
    reader = None
    with app._EditorManager(ctx=multiprocessing.get_context('spawn')) as manager:
        channel = manager.Queue(1)
        channel._editor_watch = (watchdog, manager._process)
        watchdog.start()
        try:
            if method == 'get':
                with pytest.raises(Empty):
                    channel.get(True, .5)
            elif method == 'put':
                channel.put('first')
                with pytest.raises(Full):
                    channel.put('second', block=True, timeout=.5)
            else:
                received = []
                if method == 'idle-put':
                    channel.put('first')
                    reader = threading.Thread(target=lambda: (channel.put('second'), received.append('resume')))
                else:
                    reader = threading.Thread(target=lambda: received.append(channel.get()))
                reader.start()
                time.sleep(.5)
                assert reader.is_alive()
                if method == 'idle-put':
                    assert channel.get_nowait() == 'first'
                else:
                    channel.put('resume')
                reader.join(timeout=2)
                assert received == ['resume']
            assert manager._process.is_alive()
        finally:
            watchdog.close()
            if reader is not None and reader.is_alive():
                manager._process.kill()
                reader.join(timeout=3)
        assert not watchdog.thread.is_alive()
        assert not watchdog.pending


@pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX process signals')
@pytest.mark.parametrize('stalled', [False, True], ids=['long-graph', 'shutdown-with-stalled-ipc'])
def test_editor_graph_and_shutdown_are_independent_of_ipc_deadline(tmp_path, stalled):
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    (plugins / 'hold.py').write_text(HOLD_PLUGIN)
    address = f'http://127.0.0.1:{unused_port()}'
    logs = tmp_path / 'console.log'
    def pids():
        path = tmp_path / 'pids.json'
        return json.loads(path.read_text()) if path.exists() else dict(managers=[], workers=[])
    with logs.open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', LAUNCH, '--host', '127.0.0.1',
            '--port', address.rsplit(':', 1)[1], '--debug', '--resources', str(resources),
            '--log', str(tmp_path / 'editor.log')], stdout=log, stderr=log,
            env=dict(os.environ, NNDEPLOY_EDITOR_RECOVERY_TEST=str(tmp_path)))
        def ready():
            assert process.poll() is None, logs.read_text()
            try:
                return requests.get(address + '/health/ready', timeout=.5).status_code == 200
            except requests.RequestException:
                return False
        try:
            wait(ready, 30)
            response = requests.post(address + '/api/queue', json=held_graph(tmp_path / 'control'), timeout=3)
            assert response.status_code == 202, response.text
            identity = response.json()['result']['task_id']
            wait(lambda: (tmp_path / 'control' / 'entered').exists())
            original = pids()
            if stalled:
                os.kill(original['managers'][0], signal.SIGSTOP)
            else:
                # The native Graph stays in run() for more than twice the
                # test IPC allowance. No replacement or failed task is allowed.
                time.sleep(4.2)
                assert ready()
                assert pids() == original
                (tmp_path / 'control' / 'release').touch()
                def completed():
                    result = requests.get(address + '/api/queue/task/' + identity, timeout=2)
                    assert result.status_code == 200, result.text
                    return result.json()['result']['state'] == 'SUCCEEDED'
                wait(completed)
                assert (tmp_path / 'control' / 'entered').read_text() == 'run\n'
            process.terminate()
            process.wait(timeout=10)
            assert pids() == original, 'Closing must not create a replacement queue or Worker'
            assert not any(live(pid) for group in pids().values() for pid in group)
            assert (tmp_path / 'server-closed').exists()
        finally:
            (tmp_path / 'resume').touch()
            if (tmp_path / 'control').exists():
                (tmp_path / 'control' / 'release').touch()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for group in pids().values():
                for pid in group:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


class _LockedJobQueue(app._EditorJobQueue):
    def lock_queue(self):
        self.mutex.acquire()
        return True


class _LockedJobProxy(app._EditorJobProxy):
    _exposed_ = app._EditorJobProxy._exposed_ + ('lock_queue',)

    def lock_queue(self):
        return self._callmethod('lock_queue')


class _FaultManager(app._EditorManager):
    pass


_FaultManager.register('LockedJobs', _LockedJobQueue, proxytype=_LockedJobProxy)


def test_one_stalled_queue_is_detected_even_when_manager_still_answers_other_calls(monkeypatch):
    monkeypatch.setattr(app, 'EDITOR_IPC_TIMEOUT_SECONDS', .15)
    watchdog = app._EditorIpcWatchdog()
    with _FaultManager(ctx=multiprocessing.get_context('spawn')) as manager:
        jobs, other = manager.LockedJobs(), manager.Queue()
        jobs._editor_watch = (watchdog, manager._process)
        watchdog.start()
        try:
            assert jobs.lock_queue()
            other.put_nowait('manager still responds')
            assert other.get_nowait() == 'manager still responds'
            with pytest.raises((EOFError, OSError)):
                jobs.outcomes()
            wait(lambda: not manager._process.is_alive(), 3)
            assert watchdog.timed_out is manager._process
            assert not watchdog.pending
        finally:
            watchdog.close()


def _stall_startup():
    from pathlib import Path
    Path(os.environ['NNDEPLOY_TEST_IPC_STARTUP_MARKER']).touch()
    time.sleep(30)


def _stalled_queue(*args, **kwargs):
    _stall_startup()
    return app._EditorJobQueue(*args, **kwargs)


class _StalledInitManager(app._EditorManager):
    pass


class _StalledJobManager(app._EditorManager):
    pass


class _StalledChannelManager(app._EditorManager):
    pass


_StalledJobManager.register('JobQueue', _stalled_queue, proxytype=app._EditorJobProxy)
_StalledChannelManager.register('Queue', _stalled_queue, proxytype=app._EditorQueueProxy)


@pytest.mark.parametrize('stage', ['initializer', 'job-proxy', 'channel-proxy'])
def test_editor_startup_ipc_timeout_reaps_partial_manager(tmp_path, monkeypatch, stage):
    from types import SimpleNamespace
    marker = tmp_path / 'blocked'
    monkeypatch.setenv('NNDEPLOY_TEST_IPC_STARTUP_MARKER', str(marker))
    monkeypatch.setattr(app, 'EDITOR_IPC_TIMEOUT_SECONDS', 2)
    manager_type = {'initializer': _StalledInitManager, 'job-proxy': _StalledJobManager,
                    'channel-proxy': _StalledChannelManager}[stage]
    if stage == 'initializer':
        original_start = manager_type.start
        monkeypatch.setattr(manager_type, 'start', lambda self, **kwargs:
                            original_start(self, initializer=_stall_startup))
    monkeypatch.setattr(app, '_EditorManager', lambda: manager_type(ctx=multiprocessing.get_context('spawn')))
    runtime = app._EditorRuntime(SimpleNamespace(resources=tmp_path, log=str(tmp_path / 'editor.log')))
    errors = []
    def start():
        try:
            runtime.start(lambda *args: pytest.fail('Server factory must not run after incomplete IPC startup'))
        except BaseException as error:
            errors.append(error)
    caller = threading.Thread(target=start)
    caller.start()
    process = None
    try:
        wait(marker.exists, 5)
        process = runtime.manager._process
        caller.join(timeout=4)
        assert not caller.is_alive(), 'IPC initialization must have a bounded failure path'
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError), errors
        assert runtime.closed
        assert not process.is_alive()
        assert not runtime.ipc_watchdog.thread.is_alive()
        assert not runtime.ipc_watchdog.pending
    finally:
        if process is None and runtime.manager is not None:
            process = getattr(runtime.manager, '_process', None)
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=3)
        caller.join(timeout=5)
        runtime.close()


@pytest.mark.skipif(os.name != 'posix', reason='Requires SIGSTOP')
def test_editor_shutdown_of_partial_startup_has_an_ipc_deadline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(app, 'EDITOR_IPC_TIMEOUT_SECONDS', 2)
    runtime = app._EditorRuntime(SimpleNamespace(resources=tmp_path, log=str(tmp_path / 'editor.log')))
    runtime.ipc_watchdog.start()
    runtime._channels()
    process = runtime.manager._process
    os.kill(process.pid, signal.SIGSTOP)
    caller = threading.Thread(target=runtime.close)
    caller.start()
    try:
        caller.join(timeout=4)
        assert not caller.is_alive(), 'Manager shutdown handshake must not block cleanup forever'
        assert runtime.closed
        assert not process.is_alive()
        assert not runtime.ipc_watchdog.thread.is_alive()
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=3)
        caller.join(timeout=5)
        runtime.close()


@pytest.mark.parametrize('stage', ['initializer', 'job-proxy', 'channel-proxy'])
@pytest.mark.parametrize('closing,failed_starts', [(False, 1), (False, 2), (True, 1)],
                         ids=['retry', 'retry-twice', 'close'])
def test_editor_failed_replacement_startup_retries_or_closes(tmp_path, monkeypatch, stage, closing, failed_starts):
    from types import SimpleNamespace
    from nndeploy.server.task_queue import TaskQueue
    from test_editor_worker_supervision import idle, spawn, cleanup
    marker = tmp_path / 'blocked'
    monkeypatch.setenv('NNDEPLOY_TEST_IPC_STARTUP_MARKER', str(marker))
    monkeypatch.setattr(app, 'EDITOR_IPC_TIMEOUT_SECONDS', 2)
    monkeypatch.setattr(app, 'WORKER_MONITOR_INTERVAL_SECONDS', .02)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', .02)
    stalled_type = {'initializer': _StalledInitManager, 'job-proxy': _StalledJobManager,
                    'channel-proxy': _StalledChannelManager}[stage]
    if stage == 'initializer':
        original_start = stalled_type.start
        monkeypatch.setattr(stalled_type, 'start', lambda self, **kwargs:
                            original_start(self, initializer=_stall_startup))
    managers, workers, notifications = [], [], []
    def manager():
        manager_type = stalled_type if 1 <= len(managers) <= failed_starts else _FaultManager
        instance = manager_type(ctx=multiprocessing.get_context('spawn'))
        managers.append(instance)
        return instance
    def worker(*args, storage_options=None):
        assert storage_options is None  # This runtime stub has no file storage.
        ready = multiprocessing.get_context('spawn').Event()
        process = spawn(idle, ready)
        workers.append(process)
        assert ready.wait(5)
        return process
    def factory(jobs, plugins, cancellations):
        return SimpleNamespace(queue=TaskQueue(job_mp_q=jobs),
                               notify_system_event=lambda event, data: notifications.append(event))
    monkeypatch.setattr(app, '_EditorManager', manager)
    monkeypatch.setattr(app, 'start_worker', worker)
    monkeypatch.setattr(app, 'configure_root_logger',
                        lambda *args: SimpleNamespace(stop=lambda: None, _thread=threading.current_thread()))
    runtime = app._EditorRuntime(SimpleNamespace(resources=tmp_path, log=str(tmp_path / 'editor.log')))
    try:
        runtime.start(factory)
        assert runtime.ready
        managers[0]._process.kill()
        managers[0]._process.join(timeout=3)
        wait(marker.exists, 5)
        assert not runtime.ready
        if closing:
            runtime.close()
            assert runtime.closed
            assert len(managers) == 2
            assert len(workers) == 1
            assert not notifications
        else:
            # Slow healthy starts may also hit the short test deadline. Recovery
            # must allow additional attempts instead of requiring exactly three.
            wait(lambda: len(managers) >= failed_starts + 2 and runtime.ready,
                 6 + 2 * failed_starts)
            wait(lambda: notifications == ['worker_restarted'])
            assert len(workers) == 2
            assert all(not instance._process.is_alive() for instance in managers[:-1])
        runtime.close()
        assert not runtime.ipc_watchdog.thread.is_alive()
        assert not runtime.ipc_watchdog.pending
        assert all(not instance._process.is_alive() for instance in managers)
        assert all(getattr(process, '_closed', False) for process in workers)
    finally:
        for instance in managers:
            process = getattr(instance, '_process', None)
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.kill()
                process.join(timeout=3)
        runtime.close()
        cleanup(workers)

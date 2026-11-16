"""Plugin notifications are hints; saved plugin files remain authoritative."""
import queue
import io
import importlib
import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from nndeploy.server import Server, app, worker
from test_editor_history import location
from test_editor_queue_recovery import wait
from test_server_api import text_graph


def test_plugin_notification_backlog_is_coalesced_and_all_saved_files_load(tmp_path, location, monkeypatch):
    monkeypatch.setattr(app, 'WORKER_MONITOR_INTERVAL_SECONDS', .02)
    monkeypatch.setattr(app, 'WORKER_RESTART_DELAY_SECONDS', .02)
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = app._EditorRuntime(args)
    server = runtime.start(lambda jobs, updates, cancellations: Server(
        args, jobs, updates, cancellations, database_url=location))
    with TestClient(server.app) as client:
        try:
            for name in ('first', 'second', 'third'):
                path = plugins / (name + '.py')
                source = "import os\nfrom pathlib import Path\nPath(__file__).with_suffix('.loaded').write_text(str(os.getpid()))\n"
                response = client.post('/api/nodes/upload', files={'file': (path.name, source, 'text/x-python')})
                assert response.status_code == 201, response.text
                for _ in range(100):
                    assert runtime.notify_plugin(str(path))
            assert runtime.plugin_q.qsize() == 1
            identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
            record = wait(lambda: _completed(client, identity))
            assert record['state'] == 'SUCCEEDED', record
            for name in ('first', 'second', 'third'):
                assert (plugins / (name + '.loaded')).read_text() == str(runtime.worker.pid)
            # Replacement Workers must discover all files without replaying
            # the already-drained notifications.
            previous = runtime.worker
            previous.terminate()
            wait(lambda: runtime.worker is not previous and runtime.worker.is_alive())
            identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
            assert wait(lambda: _completed(client, identity))['state'] == 'SUCCEEDED'
            for name in ('first', 'second', 'third'):
                assert (plugins / (name + '.loaded')).read_text() == str(runtime.worker.pid)
        finally:
            runtime.close()


def _completed(client, identity):
    record = client.get('/api/queue/task/' + identity).json()['result']
    return record if record['state'] in ('SUCCEEDED', 'FAILED', 'CANCELLED') else None


def test_plugin_poll_does_not_drain_an_endless_notification_producer(tmp_path, monkeypatch):
    path = tmp_path / 'external.py'
    path.write_text('value = 1')
    calls = []
    class ContinuouslyFilled:
        count = 0
        def get_nowait(self):
            self.count += 1
            if self.count > 100:
                raise AssertionError('Plugin notifications starved the next Graph')
            return str(path)
    updates = ContinuouslyFilled()
    monkeypatch.setattr(worker, 'add_global_import_lib', calls.append)
    imports = []
    monkeypatch.setattr(worker, 'import_global_import_lib', lambda: imports.append(True))
    worker.poll_plugin_updates(updates, tmp_path)
    assert updates.count <= 64
    assert calls == [str(path)]
    assert imports == [True]


@pytest.mark.parametrize('notifications', ['full', 'absent', 'disconnected'])
def test_upload_does_not_wait_for_legacy_notification_queue(tmp_path, location, monkeypatch, notifications):
    server_module = importlib.import_module('nndeploy.server.server')
    imported = []
    monkeypatch.setattr(server_module, 'add_global_import_lib', imported.append)
    monkeypatch.setattr(server_module, 'import_global_import_lib', lambda: None)

    class LegacyQueue(queue.Queue):
        def put(self, value, block=True, timeout=None):
            assert not block, 'Plugin upload must not block waiting for notification capacity'
            if notifications == 'disconnected':
                raise EOFError('Manager exited')
            return super().put(value, block, timeout)

    updates = LegacyQueue(maxsize=1)
    if notifications == 'full':
        updates.put_nowait('older.py')
    server = Server(SimpleNamespace(resources=str(tmp_path / 'resources'), debug=True),
                    plugin_update_q=None if notifications == 'absent' else updates, database_url=location)
    with TestClient(server.app) as client:
        for name in ('one.py', 'two.py'):
            response = client.post('/api/nodes/upload', files={'file': (name, 'value = 1', 'text/x-python')})
            assert response.status_code == 201, response.text
            saved = tmp_path / 'resources' / 'plugin' / name
            assert saved.read_text() == 'value = 1'
            assert str(saved.resolve()) in imported
            assert ('notification deferred' in response.json()['message']) == (notifications != 'full')
        # The native import manager recognizes lowercase extensions only.
        response = client.post('/api/nodes/upload', files={'file': ('ignored.PY', 'value = 1')})
        assert response.status_code == 400


@pytest.mark.parametrize('failed', [False, True])
def test_plugin_upload_only_publishes_complete_files(tmp_path, monkeypatch, failed):
    from nndeploy.server.utils import _save_plugin_upload
    folder = tmp_path / 'plugin'
    folder.mkdir()
    destination = folder / 'node.py'
    destination.write_text('old = True')
    seen = []
    monkeypatch.setattr(worker, 'add_global_import_lib', seen.append)
    monkeypatch.setattr(worker, 'import_global_import_lib', lambda: None)

    class Upload(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            # Scan while writing: only the complete previous file is visible.
            worker.poll_plugin_updates(None, tmp_path)
            assert set(seen) == {str(destination.resolve())}
            assert destination.read_text() == 'old = True'
            if failed and self.tell():
                raise OSError('Upload stream failed')
            return super().read(size)

    source = Upload(b'new = True\n' * 150000)
    if failed:
        with pytest.raises(OSError, match='Upload stream failed'):
            _save_plugin_upload(source, destination)
        assert destination.read_text() == 'old = True'
    else:
        _save_plugin_upload(source, destination)
        assert destination.read_bytes() == source.getvalue()
    assert list(folder.iterdir()) == [destination]


def test_cancelled_upload_waits_for_its_file_writer(tmp_path, location):
    entered, release = threading.Event(), threading.Event()

    class Upload(io.BytesIO):
        def read(self, size=-1):
            entered.set()
            assert release.wait(5), 'Test writer was not released'
            assert not self.closed
            return super().read(size)

    async def scenario():
        server = Server(SimpleNamespace(resources=str(tmp_path / 'resources'), debug=True), database_url=location)
        endpoint = next(route.endpoint for route in server.app.routes if route.path == '/api/nodes/upload')
        source = Upload(b'value = 1')
        file = UploadFile(source, filename='node.py')

        async def request():
            try:
                return await endpoint(file)
            finally:
                await file.close()

        await server.start()
        pending = asyncio.create_task(request())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            pending.cancel()
            await asyncio.sleep(.02)
            assert not pending.done()
            assert not source.closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert source.closed
            assert (tmp_path / 'resources' / 'plugin' / 'node.py').read_bytes() == b'value = 1'
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await server.close()

    asyncio.run(scenario())

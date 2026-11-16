"""Explicitly thread-safe code shares one handler with bounded execution threads."""
import asyncio
from pathlib import Path
import threading
import time

import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.logging_taskid import get_task_id
from test_engine import engine_config, until, build_engine


def concurrent_config(stream=False):
    config = engine_config(stream=stream)
    config['execution']['scheduling'] = 'concurrent'
    config['inputs']['payload'] = {'type': 'json', 'required': False, 'default': {'seen': []}}
    return config


class ThreadSafeHandler:
    def __init__(self):
        self.owner = threading.get_ident()
        self.lock = threading.Lock()
        self.active = self.peak = self.initialized = self.closed = 0
        self.threads = set()

    def init(self):
        assert threading.get_ident() == self.owner
        self.initialized += 1

    def deploy(self, values):
        identity = get_task_id()
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.get_ident())
        try:
            if values['marker']:
                Path(values['marker']).touch()
            deadline = time.monotonic() + 10
            while values['release'] and not Path(values['release']).exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('Test release was not signalled')
                time.sleep(.01)
            values['payload']['seen'].append(identity)
            assert get_task_id() == identity
            return {'text': values['text'], 'seen': values['payload']['seen']}
        finally:
            with self.lock:
                self.active -= 1

    def deinit(self):
        assert threading.get_ident() == self.owner and self.active == 0
        self.closed += 1


def build_thread_safe_handler():
    return ThreadSafeHandler()


class ThreadSafeStream(ThreadSafeHandler):
    def deploy(self, values):
        yield {'text': 'started'}
        yield super().deploy(values)


def build_thread_safe_stream():
    return ThreadSafeStream()


@pytest.mark.parametrize('stream', [False, True])
def test_concurrent_handler_reuses_initialization_without_serializing_requests(tmp_path, stream):
    server = Server(name='code', handler_factory=build_thread_safe_stream if stream else build_thread_safe_handler,
                    data_dir=tmp_path, config=concurrent_config(stream))
    async def scenario():
        release, marker = tmp_path / 'release', tmp_path / 'marker'
        await server.start()
        handler = server.entries['code']['workers'][0].executor.handler
        try:
            first_id, first = await server.submit('code', {'release': str(release), 'marker': str(marker)})
            await until(marker.exists)
            second = await asyncio.wait_for(server.deploy('code', {'text': 'short'}), 3)
            assert not first.done() and second['outputs'] == {'text': 'short', 'seen': [second['task_id']]}
            assert handler.initialized == 1 and handler.peak == 2 and len(handler.threads) == 2
            assert handler.owner not in handler.threads
            release.touch()
            assert (await first)['outputs']['seen'] == [first_id]
            if stream:
                assert [item['kind'] for item in server.db.read_task_events(first_id)] == ['start', 'delta', 'delta', 'result', 'done']
        finally:
            release.touch()
            await server.close()
        assert handler.closed == 1
    asyncio.run(scenario())


def test_concurrent_cancellation_keeps_blocked_threads_counted(tmp_path):
    server = Server(name='code', handler_factory=build_thread_safe_handler, data_dir=tmp_path, config=concurrent_config())
    async def scenario():
        release = tmp_path / 'release'
        markers = [tmp_path / str(index) for index in range(2)]
        await server.start()
        try:
            accepted = [await server.submit('code', {'release': str(release), 'marker': str(marker)}) for marker in markers]
            await until(lambda: all(path.exists() for path in markers))
            await server.cancel_task(accepted[0][0])
            with pytest.raises(ServerError) as full:
                await server.submit('code', {})
            assert full.value.status_code == 429
            assert all(not future.done() for _, future in accepted)
            release.touch()
            with pytest.raises(ServerError) as cancelled:
                await accepted[0][1]
            assert cancelled.value.code == 'task_cancelled'
            assert (await accepted[1][1])['outputs']['seen'] == [accepted[1][0]]
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_concurrent_rejects_async_handlers_and_graphs(tmp_path):
    from test_server_api import text_graph
    async def scenario():
        for source in ({'workflow': text_graph()}, {'handler_factory': build_engine}):
            server = Server(name='invalid', data_dir=tmp_path, config=concurrent_config(), **source)
            try:
                with pytest.raises(ValueError, match='Graph|synchronous'):
                    await server.start()
            finally:
                await server.close()
    asyncio.run(scenario())

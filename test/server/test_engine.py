"""Shared engine instances accept bounded overlapping work on their owning loop."""
import asyncio
from pathlib import Path
import threading

import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.logging_taskid import get_task_id


def engine_config(*, concurrency=2, cooperative=False, stream=False):
    return {'execution': {'scheduling': 'engine', 'max_concurrency_per_worker': concurrency, 'max_queue_size': 0,
                          'cooperative_cancel': cooperative},
            'streaming': {'enabled': stream},
            'inputs': {'text': {'type': 'string', 'required': False, 'default': 'echo'},
                       'release': {'type': 'string', 'required': False, 'default': ''},
                       'marker': {'type': 'string', 'required': False, 'default': ''},
                       'fail': {'type': 'boolean', 'required': False, 'default': False}}}


class ControlledEngine:
    def __init__(self):
        self.owner = threading.get_ident()
        self.active = self.peak = self.calls = self.initialized = self.closed = 0

    async def init(self):
        assert threading.get_ident() == self.owner
        self.initialized += 1

    async def deploy(self, values):
        assert threading.get_ident() == self.owner
        self.active += 1
        self.calls += 1
        self.peak = max(self.active, self.peak)
        identity = get_task_id()
        try:
            if values['marker']:
                Path(values['marker']).write_text(identity)
            while values['release'] and not Path(values['release']).exists():
                await asyncio.sleep(.01)
            if values['fail']:
                raise ValueError('controlled request failure')
            assert get_task_id() == identity
            return {'text': values['text'], 'task_id': identity}
        finally:
            self.active -= 1

    async def deinit(self):
        assert self.active == 0 and threading.get_ident() == self.owner
        self.closed += 1


def build_engine():
    return ControlledEngine()


def current_task(values):
    return {'task_id': get_task_id()}


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_serial_workers_use_the_same_isolated_task_context(tmp_path, isolation):
    server = Server(name='context', deploy=current_task, resources=tmp_path, data_dir=tmp_path, config={'execution': {'isolation': isolation}})
    async def scenario():
        await server.start()
        try:
            results = await asyncio.gather(*(server.deploy('context', {}) for _ in range(3)))
            assert len({result['task_id'] for result in results}) == 3
            assert all(result['outputs']['task_id'] == result['task_id'] for result in results)
            assert get_task_id() is None
        finally:
            await server.close()
    asyncio.run(scenario())


class StreamingEngine(ControlledEngine):
    async def deploy(self, values):
        self.active += 1
        self.calls += 1
        self.peak = max(self.active, self.peak)
        identity = get_task_id()
        try:
            yield {'text': 'started', 'task_id': identity}
            if values['marker']:
                Path(values['marker']).write_text(identity)
            while values['release'] and not Path(values['release']).exists():
                await asyncio.sleep(.01)
            assert get_task_id() == identity
            yield {'text': values['text'], 'task_id': identity}
        finally:
            self.active -= 1


def build_streaming_engine():
    return StreamingEngine()


async def until(predicate):
    for _ in range(500):
        if predicate():
            return
        await asyncio.sleep(.01)
    raise AssertionError('Expected condition did not become true')


def test_q05_one_engine_accepts_second_request_while_first_is_running(tmp_path):
    server = Server(name='engine', handler_factory=build_engine, resources=tmp_path, data_dir=tmp_path, config=engine_config())
    async def scenario():
        release, marker = tmp_path / 'release', tmp_path / 'marker'
        await server.start()
        engine = server.entries['engine']['workers'][0].executor.handler
        try:
            first_id, first = await server.submit('engine', {'text': 'long', 'release': str(release), 'marker': str(marker)})
            await until(marker.exists)
            second = await asyncio.wait_for(server.deploy('engine', {'text': 'short'}), 3)
            assert second['outputs']['text'] == 'short' and not first.done()
            assert second['outputs']['task_id'] == second['task_id'] != first_id
            assert engine.initialized == 1 and engine.peak == 2
            assert engine.owner != threading.get_ident()
            release.touch()
            assert (await first)['outputs'] == {'text': 'long', 'task_id': first_id}
        finally:
            release.touch()
            await server.close()
        assert engine.closed == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('cooperative', [False, True])
def test_engine_cancellation_is_request_scoped_and_capacity_tracks_actual_work(tmp_path, cooperative):
    server = Server(name='engine', handler_factory=build_engine, resources=tmp_path, data_dir=tmp_path, config=engine_config(cooperative=cooperative))
    async def scenario():
        release = [tmp_path / f'release-{index}' for index in range(2)]
        markers = [tmp_path / f'marker-{index}' for index in range(2)]
        await server.start()
        engine = server.entries['engine']['workers'][0].executor.handler
        try:
            accepted = [await server.submit('engine', {'release': str(release[index]), 'marker': str(markers[index])}) for index in range(2)]
            await until(lambda: all(path.exists() for path in markers))
            with pytest.raises(ServerError) as full:
                await server.submit('engine', {})
            assert full.value.status_code == 429 and engine.calls == 2
            await server.cancel_task(accepted[0][0])
            if not cooperative:
                assert server.db.get_task(accepted[0][0])['state'] == 'running'
                with pytest.raises(ServerError) as still_full:
                    await server.submit('engine', {})
                assert still_full.value.status_code == 429
                release[0].touch()
            with pytest.raises(ServerError) as stopped:
                await asyncio.wait_for(asyncio.shield(accepted[0][1]), 3)
            assert stopped.value.code == 'task_cancelled'
            assert not accepted[1][1].done() and engine.active == 1
            assert (await server.deploy('engine', {}))['outputs']['text'] == 'echo'
            release[1].touch()
            await accepted[1][1]
            assert engine.peak == 2
        finally:
            for path in release:
                path.touch()
            await server.close()
    asyncio.run(scenario())


def test_engine_failure_and_stop_do_not_destroy_other_inflight_requests(tmp_path):
    server = Server(name='engine', handler_factory=build_engine, resources=tmp_path, data_dir=tmp_path, config=engine_config())
    async def scenario():
        release, marker = tmp_path / 'release', tmp_path / 'marker'
        await server.start()
        engine = server.entries['engine']['workers'][0].executor.handler
        try:
            _, first = await server.submit('engine', {'release': str(release), 'marker': str(marker)})
            await until(marker.exists)
            with pytest.raises(ServerError):
                await server.deploy('engine', {'fail': True})
            assert not first.done() and engine.closed == 0
            assert (await server.deploy('engine', {'text': 'after-failure'}))['outputs']['text'] == 'after-failure'
            stopping = asyncio.create_task(server.stop_entry('engine'))
            await until(lambda: server.entries['engine']['status'] == 'stopping')
            assert not stopping.done() and engine.closed == 0
            release.touch()
            await first
            await stopping
            assert engine.closed == 1
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_engine_mode_requires_explicit_async_code_and_rejects_shared_graph(tmp_path):
    from test_server_api import text_graph
    async def scenario():
        for source in ({'workflow': text_graph()}, {'deploy': lambda values: values}):
            server = Server(name='invalid', resources=tmp_path, data_dir=tmp_path, config=engine_config(), **source)
            try:
                with pytest.raises(ValueError, match='Graph|async'):
                    await server.start()
                assert server.entries['invalid']['status'] == 'failed'
            finally:
                await server.close()
    asyncio.run(scenario())


def test_q05_second_stream_finishes_while_first_stream_is_still_open(tmp_path):
    from test_streaming import live_server, parse_events
    import requests
    server = Server(name='engine', handler_factory=build_streaming_engine, resources=tmp_path, data_dir=tmp_path, config=engine_config(stream=True))
    release = tmp_path / 'release'
    with live_server(server) as address:
        try:
            with requests.post(address + '/v1/server/engine/deploy', json={'mode': 'stream',
                'inputs': {'release': str(release), 'text': 'long'}}, stream=True, timeout=5) as first:
                assert first.status_code == 200, first.text
                identity = first.headers['x-task-id']
                lines = first.iter_lines(chunk_size=1)
                while next(lines) != b'id: 2':
                    pass
                second = requests.post(address + '/v1/server/engine/deploy', json={'mode': 'stream', 'inputs': {'text': 'short'}}, timeout=5)
                events = parse_events(second.text)
                assert events[-1]['data']['state'] == 'succeeded'
                assert events[2]['data']['data']['text'] == 'short'
                assert server.db.get_task(identity)['state'] == 'running'
                assert server.entries['engine']['workers'][0].executor.handler.peak == 2
        finally:
            release.touch()


def test_engine_cooperative_deadline_is_per_request(tmp_path):
    config = engine_config(cooperative=True)
    config['execution']['execution_timeout_seconds'] = .2
    server = Server(name='engine', handler_factory=build_engine, resources=tmp_path, data_dir=tmp_path, config=config)
    async def scenario():
        await server.start()
        try:
            _, long = await server.submit('engine', {'release': str(tmp_path / 'never')})
            assert (await server.deploy('engine', {'text': 'short'}))['outputs']['text'] == 'short'
            with pytest.raises(ServerError) as expired:
                await asyncio.wait_for(asyncio.shield(long), 3)
            assert expired.value.code == 'execution_timeout'
            assert server.entries['engine']['workers'][0].executor.handler.active == 0
        finally:
            await server.close()
    asyncio.run(scenario())


def test_engine_event_failure_keeps_capacity_until_uncancellable_work_finishes(tmp_path, monkeypatch):
    release, marker = tmp_path / 'release', tmp_path / 'marker'
    server = Server(name='engine', handler_factory=build_streaming_engine, resources=tmp_path, data_dir=tmp_path,
                    config=engine_config(stream=True, concurrency=1))
    def fail(*args, **kwargs):
        raise RuntimeError('event store unavailable')
    monkeypatch.setattr(server.db, 'append_task_event', fail)
    async def scenario():
        await server.start()
        try:
            identity, future = await server.submit('engine', {'release': str(release), 'marker': str(marker)})
            await until(marker.exists)
            assert not future.done() and server.db.get_task(identity)['state'] == 'running'
            with pytest.raises(ServerError) as full:
                await server.submit('engine', {})
            assert full.value.status_code == 429
            release.touch()
            with pytest.raises(ServerError) as failed:
                await future
            assert failed.value.code == 'event_recording_failed'
            assert server.db.get_task(identity)['state'] == 'failed'
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_cooperative_cancel_waits_for_engine_cleanup_before_releasing_slot(tmp_path):
    started, cleaning, released = threading.Event(), threading.Event(), threading.Event()
    async def deploy(values):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            while not released.is_set():
                await asyncio.sleep(.01)
    server = Server(name='engine', deploy=deploy, resources=tmp_path, data_dir=tmp_path, config=engine_config(concurrency=1, cooperative=True))
    async def scenario():
        await server.start()
        try:
            identity, future = await server.submit('engine', {})
            await until(started.is_set)
            await server.cancel_task(identity)
            await until(cleaning.is_set)
            assert not future.done() and server.db.get_task(identity)['state'] == 'running'
            with pytest.raises(ServerError) as full:
                await server.submit('engine', {})
            assert full.value.status_code == 429
            # A second cancellation must not interrupt the actual cleanup.
            await server.cancel_task(identity)
            assert not future.done()
            released.set()
            with pytest.raises(ServerError) as stopped:
                await future
            assert stopped.value.code == 'task_cancelled'
            assert server.db.get_task(identity)['state'] == 'cancelled'
        finally:
            released.set()
            await server.close()
    asyncio.run(scenario())


def test_stream_failure_cleanup_is_not_interrupted_by_later_deadline(tmp_path, monkeypatch):
    cleaning, release = threading.Event(), threading.Event()
    async def stream(values):
        try:
            yield {'text': 'first'}
        finally:
            cleaning.set()
            while not release.is_set():
                await asyncio.sleep(.01)
    config = engine_config(concurrency=1, cooperative=True, stream=True)
    config['streaming']['max_duration_seconds'] = .1
    server = Server(name='engine', deploy=stream, resources=tmp_path, data_dir=tmp_path, config=config)
    def unavailable(*args, **kwargs):
        raise RuntimeError('event write unavailable')
    monkeypatch.setattr(server.db, 'append_task_event', unavailable)
    async def scenario():
        await server.start()
        try:
            identity, future = await server.submit('engine', {})
            await until(cleaning.is_set)
            await asyncio.sleep(.2)
            await server.cancel_task(identity)
            assert not future.done() and server.db.get_task(identity)['state'] == 'running'
            with pytest.raises(ServerError) as full:
                await server.submit('engine', {})
            assert full.value.status_code == 429
            release.set()
            with pytest.raises(ServerError) as failed:
                await future
            assert failed.value.code == 'event_recording_failed'
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())

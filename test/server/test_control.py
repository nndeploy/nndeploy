"""Control operations own their full lifetime, including persistence and drain."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import threading
import uuid

import httpx
import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_db import db
from test_engine import until
from test_server_api import text_graph


def configuration(text):
    return {'name': 'echo', 'inputs': {'text': {'type': 'string', 'node': ['source'], 'required': False, 'default': text}},
            'outputs': {'text': {'type': 'string', 'edge': 'text'}}}


def test_conditional_config_writes_reject_a_competing_writer(db):
    scope = uuid.uuid4().hex
    original = db.save_server_config(scope, 'name', {'config': {'value': 'old'}})
    barrier = threading.Barrier(2)
    def write(value):
        barrier.wait(5)
        try:
            token = db.save_server_config(scope, 'name', {'config': {'value': value}}, checked=True, expected_token=original)
            return value, token
        except ServerError as exc:
            assert exc.code == 'configuration_conflict'
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [result for result in pool.map(write, ['first', 'second']) if result is not None]
    assert len(winners) == 1
    saved = db.get_server_config(scope, 'name')
    assert saved == {'data': {'config': {'value': winners[0][0]}}, 'write_token': winners[0][1]}
    assert db.load_server_configs(scope, with_tokens=True) == [saved]


def test_same_config_write_can_be_retried_concurrently_without_overwrite(db):
    scope, token = uuid.uuid4().hex, uuid.uuid4().hex
    data = {'config': {'value': 'one operation'}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: db.save_server_config(scope, 'name', data, checked=True,
                                                               expected_token=None, write_token=token), range(8)))
    assert results == [token] * 8
    with pytest.raises(ServerError) as conflict:
        db.save_server_config(scope, 'name', {'config': {'value': 'different'}}, checked=True, write_token=token)
    assert conflict.value.code == 'configuration_conflict'
    assert db.get_server_config(scope, 'name')['data'] == data


def test_duplicate_start_joins_initialization_and_waiter_cancel_does_not_own_it(tmp_path):
    entered, release = threading.Event(), threading.Event()
    handlers = []
    class Handler:
        def __init__(self):
            handlers.append(self)
        def init(self):
            entered.set()
            assert release.wait(5)
        def deploy(self, values):
            return values
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        first = asyncio.create_task(server.start_entry('default'))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            second = asyncio.create_task(server.start_entry('default'))
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            assert (await second)['status'] == 'ready'
            assert len(handlers) == 1
            assert (await server.deploy('default', {'value': 1}))['outputs'] == {'value': 1}
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_reload_blocks_other_controls_until_old_worker_drains_but_stays_ready(tmp_path):
    entered, release = threading.Event(), threading.Event()
    handlers = []
    class Handler:
        def __init__(self):
            self.number = len(handlers)
            handlers.append(self)
        def deploy(self, values):
            if values.get('block'):
                entered.set()
                assert release.wait(10)
            return {'number': self.number}
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        _, pending = await server.submit('default', {'block': True})
        assert await asyncio.to_thread(entered.wait, 5)
        reloading = asyncio.create_task(server.reload_entry('default'))
        try:
            await until(lambda: server.describe('default')['draining_workers'] == 1)
            assert server.describe('default')['status'] == 'reloading'
            for operation in (server.reload_entry, server.start_entry, server.stop_entry):
                with pytest.raises(ServerError) as busy:
                    await operation('default')
                assert busy.value.code == 'entry_busy'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                assert (await client.get('/health/ready')).status_code == 200
            assert (await server.deploy('default', {}))['outputs']['number'] == 1
            assert len(handlers) == 2
            release.set()
            assert (await pending)['outputs']['number'] == 0
            await reloading
            assert server.describe('default')['draining_workers'] == 0
            assert server.describe('default')['status'] == 'ready'
        finally:
            release.set()
            await reloading
            await server.close()
    asyncio.run(scenario())


def test_duplicate_stop_waits_for_one_release_even_when_first_waiter_cancels(tmp_path):
    entered, release = threading.Event(), threading.Event()
    closed = []
    class Handler:
        def deploy(self, values):
            return values
        def deinit(self):
            closed.append(1)
            entered.set()
            assert release.wait(5)
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        first = asyncio.create_task(server.stop_entry('default'))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            second = asyncio.create_task(server.stop_entry('default'))
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            with pytest.raises(ServerError) as refusal:
                await server.submit('default', {})
            assert refusal.value.code == 'not_ready'
            release.set()
            assert (await second)['status'] == 'stopped'
            assert closed == [1]
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_stop_is_persisted_before_worker_release_and_restart_observes_it(tmp_path):
    server = Server(workflow=text_graph(), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        entered, release = asyncio.Event(), asyncio.Event()
        worker = server.entries['echo']['workers'][0]
        close = worker.close
        async def delayed():
            entered.set()
            await release.wait()
            await close()
        worker.close = delayed
        stopping = asyncio.create_task(server.stop_entry('echo'))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not server.db.get_server_config(server.config_scope, 'echo')['data']['enabled']
            restarted = Server(resources=tmp_path, data_dir=tmp_path)
            await restarted.start()
            try:
                assert restarted.describe('echo')['status'] == 'stopped'
                assert restarted.describe('echo')['workers'] == 0
            finally:
                await restarted.close()
        finally:
            release.set()
            await stopping
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('committed', [False, True])
def test_config_write_retry_resolves_lost_ack_with_one_operation(tmp_path, monkeypatch, committed):
    server = Server(workflow=text_graph(), config=configuration('old'), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        save = server.db.save_server_config
        calls = []
        def lost(*args, **kwargs):
            calls.append(kwargs['write_token'])
            if len(calls) == 1:
                if committed:
                    save(*args, **kwargs)
                raise ConnectionError('configuration acknowledgement lost')
            return save(*args, **kwargs)
        monkeypatch.setattr(server.db, 'save_server_config', lost)
        try:
            await asyncio.wait_for(server.reload_entry('echo', config=configuration('new')), 5)
            assert len(calls) == 2 and len(set(calls)) == 1
            assert server.db.get_server_config(server.config_scope, 'echo')['write_token'] == calls[0]
            assert (await server.deploy('echo', {}))['outputs']['text'] == 'new'
        finally:
            await server.close()
    asyncio.run(scenario())


def test_database_outage_keeps_old_graph_until_configuration_is_confirmed(tmp_path, monkeypatch):
    server = Server(workflow=text_graph(), config=configuration('old'), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        save = server.db.save_server_config
        calls, recovered = [], threading.Event()
        def unavailable(*args, **kwargs):
            calls.append(kwargs['write_token'])
            if not recovered.is_set():
                raise ConnectionError('temporary config database failure')
            return save(*args, **kwargs)
        monkeypatch.setattr(server.db, 'save_server_config', unavailable)
        reloading = asyncio.create_task(server.reload_entry('echo', config=configuration('new')))
        try:
            await until(lambda: len(calls) >= 2)
            assert not reloading.done()
            assert (await server.deploy('echo', {}))['outputs']['text'] == 'old'
            assert server.describe('echo')['status'] == 'reloading'
            recovered.set()
            await asyncio.wait_for(reloading, 5)
            assert len(set(calls)) == 1
            assert (await server.deploy('echo', {}))['outputs']['text'] == 'new'
        finally:
            recovered.set()
            await reloading
            await server.close()
    asyncio.run(scenario())


def test_conflicting_config_change_does_not_overwrite_or_stop_live_graph(tmp_path):
    server = Server(workflow=text_graph(), config=configuration('old'), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        try:
            foreign = copy.deepcopy(server.db.get_server_config(server.config_scope, 'echo')['data'])
            foreign['config'] = configuration('another Server')
            server.db.save_server_config(server.config_scope, 'echo', foreign)
            for operation in (lambda: server.reload_entry('echo', config=configuration('my replacement')),
                              lambda: server.stop_entry('echo')):
                with pytest.raises(ServerError) as conflict:
                    await operation()
                assert conflict.value.code == 'configuration_conflict'
                assert (await server.deploy('echo', {}))['outputs']['text'] == 'old'
                assert server.db.get_server_config(server.config_scope, 'echo')['data'] == foreign
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('committed', [False, True])
def test_shutdown_during_unknown_config_commit_reports_uncertainty_and_releases_workers(tmp_path, monkeypatch, committed):
    server = Server(workflow=text_graph(), config=configuration('old'), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        save = server.db.save_server_config
        calls = []
        def lost(*args, **kwargs):
            calls.append(1)
            if committed and len(calls) == 1:
                save(*args, **kwargs)
            raise ConnectionError('database unavailable')
        monkeypatch.setattr(server.db, 'save_server_config', lost)
        reloading = asyncio.create_task(server.reload_entry('echo', config=configuration('new')))
        await until(lambda: bool(calls))
        await asyncio.wait_for(server.close(), 5)
        with pytest.raises(ServerError) as unknown:
            await reloading
        assert unknown.value.code == 'configuration_unconfirmed'
        assert not server.started and not server.entries['echo']['workers']
        restored = Server(resources=tmp_path, data_dir=tmp_path)
        await restored.start()
        try:
            assert (await restored.deploy('echo', {}))['outputs']['text'] == ('new' if committed else 'old')
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_shutdown_attempts_every_worker_and_closes_dependencies_despite_release_error(tmp_path, monkeypatch):
    released = []
    class Broken:
        def deploy(self, values):
            return values
        def deinit(self):
            raise RuntimeError('unconfirmed release')
    class Good(Broken):
        def deinit(self):
            released.append('worker')
    server = Server(name='broken', handler_factory=Broken, resources=tmp_path, data_dir=tmp_path)
    server.add(name='good', handler_factory=Good)
    db_close, storage_close = server.db.close, server.storage.close
    def close_db():
        released.append('database')
        db_close()
    def close_storage():
        released.append('storage')
        storage_close()
    monkeypatch.setattr(server.db, 'close', close_db)
    monkeypatch.setattr(server.storage, 'close', close_storage)
    async def scenario():
        await server.start()
        with pytest.raises(RuntimeError, match='unconfirmed'):
            await server.close()
        assert set(released) == {'worker', 'database', 'storage'}
        assert not server.started
        assert server.entries['broken']['status'] == 'failed'
        assert server.entries['good']['status'] == 'stopped'
    asyncio.run(scenario())


def test_server_start_waiters_share_initialization_and_cancellation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    initialized = []
    class Handler:
        def init(self):
            initialized.append(1)
            entered.set()
            assert release.wait(5)
        def deploy(self, values):
            return values
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        first = asyncio.create_task(server.start())
        second = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            second = asyncio.create_task(server.start())
            await asyncio.sleep(.05)
            assert not second.done() and not server.started
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            release.set()
            await asyncio.wait_for(second, 5)
            assert server.started and initialized == [1]
            assert (await server.deploy('default', {'value': 1}))['outputs'] == {'value': 1}
        finally:
            release.set()
            if second is not None:
                await second
            await server.close()
    asyncio.run(scenario())


def test_close_during_start_is_owned_and_rejects_new_controls(tmp_path):
    entered, release = threading.Event(), threading.Event()
    closed = []
    class Handler:
        def init(self):
            entered.set()
            assert release.wait(5)
        def deploy(self, values):
            return values
        def deinit(self):
            closed.append(1)
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        starting = asyncio.create_task(server.start())
        assert await asyncio.to_thread(entered.wait, 5)
        first = asyncio.create_task(server.close())
        try:
            await until(lambda: server.shutting_down)
            second = asyncio.create_task(server.close())
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            for operation in (server.start, lambda: server.start_entry('default')):
                with pytest.raises(ServerError) as rejected:
                    await operation()
                assert rejected.value.code == 'server_stopping'
            assert not second.done()
            release.set()
            await asyncio.wait_for(second, 5)
            await starting
            assert closed == [1] and not server.started
            assert server.entries['default']['status'] == 'stopped'
            assert server.task_maintenance.done()
            await server.close()
            assert closed == [1]
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_close_joins_every_worker_of_one_entry_before_reporting_failure(tmp_path):
    entered, release = threading.Event(), threading.Event()
    created, closed = [], []
    class Handler:
        def __init__(self):
            self.number = len(created)
            created.append(self)
        def deploy(self, values):
            return values
        def deinit(self):
            if self.number == 0:
                raise RuntimeError('first release failed')
            entered.set()
            assert release.wait(5)
            closed.append(self.number)
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path, config={'execution': {'workers': 2}})
    async def scenario():
        await server.start()
        closing = asyncio.create_task(server.close())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert not closing.done()
            release.set()
            with pytest.raises(RuntimeError, match='first release failed'):
                await closing
            assert closed == [1]
            assert server.entries['default']['status'] == 'failed'
            with pytest.raises(RuntimeError, match='first release failed'):
                await server.close()
            assert closed == [1]
        finally:
            release.set()
            await asyncio.gather(closing, return_exceptions=True)
    asyncio.run(scenario())


def test_failed_old_worker_release_blocks_more_reloads_and_is_tracked_at_shutdown(tmp_path):
    created, closed = [], []
    class Handler:
        def __init__(self):
            self.number = len(created)
            created.append(self)
        def deploy(self, values):
            return {'number': self.number}
        def deinit(self):
            closed.append(self.number)
            if self.number == 0:
                raise RuntimeError('old worker release failed')
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        try:
            with pytest.raises(RuntimeError, match='old worker release failed'):
                await server.reload_entry('default')
            assert server.describe('default')['draining_workers'] == 1
            assert server.describe('default')['error'] == 'Previous worker release was not confirmed'
            assert (await server.deploy('default', {}))['outputs'] == {'number': 1}
            for operation in (server.reload_entry, server.start_entry):
                with pytest.raises(ServerError) as refused:
                    await operation('default')
                assert refused.value.code == 'entry_busy'
            assert len(created) == 2
        finally:
            with pytest.raises(RuntimeError, match='old worker release failed'):
                await server.close()
        assert closed == [0, 1]
        assert server.entries['default']['status'] == 'stopped'
    asyncio.run(scenario())


def test_restore_uses_token_from_the_same_configuration_read(tmp_path, monkeypatch):
    server = Server(workflow=text_graph(), config=configuration('old'), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        await server.close()
        restored = Server(resources=tmp_path, data_dir=tmp_path)
        load = restored.db.load_server_configs
        changed = []
        def concurrent_update(*args, **kwargs):
            records = load(*args, **kwargs)
            replacement = copy.deepcopy(records[0]['data'])
            replacement['config'] = configuration('another Server')
            changed.append(restored.db.save_server_config(restored.config_scope, 'echo', replacement))
            return records
        monkeypatch.setattr(restored.db, 'load_server_configs', concurrent_update)
        try:
            with pytest.raises(ServerError) as conflict:
                await restored.start()
            assert conflict.value.code == 'configuration_conflict'
            actual = restored.db.get_server_config(restored.config_scope, 'echo')
            assert actual['write_token'] == changed[0]
            assert actual['data']['config']['inputs']['text']['default'] == 'another Server'
            assert restored.entries['echo']['status'] == 'failed'
            assert not restored.entries['echo']['workers']
        finally:
            await restored.close()
    asyncio.run(scenario())

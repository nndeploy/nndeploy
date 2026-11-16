"""Retry database cleanup after physical shutdown without reusing live models."""
import asyncio

import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_db import db
from test_config_files import file_record, assert_protected
from test_redis_queue import backend, redis_service


def echo(inputs):
    return inputs


@pytest.mark.parametrize('committed', [False, True], ids=['before-commit', 'lost-receipt'])
def test_stop_retries_file_release_without_repeating_model_deinit(db, tmp_path, monkeypatch, committed):
    closed = []
    class Handler:
        def deploy(self, inputs):
            return inputs
        def deinit(self):
            closed.append(True)
    server = Server(name='echo', handler_factory=Handler, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start()
        entry = server.entries['echo']
        identity = file_record(db)
        db.hold_files(entry['file_holder'], {'file_id': identity})
        release = server.db.release_files
        calls = []
        def unreliable(holder):
            if holder == entry['file_holder']:
                calls.append(holder)
                if len(calls) == 1:
                    if committed:
                        release(holder)
                    raise OSError('Controlled release failure')
            return release(holder)
        monkeypatch.setattr(server.db, 'release_files', unreliable)
        try:
            with pytest.raises(OSError, match='Controlled'):
                await server.stop_entry('echo')
            assert closed == [True]
            assert entry['status'] == 'failed' and entry['workers']
            assert not server.accepting('echo')
            if not committed:
                assert_protected(db, identity)
            with pytest.raises(ServerError):
                await server.start_entry('echo')
            stopped = await server.stop_entry('echo')
            assert stopped['status'] == 'stopped' and stopped['error'] is None
            assert closed == [True] and len(calls) == 2
            db.mark_file_deleting(identity)
            await server.start_entry('echo')
            assert (await server.deploy('echo', {}))['outputs'] == {}
        finally:
            monkeypatch.setattr(server.db, 'release_files', release)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('committed', [False, True], ids=['before-commit', 'lost-receipt'])
def test_shared_stop_retains_runtime_identity_until_retirement_confirmed(backend, tmp_path, monkeypatch, committed):
    database, redis_config = backend
    server = Server(name='echo', deploy=echo, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    async def scenario():
        await server.start()
        entry = server.entries['echo']
        guard = entry['control_guard']
        retire = server.shared_runtime.control.retire_runtime
        calls = []
        def unreliable(token):
            if token == guard.path.name:
                calls.append(token)
                if len(calls) == 1:
                    if committed:
                        retire(token)
                    raise OSError('Controlled retirement failure')
            return retire(token)
        monkeypatch.setattr(server.shared_runtime.control, 'retire_runtime', unreliable)
        try:
            with pytest.raises(OSError, match='Controlled'):
                await server.stop_entry('echo')
            assert entry.get('control_guard') is guard
            assert guard.file.closed
            assert (await server.stop_entry('echo'))['status'] == 'stopped'
            assert 'control_guard' not in entry
            assert len(calls) == 2
            await server.start_entry('echo')
            assert entry['control_guard'].path != guard.path
            assert (await server.deploy('echo', {}))['outputs'] == {}
        finally:
            monkeypatch.setattr(server.shared_runtime.control, 'retire_runtime', retire)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


def test_stop_retry_never_converts_unconfirmed_physical_release_to_success(db, tmp_path):
    closed = []
    class Handler:
        def deploy(self, inputs):
            return inputs
        def deinit(self):
            closed.append(True)
            if len(closed) == 1:
                raise RuntimeError('Physical release was not confirmed')
    server = Server(name='echo', handler_factory=Handler, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start()
        entry = server.entries['echo']
        identity = file_record(db)
        db.hold_files(entry['file_holder'], {'file_id': identity})
        try:
            for _ in range(2):
                with pytest.raises(RuntimeError, match='Physical release'):
                    await server.stop_entry('echo')
                assert closed == [True]
                assert entry['status'] == 'failed' and entry['workers']
                assert_protected(db, identity)
            with pytest.raises(ServerError):
                await server.start_entry('echo')
        finally:
            with pytest.raises(RuntimeError, match='Physical release'):
                await server.close()
    asyncio.run(scenario())


def test_retried_stop_is_owned_and_coalesces_cancelled_waiters(db, tmp_path, monkeypatch):
    import threading
    entered, resume = threading.Event(), threading.Event()
    server = Server(name='echo', deploy=echo, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start()
        entry = server.entries['echo']
        release = server.db.release_files
        calls = []
        def unreliable(holder):
            if holder == entry['file_holder']:
                calls.append(holder)
                if len(calls) == 1:
                    raise OSError('First cleanup failed')
                entered.set()
                assert resume.wait(5)
            return release(holder)
        monkeypatch.setattr(server.db, 'release_files', unreliable)
        observers = []
        try:
            with pytest.raises(OSError):
                await server.stop_entry('echo')
            first = asyncio.create_task(server.stop_entry('echo'))
            observers.append(first)
            assert await asyncio.to_thread(entered.wait, 3)
            second = asyncio.create_task(server.stop_entry('echo'))
            observers.append(second)
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not second.done() and len(calls) == 2
            resume.set()
            assert (await second)['status'] == 'stopped'
            assert len(calls) == 2
        finally:
            resume.set()
            await asyncio.gather(*observers, return_exceptions=True)
            monkeypatch.setattr(server.db, 'release_files', release)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


def check_reload_cleanup_retry(server, monkeypatch):
    async def scenario():
        await server.start()
        old = server.entries['echo']
        release = server.db.release_files
        failed = False
        def unreliable(holder):
            nonlocal failed
            if holder == old['file_holder'] and not failed:
                failed = True
                raise OSError('Old runtime cleanup failed')
            return release(holder)
        monkeypatch.setattr(server.db, 'release_files', unreliable)
        try:
            with pytest.raises(OSError, match='Old runtime'):
                await server.reload_entry('echo')
            assert old in server.retiring_entries['echo']
            assert (await server.deploy('echo', {}))['outputs'] == {}
            with pytest.raises(ServerError):
                await server.reload_entry('echo')
            assert (await server.stop_entry('echo'))['status'] == 'stopped'
            assert not server.retiring_entries['echo']
            assert not old['workers']
            await server.start_entry('echo')
            assert (await server.deploy('echo', {}))['outputs'] == {}
        finally:
            monkeypatch.setattr(server.db, 'release_files', release)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


def test_stop_retries_old_reload_cleanup(db, tmp_path, monkeypatch):
    server = Server(name='echo', deploy=echo, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    check_reload_cleanup_retry(server, monkeypatch)


def test_shared_stop_retries_old_reload_cleanup(backend, tmp_path, monkeypatch):
    database, redis_config = backend
    server = Server(name='echo', deploy=echo, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    check_reload_cleanup_retry(server, monkeypatch)


@pytest.mark.parametrize('action', ['stop', 'start'])
def test_failed_initialization_retries_cleanup_before_replacing_its_file_holder(db, tmp_path, monkeypatch, action):
    attempts = []
    class Handler:
        def init(self):
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError('Controlled initialization failure')
        def deploy(self, inputs):
            return inputs
    server = Server(name='echo', handler_factory=Handler, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    identity = file_record(db)
    hold, release = server.db.hold_files, server.db.release_files
    unavailable = True
    def hold_extra(holder, data):
        hold(holder, {'file_id': identity})
        return hold(holder, data)
    def unreliable(holder):
        if unavailable:
            raise OSError('Cleanup is unavailable')
        return release(holder)
    monkeypatch.setattr(server.db, 'hold_files', hold_extra)
    monkeypatch.setattr(server.db, 'release_files', unreliable)
    async def scenario():
        nonlocal unavailable
        # Initialize through the entry control used by the editor, keeping the
        # application alive when this one model fails to load.
        try:
            with pytest.raises(RuntimeError, match='initialization'):
                await server.start_entry('echo')
            holder = server.entries['echo']['file_holder']
            assert_protected(db, identity)
            if action == 'start':
                with pytest.raises(OSError, match='Cleanup'):
                    await server.start_entry('echo')
                assert server.entries['echo']['file_holder'] == holder
                assert attempts == [True]
            unavailable = False
            monkeypatch.setattr(server.db, 'hold_files', hold)
            if action == 'stop':
                assert (await server.stop_entry('echo'))['status'] == 'stopped'
            else:
                await server.start_entry('echo')
                assert (await server.deploy('echo', {}))['outputs'] == {}
            db.mark_file_deleting(identity)
        finally:
            unavailable = False
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


def test_rejected_shared_reload_keeps_failed_cleanup_reachable(backend, tmp_path, monkeypatch):
    from nndeploy.server.worker import Worker
    database, redis_config = backend
    def config(value):
        return {'inputs': {'value': {'type': 'string', 'required': False, 'default': value}}}
    server = Server(name='echo', deploy=echo, config=config('old'), data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    async def scenario():
        await server.start_entry('echo')
        old = server.entries['echo']
        old_token = old['control_guard'].path.name
        start, retire = Worker.start, server.shared_runtime.control.retire_runtime
        async def fail_candidate(worker, **kwargs):
            await start(worker, **kwargs)
            if worker.config.inputs['value'].default == 'new':
                raise RuntimeError('Candidate initialization failed')
        def unavailable(token):
            if token != old_token:
                raise OSError('Candidate retirement unavailable')
            return retire(token)
        monkeypatch.setattr(Worker, 'start', fail_candidate)
        monkeypatch.setattr(server.shared_runtime.control, 'retire_runtime', unavailable)
        try:
            with pytest.raises(OSError, match='Candidate retirement'):
                await server.reload_entry('echo', config=config('new'))
            assert server.entries['echo'] is old
            assert (await server.deploy('echo', {}))['outputs'] == {'value': 'old'}
            pending = server.retiring_entries.get('echo', [])
            assert len(pending) == 1, 'Failed cleanup must remain owned even after all candidate Workers close'
            assert pending[0].get('control_guard') is not None
            assert not pending[0]['workers']
            monkeypatch.setattr(Worker, 'start', start)
            monkeypatch.setattr(server.shared_runtime.control, 'retire_runtime', retire)
            assert (await server.stop_entry('echo'))['status'] == 'stopped'
            assert not server.retiring_entries['echo']
            await server.start_entry('echo')
            await server.reload_entry('echo', config=config('new'))
            assert (await server.deploy('echo', {}))['outputs'] == {'value': 'new'}
        finally:
            monkeypatch.setattr(Worker, 'start', start)
            monkeypatch.setattr(server.shared_runtime.control, 'retire_runtime', retire)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


def test_rejected_local_reload_keeps_failed_cleanup_reachable(db, tmp_path, monkeypatch):
    from nndeploy.server.worker import Worker
    server = Server(name='echo', deploy=echo, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start_entry('echo')
        old = server.entries['echo']
        start, release = Worker.start, server.db.release_files
        async def fail_candidate(worker, **kwargs):
            await start(worker, **kwargs)
            raise RuntimeError('Candidate initialization failed')
        def unavailable(holder):
            if holder != old['file_holder']:
                raise OSError('Candidate cleanup unavailable')
            return release(holder)
        monkeypatch.setattr(Worker, 'start', fail_candidate)
        monkeypatch.setattr(server.db, 'release_files', unavailable)
        try:
            with pytest.raises(RuntimeError, match='Candidate initialization'):
                await server.reload_entry('echo')
            assert server.entries['echo'] is old
            assert (await server.deploy('echo', {}))['outputs'] == {}
            pending = server.retiring_entries.get('echo', [])
            assert len(pending) == 1 and pending[0]['release_pending']
            assert not pending[0]['workers']
            with pytest.raises(ServerError):
                await server.reload_entry('echo')
            monkeypatch.setattr(Worker, 'start', start)
            monkeypatch.setattr(server.db, 'release_files', release)
            assert (await server.stop_entry('echo'))['status'] == 'stopped'
            assert not server.retiring_entries['echo']
            await server.start_entry('echo')
            await server.reload_entry('echo')
            assert (await server.deploy('echo', {}))['outputs'] == {}
        finally:
            monkeypatch.setattr(Worker, 'start', start)
            monkeypatch.setattr(server.db, 'release_files', release)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


def test_http_stop_retry_preserves_failed_receipt_and_allows_a_new_attempt(db, tmp_path, monkeypatch):
    import httpx
    server = Server(name='echo', deploy=echo, data_dir=tmp_path, admin_token='test-admin',
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start()
        entry = server.entries['echo']
        release = server.db.release_files
        calls = []
        def unreliable(holder):
            if holder == entry['file_holder']:
                calls.append(holder)
                if len(calls) == 1:
                    raise OSError('File cleanup temporarily unavailable')
            return release(holder)
        monkeypatch.setattr(server.db, 'release_files', unreliable)
        auth = {'Authorization': 'Bearer test-admin'}
        headers = dict(auth, **{'Idempotency-Key': 'first-stop'})
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                first = await client.post('/api/server/echo/stop', headers=headers)
                assert first.status_code == 500
                repeated = await client.post('/api/server/echo/stop', headers=headers)
                assert repeated.content == first.content and len(calls) == 1
                second = await client.post('/api/server/echo/stop', headers=dict(auth, **{'Idempotency-Key': 'retry-stop'}))
                assert second.status_code == 200 and second.json()['status'] == 'stopped'
                assert len(calls) == 2
                assert (await client.get(first.headers['location'], headers=auth)).content == first.content
                assert (await client.post('/api/server/echo/start', headers=auth)).status_code == 200
                assert (await server.deploy('echo', {}))['outputs'] == {}
        finally:
            monkeypatch.setattr(server.db, 'release_files', release)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())

"""Real controller deaths, isolated children and durable candidate cleanup."""
import asyncio
import multiprocessing
import os
from pathlib import Path
import time
import uuid
import sys

import pytest
from sqlalchemy import event, select, update, text

from nndeploy.server import Server
from nndeploy.server.control_guard import ControlGuard
from nndeploy.server.db import DB, file_holders
from nndeploy.server.shared import workers
from test_db import db
from test_redis_queue import backend, redis_service
from test_shared_configs import initial, prepare, ready
from test_shared_reload import config
from test_shared_server import shared_deploy, until
from test_config_files import file_record, assert_protected


pytestmark = pytest.mark.skipif(sys.platform not in ('linux', 'darwin'),
                                reason='Physical recovery currently requires a Linux or macOS kernel boot UUID')


def test_copied_lock_is_not_exit_evidence_and_gc_does_not_release_a_live_guard(tmp_path):
    import gc
    import shutil
    import weakref
    token, holder = uuid.uuid4().hex, uuid.uuid4().hex
    guard = ControlGuard.create(tmp_path / 'original', token, holder)
    reference = weakref.ref(guard)
    copy = tmp_path / 'copy' / 'control-locks' / token
    copy.parent.mkdir(parents=True)
    shutil.copy2(guard.path, copy)
    del guard
    gc.collect()
    try:
        assert reference() is not None
        with ControlGuard.recover(tmp_path / 'original', token, holder) as exited:
            assert not exited
        with ControlGuard.recover(tmp_path / 'copy', token, holder) as exited:
            assert not exited
    finally:
        reference().close()
    with ControlGuard.recover(tmp_path / 'original', token, holder) as exited:
        assert exited


def test_lock_subsystem_failure_closes_handle_without_revoking_initialization(tmp_path, monkeypatch):
    import nndeploy.server.control_guard as module
    token, holder = uuid.uuid4().hex, uuid.uuid4().hex
    guard = ControlGuard.create(tmp_path, token, holder)
    guard.close()
    handles = []
    def unavailable(file, **kwargs):
        handles.append(file)
        raise OSError('OS locking unavailable')
    with monkeypatch.context() as patch:
        patch.setattr(module, '_lock', unavailable)
        with pytest.raises(OSError, match='OS locking'):
            with ControlGuard.recover(tmp_path, token, holder):
                pytest.fail('Unconfirmed exit was accepted')
    assert handles[0].closed
    child = ControlGuard.join(guard.path)
    child.close()
    with ControlGuard.recover(tmp_path, token, holder) as exited:
        assert exited


def pending(database, directory):
    file = file_record(database)
    entry, old, new = initial(database)
    token, holder, worker = [uuid.uuid4().hex for _ in range(3)]
    guard = ControlGuard.create(directory, token, holder)
    database.shared.begin_reload(entry['entry_key'], entry['control_token'], new, 'new', {'file_id': file},
                                 write_token=token, pending_holder=holder)
    database.hold_files(holder, {'file_id': file})
    database.shared.hold_worker(entry['entry_key'], new, worker, token=token, owner_token=token)
    return entry, new, token, holder, worker, file, guard


def test_live_owner_missing_evidence_and_unknown_worker_are_not_reclaimed(db, tmp_path):
    entry, new, token, holder, worker, file, guard = pending(db, tmp_path)
    try:
        assert not db.shared.recover_pending(entry['entry_key'], tmp_path)
        guard.close()
        assert not db.shared.recover_pending(entry['entry_key'], tmp_path / 'different-workspace')
        with db.engine.begin() as connection:
            connection.execute(update(workers).where(workers.c.holder == worker).values(owner_token=None))
        assert not db.shared.recover_pending(entry['entry_key'], tmp_path)
        assert db.shared.get(entry['entry_key'])['pending_token'] == token
        assert_protected(db, file)
        # Once physical evidence covers every worker, retry can complete.
        with db.engine.begin() as connection:
            connection.execute(update(workers).where(workers.c.holder == worker).values(owner_token=token))
        assert db.shared.recover_pending(entry['entry_key'], tmp_path)
        assert db.shared.get(entry['entry_key'])['pending_token'] is None
        db.mark_file_deleting(file)
    finally:
        guard.close()


def test_recovery_rollback_keeps_references_and_revocation_survives_retry(db, tmp_path):
    entry, new, token, holder, worker, file, guard = pending(db, tmp_path)
    guard.close()
    def fail(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('DELETE FROM server_configs'):
            raise OSError('injected cleanup transaction failure')
    event.listen(db.engine, 'before_cursor_execute', fail)
    try:
        with pytest.raises(OSError, match='cleanup transaction'):
            db.shared.recover_pending(entry['entry_key'], tmp_path)
    finally:
        event.remove(db.engine, 'before_cursor_execute', fail)
    assert db.shared.get(entry['entry_key'])['pending_token'] == token
    assert_protected(db, file)
    with db.engine.connect() as connection:
        assert set(connection.execute(select(file_holders.c.holder).where(file_holders.c.holder.in_([holder, worker]))).scalars()) == {holder, worker}
    with pytest.raises(RuntimeError, match='revoked'):
        ControlGuard.join(guard.path)
    assert db.shared.recover_pending(entry['entry_key'], tmp_path)
    assert not db.shared.recover_pending(entry['entry_key'], tmp_path)
    db.mark_file_deleting(file)


def test_recovery_is_serialized_across_competing_replicas(db, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    entry, new, token, holder, worker, file, guard = pending(db, tmp_path)
    guard.close()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: db.shared.recover_pending(entry['entry_key'], tmp_path), range(16)))
    assert results.count(True) == 1
    assert db.shared.get(entry['entry_key'])['pending_token'] is None
    db.mark_file_deleting(file)


def test_old_pending_operation_keeps_its_ownership_on_schema_upgrade(db, tmp_path):
    entry, old, new = initial(db)
    token = prepare(db, entry, new)
    holder = ready(db, entry, new, token)
    with db.engine.begin() as connection:
        connection.execute(text('ALTER TABLE server_shared_entries DROP COLUMN pending_holder'))
        connection.execute(text('ALTER TABLE server_shared_workers DROP COLUMN owner_token'))
        connection.execute(text("UPDATE alembic_version SET version_num = '0017_initial_readiness'"))
    reopened = DB(db.url).init_schema()
    try:
        state = reopened.shared.get(entry['entry_key'])
        assert state['pending_token'] == token and state['pending_holder'] is None
        assert not reopened.shared.recover_pending(entry['entry_key'], tmp_path)
        with reopened.engine.connect() as connection:
            row = connection.execute(select(workers).where(workers.c.holder == holder)).mappings().one()
            assert row['ready'] and row['owner_token'] is None
        assert reopened.shared.activate(entry['entry_key'], token)['active_queue'] == new
    finally:
        reopened.close()


def _controller(options, candidate, phase, marker):
    async def scenario():
        server = Server(name='model', deploy=shared_deploy, config=config('old'), **options)
        # Attach to existing canonical state without creating unrelated old
        # workers. This is the same stopped-entry path used by editor Update.
        await server.shared_runtime.prepare('model', server.entries['model'])
        if phase == 'ready':
            original = server.db.shared.activate
            def delayed(*args, **kwargs):
                Path(marker).write_text(str(os.getpid()))
                while True:
                    time.sleep(.05)
                return original(*args, **kwargs)
            server.db.shared.activate = delayed
        await server._control_entry('model', 'reload', lambda: server._reload_entry(
            'model', config=candidate, start_stopped=True))
    asyncio.run(scenario())


def _exited(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    proc = Path(f'/proc/{pid}/stat')
    try:
        return proc.exists() and proc.read_text().rsplit(')', 1)[1].strip().startswith('Z')
    except FileNotFoundError:
        return True


@pytest.mark.parametrize('isolation,phase,restart', [('thread', 'warmup', False), ('process', 'warmup', False),
                                                  ('process', 'ready', False), ('thread', 'warmup', True)])
def test_controller_crash_waits_for_actual_child_exit_then_unblocks_update(backend, tmp_path, isolation, phase, restart):
    database, redis = backend
    options = dict(data_dir=tmp_path / 'shared', database_url=database.url.render_as_string(hide_password=False), redis_config=redis)
    observer = Server(name='model', deploy=shared_deploy, config=config('old'), **options)
    marker, release = tmp_path / 'warmup', tmp_path / 'release'
    candidate = config('candidate', isolation)
    if phase == 'warmup':
        candidate['execution']['warmup'] = [{'inputs': {'marker': str(marker), 'release': str(release)}}]
    process = multiprocessing.get_context('spawn').Process(target=_controller, args=(options, candidate, phase, str(marker)))
    async def scenario():
        nonlocal observer
        child_pid = None
        try:
            await observer.start()
            key = observer.entries['model']['shared_key']
            old = database.shared.get(key)['active_queue']
            process.start()
            await until(marker.exists)
            pending_state = database.shared.get(key)
            assert pending_state['pending_token'] and pending_state['active_queue'] == old
            assert not database.shared.recover_pending(key, options['data_dir'])
            if isolation == 'process' and phase == 'warmup':
                child_pid = int(marker.read_text())
                assert child_pid != process.pid
            if restart:
                await observer.stop_entry('model')
                await observer.close()
            process.kill()
            await asyncio.to_thread(process.join, 5)
            assert not process.is_alive()
            if child_pid is not None:
                assert not _exited(child_pid)
                # The parent has really exited, but the child still owns the
                # warmup. Neither heartbeat expiry nor parent death permits GC.
                assert not database.shared.recover_pending(key, options['data_dir'])
                await asyncio.sleep(.3)
                assert database.shared.get(key)['pending_token'] == pending_state['pending_token']
            if not restart:
                await observer.stop_entry('model')
            release.touch()
            if child_pid is not None:
                await until(lambda: _exited(child_pid))
            if restart:
                assert database.shared.get(key)['pending_token'] is not None
                observer = Server(name='model', deploy=shared_deploy, config=config('old'), **options)
                await observer.start()
            await until(lambda: database.shared.get(key)['pending_token'] is None)
            state = database.shared.get(key)
            assert state['active_queue'] == old and not state['accepting']
            with database.engine.connect() as connection:
                assert not connection.execute(select(workers.c.holder).where(workers.c.queue_key == pending_state['pending_queue'])).first()
            # Public Update/Start can proceed after the abandoned change. The
            # previous Stop remains in effect until an explicit new action.
            await observer.start_entry('model')
            await observer.reload_entry('model', config=config('new'))
            assert (await observer.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            release.touch()
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                await asyncio.to_thread(process.join, 5)
            if child_pid is not None:
                await until(lambda: _exited(child_pid))
            await observer.close()
    asyncio.run(scenario())


def test_late_isolated_child_cannot_initialize_after_recovery(tmp_path):
    from nndeploy.server.worker import ProcessWorker
    token, holder = uuid.uuid4().hex, uuid.uuid4().hex
    guard = ControlGuard.create(tmp_path, token, holder)
    guard.close()
    with ControlGuard.recover(tmp_path, token, holder) as exited:
        assert exited
    marker = tmp_path / 'warmup'
    options = config('new', 'process')
    options['execution']['warmup'] = [{'inputs': {'marker': str(marker)}}]
    server = Server(deploy=shared_deploy, config=options, data_dir=tmp_path / 'server')
    entry = server.entries['default']
    worker = ProcessWorker(entry['source'], entry['config'], server.storage, tmp_path / 'worker')
    worker.control_guard_path = str(guard.path)
    async def scenario():
        try:
            with pytest.raises(Exception, match='initialize'):
                await worker.start()
            assert worker.pid is None and not marker.exists()
        finally:
            await worker.close()
            await server.close()
    asyncio.run(scenario())

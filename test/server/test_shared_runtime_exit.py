"""Reconcile actually exited runtimes without retrying unknown computation."""
import asyncio
import json
import multiprocessing
import os
from pathlib import Path
import sys
import uuid

import pytest
from sqlalchemy import select, update, text

from nndeploy.server import Server
from nndeploy.server.control_guard import ControlGuard
from nndeploy.server.db import DB, tasks, file_holders
from nndeploy.server.errors import ServerError
from nndeploy.server.shared import workers, runtime_records
from test_db import db
from test_redis_queue import backend, redis_service
from test_shared_configs import initial
from test_shared_reload import config
from test_shared_server import shared_deploy, until
from test_shared_crash import _exited
from test_config_files import file_record, assert_protected


pytestmark = pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='Kernel boot UUID required')


def runtime(database, directory, *, bind=True):
    entry, old, _ = initial(database)
    token, holder, worker = [uuid.uuid4().hex for _ in range(3)]
    guard = ControlGuard.create(directory, token, holder)
    database.shared.bind_runtime(entry['entry_key'], token, holder, old if bind else None)
    file = file_record(database)
    database.hold_files(holder, {'file_id': file})
    if bind:
        database.shared.hold_worker(entry['entry_key'], old, worker, ready=True, owner_token=token, runtime_holder=holder)
    row = database.shared.runtimes(entry['entry_key'])[0]
    return entry, row, worker, file, guard


def claim(database, entry, worker=None):
    identity = uuid.uuid4().hex
    database.create_shared_task(identity, 'model', entry['active_queue'], entry['group_key'], {}, {}, 30)
    return database.claim_task(identity, entry['active_queue'], uuid.uuid4().hex, 3600, holder=worker)


def test_recovery_waits_for_exit_then_fails_once_preserving_unknown_execution(db, tmp_path):
    entry, row, worker, file, guard = runtime(db, tmp_path)
    lost = claim(db, entry, worker)
    unknown = claim(db, entry)
    try:
        assert not db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
        assert db.get_task(lost['task_id'])['state'] == 'running'
        assert_protected(db, file)
        guard.close()
        assert db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
        assert not db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
        record = db.get_task(lost['task_id'])
        assert record['state'] == 'failed' and record['error']['code'] == 'worker_lost'
        assert record['attempt'] == 1 and record['worker_token'] == lost['worker_token']
        assert db.get_task(unknown['task_id'])['state'] == 'running'
        assert db.queue_usage(entry['group_key'])['inflight'] == 1
        assert [event['kind'] for event in db.read_task_events(lost['task_id'])] == ['start', 'error', 'done']
        db.mark_file_deleting(file)
        with pytest.raises(RuntimeError):
            db.finish_task(lost['task_id'], 'succeeded', db.clock(), {'late': True}, worker_token=lost['worker_token'])
    finally:
        guard.close()


def test_initializer_exit_before_worker_registration_releases_its_resources(db, tmp_path):
    entry, row, worker, file, guard = runtime(db, tmp_path, bind=False)
    try:
        assert row['queue_key'] is None
        assert not db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
        guard.close()
        assert db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
        assert not db.shared.runtimes(entry['entry_key'])
        db.mark_file_deleting(file)
    finally:
        guard.close()


def test_saved_result_wins_and_cancelled_task_is_not_retried(db, tmp_path):
    entry, row, worker, file, guard = runtime(db, tmp_path)
    completed, cancelled = claim(db, entry, worker), claim(db, entry, worker)
    db.request_cancel(cancelled['task_id'])
    path = tmp_path / 'queue-results' / entry['active_queue'] / f'{completed["task_id"]}-{completed["worker_token"]}.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(record=completed, state='succeeded', outputs={'answer': 42}, error=None)))
    # Normal shutdown removed membership, but uncommitted task ownership must
    # still locate the result and protect cleanup after a subsequent restart.
    db.release_files(worker)
    db.release_files(row['holder'])
    guard.close()
    db.shared.retire_runtime(row['token'])
    assert db.shared.runtimes(entry['entry_key'])
    assert db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
    assert db.get_task(completed['task_id'])['outputs'] == {'answer': 42}
    assert db.get_task(cancelled['task_id'])['state'] == 'cancelled'
    assert db.queue_usage(entry['group_key'])['inflight'] == 0
    assert not db.shared.runtimes(entry['entry_key'])


def test_failed_commit_keeps_fenced_ownership_and_retry_releases_each_slot_once(db, tmp_path, monkeypatch):
    entry, row, worker, file, guard = runtime(db, tmp_path)
    first, second = claim(db, entry, worker), claim(db, entry, worker)
    guard.close()
    def failure(*args, **kwargs):
        raise OSError('database unavailable')
    with monkeypatch.context() as patch:
        patch.setattr(db, 'finish_task', failure)
        with pytest.raises(OSError):
            db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
    assert_protected(db, file)
    with pytest.raises(ServerError) as fenced:
        claim(db, entry, worker)
    assert fenced.value.code == 'worker_unavailable'
    # A bounded pass leaves ownership for the next pass. No whole-runtime
    # release occurs while any accepted execution still needs reconciliation.
    assert not db.shared.recover_runtime(entry['entry_key'], row, tmp_path, limit=1)
    assert_protected(db, file)
    assert db.shared.recover_runtime(entry['entry_key'], row, tmp_path, limit=1)
    assert all(db.get_task(value['task_id'])['state'] == 'failed' for value in (first, second))
    assert db.queue_usage(entry['group_key'])['inflight'] == 1  # fenced claim remains pending
    db.mark_file_deleting(file)


def test_recovery_commit_ack_loss_does_not_duplicate_events_or_capacity_release(db, tmp_path, monkeypatch):
    entry, row, worker, file, guard = runtime(db, tmp_path)
    record = claim(db, entry, worker)
    guard.close()
    original = db.finish_task
    def lose_ack(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('lost commit acknowledgement')
    with monkeypatch.context() as patch:
        patch.setattr(db, 'finish_task', lose_ack)
        with pytest.raises(OSError):
            db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
    assert db.get_task(record['task_id'])['state'] == 'failed'
    assert db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
    assert len(db.read_task_events(record['task_id'])) == 3
    assert db.queue_usage(entry['group_key'])['inflight'] == 0


def test_late_claim_and_physical_recovery_are_serialized(db, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    for index in range(5):
        entry, row, worker, file, guard = runtime(db, tmp_path / str(index))
        identity = uuid.uuid4().hex
        db.create_shared_task(identity, 'model', entry['active_queue'], entry['group_key'], {}, {}, 30)
        guard.close()
        barrier = threading.Barrier(2)
        def late():
            barrier.wait()
            try:
                return db.claim_task(identity, entry['active_queue'], uuid.uuid4().hex, 30, holder=worker)
            except ServerError as exc:
                assert exc.code == 'worker_unavailable'
        def recover():
            barrier.wait()
            return db.shared.recover_runtime(entry['entry_key'], row, tmp_path / str(index))
        with ThreadPoolExecutor(max_workers=2) as pool:
            claimed, recovered = pool.submit(late), pool.submit(recover)
            claimed.result()
            assert recovered.result()
        state = db.get_task(identity)['state']
        assert state in ('pending', 'failed')
        assert db.queue_usage(entry['group_key'])['inflight'] == (1 if state == 'pending' else 0)
        assert not db.shared.runtimes(entry['entry_key'])


def test_corrupt_result_stays_unconfirmed_until_it_can_be_replayed(db, tmp_path):
    entry, row, worker, file, guard = runtime(db, tmp_path)
    record = claim(db, entry, worker)
    path = tmp_path / 'queue-results' / entry['active_queue'] / f'{record["task_id"]}-{record["worker_token"]}.json'
    path.parent.mkdir(parents=True)
    path.write_text('{corrupt')
    guard.close()
    with pytest.raises(ValueError):
        db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
    assert db.get_task(record['task_id'])['state'] == 'running'
    assert db.queue_usage(entry['group_key'])['inflight'] == 1
    assert_protected(db, file)
    path.write_text(json.dumps(dict(record=record, state='succeeded', outputs={'saved': True}, error=None)))
    assert db.shared.recover_runtime(entry['entry_key'], row, tmp_path)
    assert db.get_task(record['task_id'])['outputs'] == {'saved': True}


def _running_server(options, isolation, marker, release, accepted, completed=False):
    async def scenario():
        server = Server(name='model', deploy=shared_deploy, config=config('old', isolation), **options)
        await server.start()
        if completed:
            import time
            def unavailable(*args, **kwargs):
                while True:
                    time.sleep(.05)
            server.db.finish_task = unavailable
        identity, future = await server.submit('model', {'marker': marker, 'release': release}, idempotency_key='crash')
        Path(accepted).write_text(identity)
        await future
        await asyncio.Event().wait()
    asyncio.run(scenario())


def test_real_completed_result_survives_owner_death_and_redis_ack_is_repaired(backend, tmp_path):
    database, redis = backend
    settings = dict(data_dir=tmp_path / 'shared', database_url=database.url.render_as_string(hide_password=False), redis_config=redis)
    marker, accepted = tmp_path / 'executed', tmp_path / 'accepted'
    process = multiprocessing.get_context('spawn').Process(target=_running_server,
        args=(settings, 'process', str(marker), '', str(accepted), True))
    observer = Server(name='model', deploy=shared_deploy, config=config('old', 'process'), **settings)
    async def scenario():
        child_pid = None
        try:
            process.start()
            await until(lambda: accepted.exists() and marker.exists())
            identity = accepted.read_text()
            original = database.get_task(identity)
            journal = settings['data_dir'] / 'queue-results' / original['queue_key']
            await until(lambda: bool(list(journal.glob('*.json'))))
            child_pid = int(marker.read_text())
            process.kill()
            await asyncio.to_thread(process.join, 5)
            await until(lambda: _exited(child_pid))
            assert database.get_task(identity)['state'] == 'running'
            await observer.start()
            await until(lambda: database.get_task(identity)['state'] == 'succeeded')
            result = database.get_task(identity)
            assert result['outputs']['pid'] == child_pid
            assert result['worker_token'] == original['worker_token']
            assert int(marker.read_text()) == child_pid
            queue = observer.entries['model']['queue']
            for _ in range(100):
                await queue.reconcile()
                if await queue.client.hget(queue.published, identity) is None:
                    break
                await asyncio.sleep(.02)
            else:
                pytest.fail('Terminal Redis delivery was not acknowledged')
            await until(lambda: not list(journal.glob('*.json')))
            assert database.queue_usage(original['group_key'])['inflight'] == 0
            assert [value['kind'] for value in database.read_task_events(identity)] == ['start', 'result', 'done']
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                await asyncio.to_thread(process.join, 5)
            if child_pid is not None:
                await until(lambda: _exited(child_pid))
            await observer.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('isolation,cancel', [('thread', False), ('process', False), ('process', True)])
def test_real_worker_owner_crash_releases_only_after_child_exit_and_does_not_rerun(backend, tmp_path, isolation, cancel):
    database, redis = backend
    settings = dict(data_dir=tmp_path / 'shared', database_url=database.url.render_as_string(hide_password=False), redis_config=redis)
    marker, release, accepted = [tmp_path / name for name in ('running', 'release', 'accepted')]
    process = multiprocessing.get_context('spawn').Process(target=_running_server,
        args=(settings, isolation, str(marker), str(release), str(accepted)))
    observer = Server(name='model', deploy=shared_deploy, config=config('old', isolation), **settings)
    async def scenario():
        child_pid = None
        try:
            process.start()
            await until(lambda: accepted.exists() and marker.exists())
            identity = accepted.read_text()
            record = database.get_task(identity)
            assert record['runtime_token'] and record['runtime_holder']
            child_pid = int(marker.read_text()) if isolation == 'process' else None
            await observer.start()
            process.kill()
            await asyncio.to_thread(process.join, 5)
            if cancel:
                database.request_cancel(identity)
            if child_pid is not None:
                assert not _exited(child_pid)
                await asyncio.sleep(.4)
                assert database.get_task(identity)['state'] == 'running'
                assert database.queue_usage(record['group_key'])['inflight'] == 1
            release.touch()
            if child_pid is not None:
                await until(lambda: _exited(child_pid))
            await until(lambda: database.get_task(identity)['state'] in ('failed', 'cancelled'))
            final = database.get_task(identity)
            assert final['state'] == ('cancelled' if cancel else 'failed')
            assert final['error']['code'] == ('task_cancelled' if cancel else 'worker_lost')
            assert final['attempt'] == 1 and final['worker_token'] == record['worker_token']
            assert database.queue_usage(record['group_key'])['inflight'] == 0
            repeated, future = await observer.submit('model', {'marker': str(marker), 'release': str(release)}, idempotency_key='crash')
            assert repeated == identity
            with pytest.raises(ServerError):
                await future
            assert int(marker.read_text()) == (child_pid or process.pid)
            assert (await observer.deploy('model', {}))['outputs']['text'] == 'old'
            # Dead holders no longer keep old snapshots pinned across Reload.
            await observer.reload_entry('model', config=config('new', isolation))
            await observer.reload_entry('model', config=config('newer', isolation))
            assert (await observer.deploy('model', {}))['outputs']['text'] == 'newer'
            assert not database.shared.draining(observer.entries['model']['shared_key'])
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


def test_upgrade_does_not_invent_runtime_ownership_for_old_running_tasks(db, tmp_path):
    entry, old, _ = initial(db)
    record = claim(db, entry)
    with db.engine.begin() as connection:
        connection.execute(text('DROP INDEX ix_server_tasks_runtime' + (' ON server_tasks' if db.engine.dialect.name == 'mysql' else '')))
        connection.execute(text('ALTER TABLE server_tasks DROP COLUMN runtime_token'))
        connection.execute(text('ALTER TABLE server_tasks DROP COLUMN runtime_holder'))
        connection.execute(text('ALTER TABLE server_shared_workers DROP COLUMN runtime_holder'))
        connection.execute(text('DROP TABLE server_shared_runtimes'))
        connection.execute(text("UPDATE alembic_version SET version_num = '0018_control_guards'"))
    upgraded = DB(db.url).init_schema()
    try:
        old_record = upgraded.get_task(record['task_id'])
        assert old_record['state'] == 'running' and old_record['runtime_token'] is None
        assert not upgraded.shared.runtimes(entry['entry_key'])
        assert upgraded.queue_usage(entry['group_key'])['inflight'] == 1
    finally:
        upgraded.close()


def test_server_shutdown_joins_runtime_recovery_before_closing_database(backend, tmp_path, monkeypatch):
    import threading
    database, redis = backend
    server = Server(name='model', deploy=shared_deploy, config=config('old'), data_dir=tmp_path / 'shared',
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original_recover, original_close = server.db.shared.recover_runtime, server.db.close
    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        assert not closed.is_set()
        return original_recover(*args, **kwargs)
    def closing():
        closed.set()
        original_close()
    monkeypatch.setattr(server.db.shared, 'recover_runtime', slow)
    monkeypatch.setattr(server.db, 'close', closing)
    async def scenario():
        stopping = None
        try:
            await server.start()
            await until(entered.is_set)
            stopping = asyncio.create_task(server.close())
            await asyncio.sleep(.1)
            assert not stopping.done() and not closed.is_set()
            release.set()
            await stopping
            assert closed.is_set()
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_one_corrupt_result_does_not_starve_other_exited_runtimes(backend, tmp_path):
    database, redis = backend
    server = Server(name='model', deploy=shared_deploy, config=config('old'), data_dir=tmp_path / 'shared',
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis)
    async def scenario():
        try:
            await server.start()
            entry = server.entries['model']
            shared = database.shared.get(entry['shared_key'])
            records = []
            for index in (1, 2):
                token, holder, worker = f'{index:032x}', uuid.uuid4().hex, uuid.uuid4().hex
                guard = ControlGuard.create(server.data_dir, token, holder)
                try:
                    database.shared.bind_runtime(entry['shared_key'], token, holder, entry['queue'].key)
                    database.shared.hold_worker(entry['shared_key'], entry['queue'].key, worker,
                                               ready=True, owner_token=token, runtime_holder=holder)
                    record = claim(database, shared, worker)
                    records.append(record)
                    if index == 1:
                        path = server.data_dir / 'queue-results' / entry['queue'].key / f'{record["task_id"]}-{record["worker_token"]}.json'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('{corrupt')
                finally:
                    guard.close()
            for _ in range(3):
                await server.shared_runtime.cleanup(entry)
            assert database.get_task(records[0]['task_id'])['state'] == 'running'
            assert database.get_task(records[1]['task_id'])['state'] == 'failed'
            assert database.queue_usage(shared['group_key'])['inflight'] == 1
        finally:
            await server.close()
    asyncio.run(scenario())

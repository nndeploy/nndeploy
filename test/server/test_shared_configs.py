"""Durable shared Reload transactions, using real DBs and Redis queues."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

import pytest
from sqlalchemy import inspect, text

from nndeploy.server.db import DB
from nndeploy.server.errors import ServerError
from nndeploy.server.task_queue import RedisTaskQueue
from test_db import db
from test_redis_queue import backend, redis_service
from test_config_files import file_record, assert_protected


def initial(database, data=None):
    group, old, new = [uuid.uuid4().hex for _ in range(3)]
    for key in (group, old, new):
        database.register_queue(key, 8)
    entry = database.shared.register(group, 'model', old, 'old', data or {'version': 'old'})
    return entry, old, new


def prepare(database, entry, new, data=None, count=1):
    token = uuid.uuid4().hex
    database.shared.begin_reload(entry['entry_key'], entry['control_token'], new, 'new', data or {'version': 'new'},
                                 count=count, write_token=token)
    return token


def ready(database, entry, new, token):
    holder = uuid.uuid4().hex
    database.shared.hold_worker(entry['entry_key'], new, holder, token=token)
    database.shared.hold_worker(entry['entry_key'], new, holder, token=token, ready=True)
    return holder


def test_switch_fences_admission_and_old_start_without_losing_accepted_tasks(db):
    entry, old, new = initial(db)
    task = uuid.uuid4().hex
    db.create_shared_task(task, 'model', old, entry['group_key'], {}, {}, 30)
    token = prepare(db, entry, new)
    with pytest.raises(ServerError, match='warmup'):
        db.shared.activate(entry['entry_key'], token)
    with pytest.raises(ServerError) as candidate:
        db.create_shared_task(uuid.uuid4().hex, 'model', new, entry['group_key'], {}, {}, 30)
    assert candidate.value.code == 'entry_changed'
    assert db.queue_usage(entry['group_key'])['inflight'] == 1
    ready(db, entry, new, token)
    current = db.shared.activate(entry['entry_key'], token)
    assert current['active_queue'] == new and current['data'] == {'version': 'new'}
    assert db.get_task(task)['state'] == 'pending'
    for operation in (lambda: db.set_queue_accepting(old, True),
                      lambda: db.set_queue_accepting(old, False),
                      lambda: db.create_shared_task(uuid.uuid4().hex, 'model', old, entry['group_key'], {}, {}, 30)):
        with pytest.raises(ServerError) as stale:
            operation()
        assert stale.value.code == 'entry_changed'
    assert db.queue_usage(entry['group_key'])['inflight'] == 1
    assert db.claim_task(task, old, 'old-worker', 30)
    db.finish_task(task, 'succeeded', time.time(), {'version': 'old'}, worker_token='old-worker')
    db.shared.discard(entry['entry_key'], old)
    db.create_shared_task(uuid.uuid4().hex, 'model', new, entry['group_key'], {}, {}, 30)
    assert db.queue_usage(entry['group_key'])['inflight'] == 1
    # Lost commit acknowledgement cannot undo a subsequent Stop.
    db.set_queue_accepting(new, False)
    assert not db.shared.activate(entry['entry_key'], token)['accepting']


def test_initial_registration_is_serialized_and_restart_follows_new_active_config(db):
    entry, old, new = initial(db)
    token = prepare(db, entry, new)
    ready(db, entry, new, token)
    db.shared.activate(entry['entry_key'], token)
    db.shared.discard(entry['entry_key'], old)
    restarted = DB(db.url).init_schema()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda _: restarted.shared.register(entry['group_key'], 'model', old, 'old', {'version': 'stale'}), range(8)))
        assert all(row['active_queue'] == new and row['data'] == {'version': 'new'} for row in rows)
        unknown = uuid.uuid4().hex
        db.register_queue(unknown, 8)
        with pytest.raises(ServerError):
            restarted.shared.register(entry['group_key'], 'model', unknown, 'unknown', {})
    finally:
        restarted.close()


def test_concurrent_first_registration_keeps_one_canonical_configuration(db):
    group, queue = uuid.uuid4().hex, uuid.uuid4().hex
    db.register_queue(group, 8)
    db.register_queue(queue, 8)
    barrier = threading.Barrier(4)
    def register(index):
        barrier.wait(timeout=5)
        return db.shared.register(group, 'model', queue, 'config', {'local_http_policy': index})
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(register, range(4)))
    assert len({row['control_token'] for row in rows}) == 1
    assert len({row['data']['local_http_policy'] for row in rows}) == 1


def test_atomic_switch_serializes_concurrent_admission_start_and_worker_exit(db):
    entry, old, new = initial(db)
    token = prepare(db, entry, new)
    ready(db, entry, new, token)
    barrier = threading.Barrier(3)
    def operation(kind):
        barrier.wait(timeout=5)
        try:
            if kind == 'switch':
                return kind, db.shared.activate(entry['entry_key'], token)
            if kind == 'start':
                return kind, db.set_queue_accepting(old, True)
            identity = uuid.uuid4().hex
            db.create_shared_task(identity, 'model', old, entry['group_key'], {}, {}, 30)
            return kind, identity
        except ServerError as failure:
            return kind, failure
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(operation, ('switch', 'start', 'submit')))
    current = db.shared.get(entry['entry_key'])
    if isinstance(results['switch'], ServerError):
        assert results['switch'].code == 'shared_control_changed'
        assert current['active_queue'] == old
        assert not isinstance(results['start'], ServerError)
    else:
        assert current['active_queue'] == new
        assert isinstance(results['start'], ServerError) and results['start'].code == 'entry_changed'
        assert not db.queue_usage(old)['accepting']
    accepted = not isinstance(results['submit'], ServerError)
    assert db.queue_usage(old)['inflight'] == int(accepted)
    assert db.queue_usage(entry['group_key'])['inflight'] == int(accepted)
    if accepted:
        assert db.get_task(results['submit'])['queue_key'] == old
    else:
        assert results['submit'].code == 'entry_changed'


def test_candidate_exit_before_activation_keeps_old_configuration(db):
    entry, old, new = initial(db)
    token = prepare(db, entry, new)
    holder = ready(db, entry, new, token)
    db.release_files(holder)
    with pytest.raises(ServerError) as failure:
        db.shared.activate(entry['entry_key'], token)
    assert failure.value.code == 'entry_not_ready'
    assert db.shared.get(entry['entry_key'])['active_queue'] == old


def test_only_one_concurrent_reload_owns_preparation(db):
    entry, old, first = initial(db)
    second = uuid.uuid4().hex
    db.register_queue(second, 8)
    barrier = threading.Barrier(2)
    def start(key):
        barrier.wait(timeout=5)
        try:
            return key, prepare(db, entry, key)
        except ServerError as failure:
            return failure
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, (first, second)))
    successes = [value for value in results if isinstance(value, tuple)]
    assert len(successes) == 1
    assert sum(isinstance(value, ServerError) for value in results) == 1
    key, token = successes[0]
    current = db.shared.get(entry['entry_key'])
    assert current['pending_queue'] == key and current['active_queue'] == old
    assert db.shared.begin_reload(entry['entry_key'], entry['control_token'], key, 'new', {'version': 'new'}, write_token=token) == token
    with pytest.raises(ServerError):
        db.shared.begin_reload(entry['entry_key'], entry['control_token'], key, 'new', {'version': 'different'}, write_token=token)


@pytest.mark.parametrize('resume', [False, True])
def test_start_stop_during_preparation_invalidates_activation(db, resume):
    entry, old, new = initial(db)
    token = prepare(db, entry, new)
    holder = ready(db, entry, new, token)
    db.set_queue_accepting(old, False)
    if resume:
        db.set_queue_accepting(old, True)
    with pytest.raises(ServerError) as changed:
        db.shared.activate(entry['entry_key'], token)
    assert changed.value.code == 'shared_control_changed'
    assert db.shared.get(entry['entry_key'])['active_queue'] == old
    db.release_files(holder)
    db.shared.discard(entry['entry_key'], new, token=token)
    assert db.queue_usage(old)['accepting'] == resume


def test_config_files_survive_restart_and_only_release_after_physical_workers_close(db):
    old_file, new_file = file_record(db), file_record(db)
    entry, old, new = initial(db, {'default': {'file_id': old_file}})
    old_holder = ready(db, entry, old, None)
    token = prepare(db, entry, new, {'default': {'file_id': new_file}})
    new_holder = ready(db, entry, new, token)
    assert_protected(db, old_file)
    assert_protected(db, new_file)
    db.shared.activate(entry['entry_key'], token)
    restarted = DB(db.url).init_schema()
    try:
        with pytest.raises(ServerError) as held:
            restarted.shared.discard(entry['entry_key'], old)
        assert held.value.code == 'entry_busy'
        assert_protected(restarted, old_file)
        restarted.release_files(old_holder)
        restarted.shared.discard(entry['entry_key'], old)
        restarted.shared.discard(entry['entry_key'], old)
        restarted.mark_file_deleting(old_file)
        restarted.release_files(new_holder)
        assert_protected(restarted, new_file)  # Active config survives a stopped runtime.
    finally:
        restarted.close()


def test_abort_requires_closed_workers_and_late_activation_cannot_resurrect_candidate(db):
    entry, old, new = initial(db)
    file = file_record(db)
    token = prepare(db, entry, new, {'file_id': file})
    holder = ready(db, entry, new, token)
    with pytest.raises(ServerError) as held:
        db.shared.discard(entry['entry_key'], new, token=token)
    assert held.value.code == 'entry_busy'
    db.release_files(holder)
    db.shared.discard(entry['entry_key'], new, token=token)
    db.shared.discard(entry['entry_key'], new, token=token)
    db.mark_file_deleting(file)
    with pytest.raises(ServerError):
        db.shared.activate(entry['entry_key'], token)
    with pytest.raises(ServerError):
        db.shared.hold_worker(entry['entry_key'], new, holder, token=token)
    assert db.shared.get(entry['entry_key'])['active_queue'] == old
    assert db.queue_usage(old)['accepting']


def test_failed_preparation_rolls_back_config_and_file_references(db):
    entry, old, new = initial(db)
    good, missing = file_record(db), 'file_' + uuid.uuid4().hex
    with pytest.raises(ServerError) as unavailable:
        prepare(db, entry, new, {'files': [{'file_id': good}, {'file_id': missing}]})
    assert unavailable.value.code == 'invalid_file'
    assert db.shared.get(entry['entry_key'])['pending_queue'] is None
    db.mark_file_deleting(good)
    prepare(db, entry, new)  # The failed transaction did not claim the queue.


def test_all_candidate_workers_must_be_ready_and_old_workers_block_next_reload(db):
    entry, old, new = initial(db)
    old_holder = ready(db, entry, old, None)
    token = prepare(db, entry, new, count=2)
    ready(db, entry, new, token)
    with pytest.raises(ServerError) as not_ready:
        db.shared.activate(entry['entry_key'], token)
    assert not_ready.value.code == 'entry_not_ready'
    ready(db, entry, new, token)
    current = db.shared.activate(entry['entry_key'], token)
    next_queue = uuid.uuid4().hex
    db.register_queue(next_queue, 8)
    with pytest.raises(ServerError) as draining:
        prepare(db, current, next_queue)
    assert draining.value.code == 'entry_busy'
    db.release_files(old_holder)
    db.shared.discard(entry['entry_key'], old)
    prepare(db, current, next_queue)


def test_shared_schema_upgrade_preserves_preexisting_queues_and_tasks(db):
    from nndeploy.server.shared import entries, configs, workers, runtime_records
    group, queue, task = [uuid.uuid4().hex for _ in range(3)]
    db.register_queue(group, 4)
    db.register_queue(queue, 2)
    db.create_shared_task(task, 'legacy', queue, group, {'text': 'kept'}, {}, 30)
    # Isolated fixture schema may contain preceding tests' rows on remote DBs;
    # only new coordination tables are dropped, never existing tasks/configs.
    with db.engine.begin() as connection:
        for table in (runtime_records, workers, configs, entries):
            table.drop(connection)
        connection.execute(text("UPDATE alembic_version SET version_num = '0015_queue_control_tokens'"))
    reopened = DB(db.url).init_schema()
    try:
        assert all(table.name in inspect(reopened.engine).get_table_names() for table in (entries, configs, workers))
        assert reopened.get_task(task)['inputs'] == {'text': 'kept'}
        assert reopened.queue_usage(group)['inflight'] == 1
        assert reopened.queue_usage(queue)['inflight'] == 1
    finally:
        reopened.close()


def test_real_redis_old_queue_drains_while_new_queue_accepts_and_replays_old_idempotency(backend):
    database, config = backend
    async def scenario():
        old = RedisTaskQueue(database, config, 'model', 'old', 2)
        new = RedisTaskQueue(database, config, 'model', 'new', 2)
        try:
            await old.start()
            await new.start()
            entry = database.shared.register(old.group_key, 'model', old.key, 'old', {'version': 'old'})
            before = await old.put({'value': 'old'}, idempotency_key='request')
            token = prepare(database, entry, new.key)
            ready(database, entry, new.key, token)
            database.shared.activate(entry['entry_key'], token)
            with pytest.raises(ServerError) as changed:
                await old.put({'value': 'obsolete'})
            assert changed.value.code == 'entry_changed'
            assert (await new.put({'value': 'old'}, idempotency_key='request'))['task_id'] == before['task_id']
            after = await new.put({'value': 'new'})
            assert database.queue_usage(old.group_key)['inflight'] == 2
            received_old, received_new = await old.get('old-worker'), await new.get('new-worker')
            assert received_old['task_id'] == before['task_id']
            assert received_new['task_id'] == after['task_id']
            await old.finish(received_old, outputs={'version': 'old'})
            database.shared.discard(entry['entry_key'], old.key)
            await new.finish(received_new, outputs={'version': 'new'})
            assert database.queue_usage(old.group_key)['inflight'] == 0
        finally:
            await old.close()
            await new.close()
    asyncio.run(scenario())

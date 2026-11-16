"""Internal durable configuration switching for shared Server queues.

The namespace capacity row serializes registration, admission and switching on
all three databases (including SQLite). Model initialization and Redis calls
never run inside these transactions. Runtime holders have no automatic expiry:
only confirmed Worker shutdown may release them through DB.release_files().
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid

from sqlalchemy import Table, Column, String, Integer, Boolean, ForeignKey, UniqueConstraint, select, insert, update, delete

from .db import metadata, queue_limits, server_configs, config_file_refs, file_holders, tasks, encode, _file_ids
from .errors import ServerError

entries = Table('server_shared_entries', metadata,
    Column('entry_key', String(64), primary_key=True),
    Column('group_key', String(64), ForeignKey('server_queue_limits.queue_key'), nullable=False),
    Column('name', String(128), nullable=False),
    Column('active_queue', String(64), ForeignKey('server_queue_limits.queue_key'), nullable=False),
    Column('control_token', String(32), nullable=False),
    Column('pending_queue', String(64), ForeignKey('server_queue_limits.queue_key')),
    Column('pending_token', String(32)), Column('pending_control_token', String(32)),
    Column('pending_holder', String(32)),
    UniqueConstraint('group_key', 'name', name='uq_server_shared_entry'))
configs = Table('server_shared_configs', metadata,
    Column('queue_key', String(64), ForeignKey('server_queue_limits.queue_key'), primary_key=True),
    Column('entry_key', String(64), ForeignKey('server_shared_entries.entry_key'), nullable=False, index=True),
    Column('fingerprint', String(64), nullable=False),
    Column('config_id', String(64), ForeignKey('server_configs.config_id')),
    Column('role', String(16), nullable=False), Column('workers', Integer, nullable=False),
    Column('initialized', Boolean, nullable=False, server_default='1'),
    Column('load_failed', Boolean, nullable=False, server_default='0'),
    Column('write_token', String(32), nullable=False))
workers = Table('server_shared_workers', metadata,
    Column('holder', String(64), ForeignKey('server_file_holders.holder', ondelete='CASCADE'), primary_key=True),
    Column('queue_key', String(64), ForeignKey('server_shared_configs.queue_key'), nullable=False, index=True),
    Column('ready', Boolean, nullable=False, server_default='0'),
    Column('owner_token', String(32)), Column('runtime_holder', String(32)))
runtime_records = Table('server_shared_runtimes', metadata,
    Column('token', String(32), primary_key=True), Column('holder', String(32), nullable=False),
    # Registration precedes schema discovery, so the entry and queue may not
    # exist yet. The opaque entry key still locates an interrupted initializer.
    Column('entry_key', String(64), nullable=False, index=True),
    Column('queue_key', String(64), ForeignKey('server_shared_configs.queue_key')),
    Column('fenced', Boolean, nullable=False, server_default='0'))


def _conflict(message):
    return ServerError(409, 'shared_control_changed', message, retryable=True)


class SharedControl:
    def __init__(self, db):
        self.db = db

    @staticmethod
    def entry_key(group_key, name):
        return hashlib.sha256(encode([group_key, name]).encode()).hexdigest()

    @staticmethod
    def _lock_group(connection, group_key):
        locked = connection.execute(update(queue_limits).where(queue_limits.c.queue_key == group_key)
                                    .values(capacity=queue_limits.c.capacity))
        if locked.rowcount != 1:
            raise ServerError(404, 'queue_not_found', 'Shared namespace queue does not exist')

    def _entry(self, connection, entry_key):
        group = connection.execute(select(entries.c.group_key).where(entries.c.entry_key == entry_key)).scalar()
        if group is None:
            raise ServerError(404, 'entry_not_found', 'Shared entry does not exist')
        self._lock_group(connection, group)
        return connection.execute(select(entries).where(entries.c.entry_key == entry_key).with_for_update()).mappings().one()

    @staticmethod
    def _queue(connection, queue_key):
        row = connection.execute(select(queue_limits).where(queue_limits.c.queue_key == queue_key).with_for_update()).mappings().first()
        if row is None:
            raise ServerError(404, 'queue_not_found', 'Shared queue does not exist')
        return row

    def _snapshot(self, connection, entry_key, queue_key, fingerprint, data, count, token, role):
        identity = hashlib.sha256(encode([entry_key, queue_key]).encode()).hexdigest()
        connection.execute(insert(server_configs).values(config_id=identity, scope=entry_key, name=queue_key,
                                                        data=encode(data), write_token=token))
        for file_id in sorted(_file_ids(data)):
            self.db._lock_available_file(connection, file_id)
            connection.execute(insert(config_file_refs).values(config_id=identity, file_id=file_id))
        connection.execute(insert(configs).values(queue_key=queue_key, entry_key=entry_key, fingerprint=fingerprint,
                           config_id=identity, workers=count, write_token=token, role=role, initialized=False, load_failed=False))

    @staticmethod
    def _validate(fingerprint, count, token=None):
        if not isinstance(fingerprint, str) or not 1 <= len(fingerprint) <= 64 or type(count) is not int or count < 1:
            raise ValueError('Shared configuration requires a fingerprint and positive worker count')
        if token is not None and (not isinstance(token, str) or len(token) != 32):
            raise ValueError('Shared configuration operation requires a 32-character token')

    @staticmethod
    def _get(connection, entry_key):
        row = connection.execute(select(entries, configs.c.fingerprint, configs.c.initialized, configs.c.load_failed, server_configs.c.data,
                                        queue_limits.c.accepting, queue_limits.c.control_token.label('queue_control_token')).select_from(entries.join(
            configs, configs.c.queue_key == entries.c.active_queue).join(
            server_configs, server_configs.c.config_id == configs.c.config_id).join(
            queue_limits, queue_limits.c.queue_key == entries.c.active_queue)).where(entries.c.entry_key == entry_key).with_for_update()).mappings().first()
        if row is None:
            raise ServerError(404, 'entry_not_found', 'Shared entry does not exist')
        result = dict(row)
        result['data'] = json.loads(result['data'])
        return result

    def get(self, entry_key):
        with self.db.engine.begin() as connection:
            self._entry(connection, entry_key)
            return self._get(connection, entry_key)

    def status(self, entry_key):
        """Read one coherent routing snapshot without loading workflow JSON."""
        with self.db.engine.connect() as connection:
            row = connection.execute(select(entries, queue_limits.c.accepting, queue_limits.c.control_token.label('queue_control_token')).join(
                queue_limits, entries.c.active_queue == queue_limits.c.queue_key).where(entries.c.entry_key == entry_key)).mappings().first()
            if row is None:
                raise ServerError(404, 'entry_not_found', 'Shared entry does not exist')
            return dict(row)

    def register(self, group_key, name, queue_key, fingerprint, data, *, count=1):
        """Join a known configuration; old known callers discover the active one."""
        self._validate(fingerprint, count)
        key = self.entry_key(group_key, name)
        if group_key == queue_key:
            raise ValueError('Entry queue must differ from namespace queue')
        with self.db.engine.begin() as connection:
            self._lock_group(connection, group_key)
            entry = connection.execute(select(entries).where(entries.c.entry_key == key).with_for_update()).mappings().first()
            member = connection.execute(select(configs).where(configs.c.queue_key == queue_key).with_for_update()).mappings().first()
            if entry:
                active = connection.execute(select(configs).where(configs.c.queue_key == entry['active_queue']).with_for_update()).mappings().one()
                known = connection.execute(select(configs.c.queue_key).where(configs.c.entry_key == key,
                    configs.c.fingerprint == fingerprint, configs.c.role.in_(('active', 'draining', 'retired'))).with_for_update()).first()
                repair = active['load_failed'] and not active['initialized']
                if (not known and not repair) or member and member['entry_key'] != key:
                    raise _conflict('Use the active shared configuration or reload the existing entry')
            else:
                self._queue(connection, queue_key)
                if member:
                    raise _conflict('Shared queue belongs to another entry')
                token = uuid.uuid4().hex
                connection.execute(insert(entries).values(entry_key=key, group_key=group_key, name=name,
                                                          active_queue=queue_key, control_token=token))
                self._snapshot(connection, key, queue_key, fingerprint, data, count, token, 'active')
            return self._get(connection, key)

    def begin_reload(self, entry_key, expected_token, queue_key, fingerprint, data, *, count=1, write_token,
                     repair=False, expected_queue_token=None, pending_holder=None):
        """Save a candidate before loading models; active admission stays open."""
        self._validate(fingerprint, count, write_token)
        with self.db.engine.begin() as connection:
            entry = self._entry(connection, entry_key)
            if entry['pending_token'] == write_token and entry['pending_queue'] == queue_key:
                member = connection.execute(select(configs, server_configs.c.data).join(
                    server_configs, configs.c.config_id == server_configs.c.config_id).where(configs.c.queue_key == queue_key).with_for_update()).mappings().one()
                if (member['fingerprint'] == fingerprint and member['workers'] == count and member['data'] == encode(data)
                        and entry['pending_holder'] == pending_holder
                        and entry['control_token'] == expected_token and (expected_queue_token is None or expected_queue_token == entry['pending_control_token'])):
                    return write_token
                raise _conflict('Reload operation was reused with different content')
            if entry['control_token'] != expected_token or entry['pending_queue'] is not None or write_token == expected_token:
                raise _conflict('Shared configuration changed or another reload is pending')
            if connection.execute(select(configs.c.queue_key).where(configs.c.entry_key == entry_key, configs.c.role == 'draining').with_for_update()).first():
                raise ServerError(409, 'entry_busy', 'Previous shared workers are still draining', retryable=True)
            if queue_key in (entry['group_key'], entry['active_queue']):
                raise _conflict('Reload requires a new configuration queue')
            active = self._queue(connection, entry['active_queue'])
            if expected_queue_token is not None and active['control_token'] != expected_queue_token:
                raise _conflict('Start or Stop changed before configuration preparation')
            if repair:
                member = connection.execute(select(configs).where(configs.c.queue_key == entry['active_queue']).with_for_update()).mappings().one()
                held = connection.execute(select(workers.c.holder).where(workers.c.queue_key == entry['active_queue']).with_for_update()).first()
                if member['initialized'] or not member['load_failed'] or held or active['inflight']:
                    raise _conflict('Initial configuration is no longer available for repair')
            candidate = self._queue(connection, queue_key)
            if (not active['accepting'] and active['inflight']) or candidate['inflight'] or connection.execute(select(configs.c.queue_key).where(configs.c.queue_key == queue_key).with_for_update()).first():
                raise _conflict('Reload requires a ready or fully stopped entry and an unused candidate queue')
            self._snapshot(connection, entry_key, queue_key, fingerprint, data, count, write_token, 'candidate')
            connection.execute(update(queue_limits).where(queue_limits.c.queue_key == queue_key).values(accepting=False, control_token=write_token))
            connection.execute(update(entries).where(entries.c.entry_key == entry_key).values(
                pending_queue=queue_key, pending_token=write_token, pending_control_token=active['control_token'], pending_holder=pending_holder))
            return write_token

    def hold_worker(self, entry_key, queue_key, holder, *, token=None, ready=False, draining=False, owner_token=None, runtime_holder=None):
        """Record a runtime before initialization, then mark it ready after warmup."""
        with self.db.engine.begin() as connection:
            entry = self._entry(connection, entry_key)
            if runtime_holder is not None:
                runtime = connection.execute(select(runtime_records).where(runtime_records.c.token == owner_token).with_for_update()).mappings().first()
                if (not runtime or runtime['fenced'] or runtime['holder'] != runtime_holder
                        or runtime['entry_key'] != entry_key or runtime['queue_key'] != queue_key):
                    raise _conflict('Worker runtime is no longer available')
            old = None
            if draining:
                old = connection.execute(select(configs.c.queue_key).where(configs.c.entry_key == entry_key,
                    configs.c.queue_key == queue_key, configs.c.role == 'draining').with_for_update()).first()
                if not old:
                    raise _conflict('Old configuration has already been retired')
            if not old and queue_key != entry['active_queue'] and not (queue_key == entry['pending_queue'] and token == entry['pending_token']):
                raise _conflict('Worker configuration is no longer active or being prepared')
            if queue_key == entry['pending_queue'] and entry['pending_holder'] and owner_token != entry['pending_token']:
                raise _conflict('Candidate worker requires its controller exit guard')
            if queue_key == entry['active_queue'] and entry['pending_queue'] is not None:
                established = connection.execute(select(configs.c.initialized).where(configs.c.queue_key == queue_key).with_for_update()).scalar_one()
                if not established:
                    raise _conflict('Initial configuration is being repaired; wait for its replacement')
            held = connection.execute(select(workers).where(workers.c.holder == holder).with_for_update()).mappings().first()
            if held and (held['queue_key'] != queue_key or held['owner_token'] != owner_token or held['runtime_holder'] != runtime_holder):
                raise _conflict('Worker holder belongs to another configuration')
            if not connection.execute(select(file_holders.c.holder).where(file_holders.c.holder == holder).with_for_update()).first():
                connection.execute(insert(file_holders).values(holder=holder, host=socket.gethostname(), pid=os.getpid(),
                                                               created_at=self.db.clock(connection)))
            if held:
                connection.execute(update(workers).where(workers.c.holder == holder).values(ready=ready))
            else:
                connection.execute(insert(workers).values(holder=holder, queue_key=queue_key, ready=ready,
                                                         owner_token=owner_token, runtime_holder=runtime_holder))
            # A fresh retry owns initialization now; an old failure report must
            # not let another configuration replace it while it loads.
            connection.execute(update(configs).where(configs.c.queue_key == queue_key).values(load_failed=False))

    def confirm_ready(self, entry_key, queue_key, holders):
        """Only a complete local runtime, after all startup checks, establishes an entry."""
        with self.db.engine.begin() as connection:
            entry = self._entry(connection, entry_key)
            if entry['active_queue'] != queue_key:
                raise _conflict('Active configuration changed during initialization')
            member = connection.execute(select(configs).where(configs.c.queue_key == queue_key).with_for_update()).mappings().one()
            ready = connection.execute(select(workers.c.holder).where(workers.c.queue_key == queue_key,
                workers.c.holder.in_(holders), workers.c.ready.is_(True)).with_for_update()).all()
            if len(ready) != member['workers'] or len(set(holders)) != member['workers']:
                raise ServerError(409, 'entry_not_ready', 'All local workers must finish initialization before confirming readiness')
            connection.execute(update(configs).where(configs.c.queue_key == queue_key).values(initialized=True, load_failed=False))

    def initialization_failed(self, entry_key, queue_key):
        """Record confirmed failure without inferring physical exit from a timeout."""
        with self.db.engine.begin() as connection:
            entry = self._entry(connection, entry_key)
            if entry['active_queue'] != queue_key:
                return
            member = connection.execute(select(configs).where(configs.c.queue_key == queue_key).with_for_update()).mappings().one()
            queue = self._queue(connection, queue_key)
            if member['initialized'] or queue['inflight'] or connection.execute(select(workers.c.holder).where(
                    workers.c.queue_key == queue_key).with_for_update()).first():
                return
            connection.execute(update(configs).where(configs.c.queue_key == queue_key).values(load_failed=True))

    def activate(self, entry_key, token):
        with self.db.engine.begin() as connection:
            entry = self._entry(connection, entry_key)
            if entry['control_token'] == token:
                return self._get(connection, entry_key)
            if entry['pending_token'] != token:
                raise _conflict('Reload is no longer pending')
            active = self._queue(connection, entry['active_queue'])
            if active['control_token'] != entry['pending_control_token']:
                raise _conflict('Start or Stop changed while the new configuration was loading')
            member = connection.execute(select(configs).where(configs.c.queue_key == entry['pending_queue']).with_for_update()).mappings().one()
            ready = connection.execute(select(workers.c.holder).where(workers.c.queue_key == entry['pending_queue'], workers.c.ready.is_(True)).with_for_update()).all()
            if len(ready) < member['workers']:
                raise ServerError(409, 'entry_not_ready', 'Candidate workers have not completed initialization and warmup')
            for key, accepting, role in ((entry['active_queue'], False, 'draining'), (entry['pending_queue'], True, 'active')):
                connection.execute(update(queue_limits).where(queue_limits.c.queue_key == key).values(accepting=accepting, control_token=token))
                connection.execute(update(configs).where(configs.c.queue_key == key).values(role=role))
            connection.execute(update(configs).where(configs.c.queue_key == entry['pending_queue']).values(initialized=True, load_failed=False))
            connection.execute(update(entries).where(entries.c.entry_key == entry_key).values(
                active_queue=entry['pending_queue'], control_token=token, pending_queue=None, pending_token=None, pending_control_token=None, pending_holder=None))
            return self._get(connection, entry_key)

    def discard(self, entry_key, queue_key, *, token=None):
        """Abort a candidate or retire old config only after all physical users exit."""
        with self.db.engine.begin() as connection:
            entry = self._entry(connection, entry_key)
            member = connection.execute(select(configs).where(configs.c.queue_key == queue_key).with_for_update()).mappings().first()
            if not member or member['entry_key'] != entry_key or queue_key == entry['active_queue']:
                raise _conflict('Cannot discard an active or unrelated configuration')
            candidate = member['role'] in ('candidate', 'aborted')
            if candidate and token != member['write_token']:
                raise _conflict('Only the pending reload may discard its candidate')
            if member['role'] in ('retired', 'aborted'):
                return
            if self._queue(connection, queue_key)['inflight'] or connection.execute(select(workers.c.holder).where(workers.c.queue_key == queue_key).with_for_update()).first():
                raise ServerError(409, 'entry_busy', 'Tasks or workers still use the shared configuration', retryable=True)
            if candidate:
                connection.execute(update(entries).where(entries.c.entry_key == entry_key).values(
                    pending_queue=None, pending_token=None, pending_control_token=None, pending_holder=None))
            connection.execute(update(configs).where(configs.c.queue_key == queue_key).values(config_id=None, role='aborted' if candidate else 'retired'))
            connection.execute(delete(server_configs).where(server_configs.c.config_id == member['config_id']))

    def recover_pending(self, entry_key, directory):
        """Abort an uncommitted change only with exclusive physical exit proof.

        Leaves the active configuration, admission intent and tasks untouched.
        Unknown legacy or remote owners stay held; a timeout is never evidence.
        """
        from .control_guard import ControlGuard
        state = self.status(entry_key)
        token, holder = state['pending_token'], state['pending_holder']
        if not token or not holder:
            return False
        with ControlGuard.recover(directory, token, holder) as exited:
            if not exited:
                return False
            with self.db.engine.begin() as connection:
                entry = self._entry(connection, entry_key)
                if entry['pending_token'] != token or entry['pending_holder'] != holder:
                    return False
                key = entry['pending_queue']
                member = connection.execute(select(configs).where(configs.c.queue_key == key).with_for_update()).mappings().one()
                held = connection.execute(select(workers).where(workers.c.queue_key == key).with_for_update()).mappings().all()
                if member['role'] != 'candidate' or self._queue(connection, key)['inflight'] or any(row['owner_token'] != token for row in held):
                    return False
                connection.execute(delete(file_holders).where(file_holders.c.holder.in_([holder] + [row['holder'] for row in held])))
                connection.execute(update(entries).where(entries.c.entry_key == entry_key).values(
                    pending_queue=None, pending_token=None, pending_control_token=None, pending_holder=None))
                connection.execute(update(configs).where(configs.c.queue_key == key).values(config_id=None, role='aborted'))
                connection.execute(delete(server_configs).where(server_configs.c.config_id == member['config_id']))
                connection.execute(delete(runtime_records).where(runtime_records.c.token == token,
                    runtime_records.c.holder == holder, runtime_records.c.entry_key == entry_key))
                return True

    def bind_runtime(self, entry_key, token, holder, queue_key=None):
        """Persist ownership before model loading, including factory discovery."""
        with self.db.engine.begin() as connection:
            if queue_key is not None:
                self._entry(connection, entry_key)
                if not connection.execute(select(configs.c.queue_key).where(configs.c.entry_key == entry_key,
                        configs.c.queue_key == queue_key).with_for_update()).first():
                    raise _conflict('Runtime queue no longer belongs to this entry')
            row = connection.execute(select(runtime_records).where(runtime_records.c.token == token).with_for_update()).mappings().first()
            if row:
                if (row['fenced'] or row['holder'] != holder or row['entry_key'] != entry_key
                        or queue_key is not None and row['queue_key'] not in (None, queue_key)):
                    raise _conflict('Runtime ownership changed')
                if queue_key is not None:
                    connection.execute(update(runtime_records).where(runtime_records.c.token == token).values(queue_key=queue_key))
            else:
                connection.execute(insert(runtime_records).values(token=token, holder=holder, entry_key=entry_key, queue_key=queue_key))

    def retire_runtime(self, token):
        """Normal physical shutdown may still leave a result awaiting commit."""
        with self.db.engine.begin() as connection:
            row = connection.execute(select(runtime_records).where(runtime_records.c.token == token)).mappings().first()
            if row is None:
                return
            if row['queue_key'] is not None:
                self._entry(connection, row['entry_key'])
            connection.execute(update(runtime_records).where(runtime_records.c.token == token).values(fenced=True))
            if (not connection.execute(select(tasks.c.task_id).where(tasks.c.runtime_token == token, tasks.c.state == 'running').with_for_update()).first()
                    and not connection.execute(select(workers.c.holder).where(workers.c.owner_token == token).with_for_update()).first()):
                connection.execute(delete(runtime_records).where(runtime_records.c.token == token))

    def claim_runtime(self, connection, queue_key, holder):
        """Fence late claims under the same namespace lock used by recovery."""
        self.lock_control(connection, queue_key)
        row = connection.execute(select(workers).where(workers.c.holder == holder,
            workers.c.queue_key == queue_key).with_for_update()).mappings().first()
        if not row or not row['ready'] or not row['owner_token'] or not row['runtime_holder']:
            raise ServerError(409, 'worker_unavailable', 'Worker no longer owns a ready runtime', retryable=True)
        runtime = connection.execute(select(runtime_records).where(runtime_records.c.token == row['owner_token']).with_for_update()).mappings().first()
        if not runtime or runtime['fenced'] or runtime['holder'] != row['runtime_holder'] or runtime['queue_key'] != queue_key:
            raise ServerError(409, 'worker_unavailable', 'Worker runtime has been fenced', retryable=True)
        return dict(runtime_token=row['owner_token'], runtime_holder=row['runtime_holder'])

    def runtimes(self, entry_key, after='', limit=100):
        """Include result-pending tasks even when normal close removed workers."""
        with self.db.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(runtime_records).where(
                runtime_records.c.entry_key == entry_key, runtime_records.c.token > after)
                .order_by(runtime_records.c.token).limit(limit)).mappings()]

    def recover_runtime(self, entry_key, runtime, directory, limit=100):
        """Reconcile results or fail lost work only after all physical users exit."""
        from pathlib import Path
        from .config import load_json
        from .control_guard import ControlGuard
        from .db import TERMINAL
        token, holder, key = runtime['token'], runtime['holder'], runtime['queue_key']
        with ControlGuard.recover(directory, token, holder) as exited:
            if not exited:
                return False
            with self.db.engine.begin() as connection:
                self._entry(connection, entry_key)
                current = connection.execute(select(runtime_records).where(runtime_records.c.token == token).with_for_update()).mappings().first()
                if not current or any(current[field] != runtime[field] for field in ('entry_key', 'holder', 'queue_key')):
                    return False
                if key is None:
                    connection.execute(delete(file_holders).where(file_holders.c.holder == holder))
                    connection.execute(delete(runtime_records).where(runtime_records.c.token == token))
                    return True
                member = connection.execute(select(configs).where(configs.c.entry_key == entry_key,
                    configs.c.queue_key == key).with_for_update()).mappings().first()
                if not member or member['role'] == 'candidate':
                    return False
                owned = list(connection.execute(select(workers).where(workers.c.owner_token == token).with_for_update()).mappings())
                if any(row['runtime_holder'] != holder or row['queue_key'] != key for row in owned):
                    return False
                connection.execute(update(runtime_records).where(runtime_records.c.token == token).values(fenced=True))
                # Keep membership until result reconciliation is complete, but
                # prevent any delayed claim from taking another task meanwhile.
                connection.execute(update(workers).where(workers.c.owner_token == token).values(ready=False))
                records = list(connection.execute(select(tasks).where(tasks.c.queue_key == key,
                    tasks.c.runtime_token == token, tasks.c.runtime_holder == holder, tasks.c.state == 'running')
                    .order_by(tasks.c.task_id).limit(limit).with_for_update()).mappings())
            for record in records:
                path = Path(directory) / 'queue-results' / key / f'{record["task_id"]}-{record["worker_token"]}.json'
                result = None
                try:
                    result = load_json(path)
                except FileNotFoundError:
                    pass
                if result is not None:
                    saved = result['record']
                    if any(saved.get(field) != record[field] for field in (
                            'task_id', 'queue_key', 'worker_token', 'attempt', 'runtime_token', 'runtime_holder')):
                        raise RuntimeError('Saved result does not belong to the exited runtime')
                    state, outputs, error = result['state'], result['outputs'], result['error']
                else:
                    cancelled = record['cancel_requested']
                    state, outputs = ('cancelled' if cancelled else 'failed'), None
                    error = dict(code='task_cancelled' if cancelled else 'worker_lost',
                                 message='Worker exited before its result was recorded', retryable=False,
                                 task_id=record['task_id'])
                try:
                    self.db.finish_task(record['task_id'], state, self.db.clock(), outputs, error,
                                        record['attempt'], record['worker_token'])
                except RuntimeError:
                    current = self.db.get_task(record['task_id'])
                    if not current or current['state'] not in TERMINAL or any(current[field] != record[field]
                            for field in ('attempt', 'worker_token', 'runtime_token', 'runtime_holder')):
                        raise
            with self.db.engine.begin() as connection:
                self._entry(connection, entry_key)
                remaining = connection.execute(select(tasks.c.task_id).where(tasks.c.runtime_token == token,
                    tasks.c.state == 'running').with_for_update()).first()
                if remaining:
                    return False
                owned = list(connection.execute(select(workers.c.holder).where(workers.c.owner_token == token,
                    workers.c.runtime_holder == holder, workers.c.queue_key == key).with_for_update()).scalars())
                connection.execute(delete(file_holders).where(file_holders.c.holder.in_(owned + [holder])))
                connection.execute(delete(runtime_records).where(runtime_records.c.token == token))
            self.initialization_failed(entry_key, key)
            return True

    def check_admission(self, connection, queue_key, group_key=None):
        """Called under the namespace lock in the task/control transaction."""
        row = connection.execute(select(configs.c.role, entries.c.active_queue, entries.c.group_key).join(
            entries, configs.c.entry_key == entries.c.entry_key).where(configs.c.queue_key == queue_key).with_for_update()).first()
        if row and (row.role != 'active' or row.active_queue != queue_key or group_key is not None and row.group_key != group_key):
            raise ServerError(409, 'entry_changed', 'Shared configuration changed; refresh the entry before submitting', retryable=True)

    def lock_control(self, connection, queue_key):
        key = connection.execute(select(configs.c.entry_key).where(configs.c.queue_key == queue_key)).scalar()
        if key is not None:
            self._entry(connection, key)

    def lock_holder(self, connection, holder):
        key = connection.execute(select(configs.c.entry_key).join(
            workers, workers.c.queue_key == configs.c.queue_key).where(workers.c.holder == holder)).scalar()
        if key is not None:
            self._entry(connection, key)

    def draining(self, entry_key):
        with self.db.engine.connect() as connection:
            return list(connection.execute(select(configs.c.queue_key).where(
                configs.c.entry_key == entry_key, configs.c.role == 'draining')).scalars())

    def recoverable(self, entry_key, result_queues=()):
        """Only queued work can be resumed without proving a lost worker's exit."""
        pending = select(tasks.c.task_id).where(tasks.c.queue_key == configs.c.queue_key, tasks.c.state == 'pending').exists()
        with self.db.engine.connect() as connection:
            rows = connection.execute(select(configs.c.queue_key, configs.c.fingerprint, server_configs.c.data).join(
                server_configs, configs.c.config_id == server_configs.c.config_id).where(
                    configs.c.entry_key == entry_key, configs.c.role == 'draining',
                    pending | configs.c.queue_key.in_(result_queues))).mappings().all()
            return [dict(entry_key=entry_key, active_queue=row['queue_key'], fingerprint=row['fingerprint'],
                         data=json.loads(row['data'])) for row in rows]

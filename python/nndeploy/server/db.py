"""Transactional Server records, with SQLite, MySQL and PostgreSQL backends."""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
import uuid
from pathlib import Path

from sqlalchemy import (MetaData, Table, Column, String, Text, Integer, BigInteger, Double,
                        DateTime, Boolean, Index, UniqueConstraint, ForeignKey, ForeignKeyConstraint, create_engine, select,
                        insert, update, delete, func, event, cast)
from sqlalchemy.engine import URL, make_url
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import IntegrityError

from .errors import ServerError

metadata = MetaData()
DATABASE_TIMEOUT_SECONDS = 30.0


def _postgres_connection(timeout):
    import psycopg

    class TimedConnection(psycopg.Connection):
        def wait(self, gen, interval=.1, timeout=None):
            allowance = self._server_timeout if timeout is None else min(timeout, self._server_timeout)
            deadline = time.monotonic() + allowance

            def bounded():
                try:
                    state = next(gen)
                    while True:
                        ready = yield state
                        if time.monotonic() >= deadline:
                            # Close in the executing thread, without a blocking
                            # cancellation handshake or an abandoned SQL thread.
                            # COMMIT may already have happened: callers reconcile
                            # using their existing request/ownership identities.
                            gen.close()
                            self.close()
                            raise psycopg.OperationalError('Database operation timed out; commit outcome may be unknown')
                        state = gen.send(ready)
                except StopIteration as result:
                    return result.value
                finally:
                    gen.close()

            # Psycopg 3.2 and 3.3 both resume the generator at this interval.
            return super().wait(bounded(), interval=min(interval or .1, .1))

    TimedConnection._server_timeout = timeout
    return TimedConnection


def _workflow_table(name, template=False):
    columns = [Column('id', String(64), primary_key=True), Column('path', String(768), nullable=False, unique=True),
               Column('name', Text), Column('ext', String(32)), Column('size', BigInteger),
               Column('cover', Text), Column('requirements', Text)]
    if template:
        columns.append(Column('category', Text))
    columns.append(Column('created_at', DateTime, server_default=func.current_timestamp()))
    return Table(name, metadata, *columns)


workflows = _workflow_table('workflows')
templates = _workflow_table('templates', True)
tasks = Table('server_tasks', metadata,
    Column('task_id', String(64), primary_key=True), Column('name', String(128), nullable=False),
    Column('request_id', String(36)),
    Column('callback_url', Text), Column('callback_scope', String(64)),
    Column('callback_state', String(16)), Column('callback_attempts', Integer, nullable=False, server_default='0'),
    Column('callback_next_at', Double, nullable=False, server_default='0'),
    Column('callback_status_code', Integer), Column('callback_error', Text),
    Column('state', String(32), nullable=False), Column('created_at', Double, nullable=False),
    Column('started_at', Double), Column('finished_at', Double),
    Column('outputs', Text().with_variant(LONGTEXT(), 'mysql')), Column('error', Text().with_variant(LONGTEXT(), 'mysql')),
    Column('inputs', Text().with_variant(LONGTEXT(), 'mysql')), Column('parameters', Text().with_variant(LONGTEXT(), 'mysql')),
    Column('input_state', Text),
    Column('owner', String(128)), Column('owner_key', String(64)),
    Column('idempotency_key', String(128)), Column('fingerprint', String(64)),
    Column('attempt', Integer, nullable=False, server_default='1'),
    Column('cancel_requested', Boolean, nullable=False, server_default='0'),
    Column('queue_key', String(64)), Column('group_key', String(64)),
    Column('expires_at', Double), Column('worker_token', String(64)), Column('lease_expires_at', Double),
    Column('runtime_token', String(32)), Column('runtime_holder', String(32)),
    Column('published_at', Double), Column('event_seq', Integer, nullable=False, server_default='0'),
    Column('retention_seconds', Double), Column('purge_after', Double),
    UniqueConstraint('name', 'idempotency_key', name='uq_server_tasks_idempotency'),
    Index('ix_server_tasks_callbacks', 'callback_scope', 'callback_state', 'callback_next_at'),
    Index('ix_server_tasks_name', 'name', 'created_at'),
    Index('ix_server_tasks_purge', 'purge_after', 'task_id'),
    Index('ix_server_tasks_runtime', 'runtime_token', 'state', 'queue_key'),
    Index('ix_server_tasks_owner_state', 'owner_key', 'state'))
subject_locks = Table('server_subject_locks', metadata,
    Column('subject_key', String(64), primary_key=True))
files = Table('server_files', metadata,
    Column('file_id', String(64), primary_key=True), Column('storage_key', String(512), nullable=False, unique=True),
    Column('media_type', String(128), nullable=False), Column('size', BigInteger, nullable=False),
    Column('created_at', Double, nullable=False), Column('sha256', String(64)), Column('owner', String(128)),
    Column('owner_key', String(64)),
    Column('backend', String(128), server_default='local'), Column('state', String(16), nullable=False, server_default='ready'),
    Column('storage_scope', String(128)), Column('purge_after', Double), Column('protected_until', Double),
    Column('cleanup_after', Double), Column('producer_task_id', String(64)),
    Index('ix_server_files_cleanup', 'storage_scope', 'state', 'purge_after'),
    Index('ix_server_files_producer', 'producer_task_id'),
    Index('ix_server_files_owner', 'owner_key'))
file_refs = Table('server_file_refs', metadata,
    Column('file_id', String(64), ForeignKey('server_files.file_id'), primary_key=True),
    Column('task_id', String(64), ForeignKey('server_tasks.task_id', ondelete='CASCADE'), primary_key=True))
server_configs = Table('server_configs', metadata,
    Column('config_id', String(64), primary_key=True), Column('scope', String(64), nullable=False),
    Column('name', String(128), nullable=False), Column('data', Text().with_variant(LONGTEXT(), 'mysql'), nullable=False),
    Column('write_token', String(64)),
    Index('ix_server_configs_scope', 'scope'))
config_file_refs = Table('server_config_file_refs', metadata,
    Column('config_id', String(64), ForeignKey('server_configs.config_id', ondelete='CASCADE'), primary_key=True),
    Column('file_id', String(64), ForeignKey('server_files.file_id'), primary_key=True))
file_holders = Table('server_file_holders', metadata,
    Column('holder', String(64), primary_key=True), Column('host', String(256), nullable=False),
    Column('pid', BigInteger, nullable=False), Column('created_at', Double, nullable=False))
file_holds = Table('server_file_holds', metadata,
    Column('holder', String(64), ForeignKey('server_file_holders.holder', ondelete='CASCADE'), primary_key=True),
    Column('file_id', String(64), ForeignKey('server_files.file_id'), primary_key=True))
queue_limits = Table('server_queue_limits', metadata,
    Column('queue_key', String(64), primary_key=True), Column('capacity', Integer, nullable=False),
    Column('inflight', Integer, nullable=False, server_default='0'),
    Column('accepting', Boolean, nullable=False, server_default='1'),
    Column('control_token', String(32), nullable=False, server_default='0'))
server_meta = Table('server_meta', metadata,
    Column('key', String(64), primary_key=True), Column('value', String(256), nullable=False))
task_events = Table('server_task_events', metadata,
    Column('task_id', String(64), ForeignKey('server_tasks.task_id', ondelete='CASCADE'), primary_key=True),
    Column('sequence', Integer, primary_key=True), Column('kind', String(32), nullable=False),
    Column('data', Text().with_variant(LONGTEXT(), 'mysql'), nullable=False))
task_inputs = Table('server_task_inputs', metadata,
    Column('task_id', String(64), ForeignKey('server_tasks.task_id', ondelete='CASCADE'), primary_key=True),
    Column('sequence', Integer, primary_key=True),
    Column('data', Text().with_variant(LONGTEXT(), 'mysql'), nullable=False))
editor_history = Table('server_editor_history', metadata,
    Column('scope', String(64), primary_key=True),
    Column('task_id', String(64), primary_key=True),
    Column('finished_at', Double, nullable=False),
    Column('record', Text().with_variant(LONGTEXT(), 'mysql'), nullable=False),
    Index('ix_server_editor_history_finished', 'scope', 'finished_at', 'task_id'))
editor_tasks = Table('server_editor_tasks', metadata,
    Column('scope', String(64), primary_key=True), Column('task_id', String(64), primary_key=True),
    Column('token', String(32), nullable=False), Column('holder', String(32), nullable=False),
    Column('state', String(16), nullable=False), Column('created_at', Double, nullable=False),
    Column('sequence', BigInteger, nullable=False), Column('priority', Integer, nullable=False),
    Column('payload', Text().with_variant(LONGTEXT(), 'mysql'), nullable=False),
    Column('cancel_requested', Boolean, nullable=False, server_default='0'),
    Column('dispatched_at', Double), Column('started_at', Double), Column('worker_pid', BigInteger),
    Index('ix_server_editor_tasks_recovery', 'scope', 'state', 'sequence'))
editor_file_refs = Table('server_editor_file_refs', metadata,
    Column('scope', String(64), primary_key=True), Column('task_id', String(64), primary_key=True),
    Column('node_key', String(64), primary_key=True), Column('node_path', Text, nullable=False),
    Column('field_path', Text),
    Column('kind', String(8), nullable=False, server_default='output'),
    Column('file_id', String(64), ForeignKey('server_files.file_id'), nullable=False),
    ForeignKeyConstraint(['scope', 'task_id'], ['server_editor_tasks.scope', 'server_editor_tasks.task_id'], ondelete='CASCADE'),
    Index('ix_server_editor_file_refs_file', 'file_id'))

TERMINAL = ('succeeded', 'failed', 'cancelled', 'expired')


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def _file_ids(value):
    result = set()
    if isinstance(value, dict):
        identity = value.get('file_id')
        if isinstance(identity, str) and identity.startswith('file_'):
            result.add(identity)
        for item in value.values():
            result.update(_file_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_file_ids(item))
    return result


class _EditorHistory:
    """Durable editor task records and the existing workspace history response.

    Admission and recovery share the Server/Worker physical exit evidence.
    Native output objects remain in the local queue cache. They are not part of
    the historical HTTP response and must never be unpickled from a database.
    """

    def __init__(self, db, scope, directory):
        self.db, self.scope = db, scope
        self.directory = Path(directory)
        self.token, self.holder = uuid.uuid4().hex, uuid.uuid4().hex
        self.cursor = -1
        self.counter_key = 'editor-seq-' + hashlib.sha256(scope.encode()).hexdigest()[:40]
        try:
            with self.db.engine.begin() as connection:
                connection.execute(insert(server_meta).values(key=self.counter_key, value='0'))
        except IntegrityError:
            pass
        from .control_guard import ControlGuard
        self.guard = ControlGuard.create(self.directory, self.token, self.holder)
        self.closed = False

    def _where(self, task_id, token=None):
        conditions = [editor_tasks.c.scope == self.scope, editor_tasks.c.task_id == task_id]
        if token is not None:
            conditions.append(editor_tasks.c.token == token)
        return conditions

    def admit(self, payload, priority, submitted, *, idempotent=False, rejection=None, prepare=None):
        if self.closed:
            raise ServerError(503, 'server_stopping', 'Editor has stopped', accepted=False)
        identity = payload['id']
        data = encode(payload)
        with self.db.engine.begin() as connection:
            # This existing metadata row serializes sequence allocation and
            # duplicate admission in the workspace across all three databases.
            connection.execute(update(server_meta).where(server_meta.c.key == self.counter_key).values(
                value=cast(cast(server_meta.c.value, BigInteger) + 1, String)))
            existing = connection.execute(select(editor_tasks).where(*self._where(identity))).mappings().first()
            if existing is not None:
                if idempotent:
                    matches = self.request_hash(json.loads(existing['payload'])) == self.request_hash(payload)
                    if existing['state'] == 'terminal':
                        digest = json.loads(existing['payload']).get('request_sha256')
                        if digest is None:
                            history = connection.execute(select(editor_history.c.record).where(
                                editor_history.c.scope == self.scope,
                                editor_history.c.task_id == identity)).scalar_one_or_none()
                            digest = self.request_hash(json.loads(history)['task']) if history is not None else None
                        matches = digest == self.request_hash(payload)
                    if not matches or existing['priority'] != priority:
                        raise ServerError(409, 'editor_task_conflict', 'Task identity has a different request')
                    return dict(existing)
                if (existing['token'] != self.token or existing['state'] != 'pending'
                        or existing['payload'] != data or existing['priority'] != priority):
                    raise ServerError(409, 'editor_task_conflict', 'Task identity is already in use')
                return dict(existing)
            if connection.execute(select(editor_history.c.task_id).where(
                    editor_history.c.scope == self.scope, editor_history.c.task_id == identity)).first():
                raise ServerError(409, 'editor_task_conflict', 'Task already has a recorded result')
            if rejection is not None:
                raise rejection
            if prepare is not None:
                payload = prepare(payload, connection)
                data = encode(payload)
            sequence = int(connection.execute(select(server_meta.c.value).where(
                server_meta.c.key == self.counter_key)).scalar_one())
            values = dict(scope=self.scope, task_id=identity, token=self.token, holder=self.holder,
                          state='pending', created_at=submitted, sequence=sequence, priority=priority,
                          payload=data, cancel_requested=False)
            connection.execute(insert(editor_tasks).values(**values))
            self._hold_inputs(connection, identity, payload)
            return values

    def _hold_inputs(self, connection, identity, payload):
        from .resources import workflow_file_refs
        try:
            references = workflow_file_refs(payload.get('graph_json', {}))
        except ValueError as exc:
            raise ServerError(422, 'invalid_editor_resources', str(exc), accepted=False) from None
        # Lock before inserting references, in the same order as other file
        # consumers. A concurrent deletion either wins before admission or
        # observes the committed reference; it cannot slip between the two.
        for file_id in sorted({item[2] for item in references}):
            self.db._lock_available_file(connection, file_id)
        existing = set(connection.execute(select(editor_file_refs.c.node_key).where(
            editor_file_refs.c.scope == self.scope, editor_file_refs.c.task_id == identity,
            editor_file_refs.c.kind == 'input')).scalars())
        for node_path, field_path, file_id in references:
            key = hashlib.sha256(encode(['input', node_path, field_path]).encode()).hexdigest()
            if key not in existing:
                connection.execute(insert(editor_file_refs).values(scope=self.scope, task_id=identity,
                    kind='input', node_key=key, node_path=encode(node_path),
                    field_path=encode(field_path), file_id=file_id))
                existing.add(key)

    def dispatch(self, identity):
        with self.db.engine.begin() as connection:
            result = connection.execute(update(editor_tasks).where(*self._where(identity, self.token),
                editor_tasks.c.state.in_(('pending', 'dispatched')), editor_tasks.c.cancel_requested.is_(False)).values(
                    state='dispatched', dispatched_at=func.coalesce(editor_tasks.c.dispatched_at, time.time())))
            if result.rowcount != 1:
                raise ServerError(409, 'editor_dispatch_unconfirmed', 'Task is cancelled or ownership has changed')

    def undispatched(self, identity):
        with self.db.engine.begin() as connection:
            connection.execute(update(editor_tasks).where(*self._where(identity, self.token),
                editor_tasks.c.state == 'dispatched').values(state='pending', dispatched_at=None))

    def started(self, identity, started, pid):
        with self.db.engine.begin() as connection:
            connection.execute(update(editor_tasks).where(*self._where(identity, self.token),
                editor_tasks.c.state.in_(('pending', 'dispatched', 'running'))).values(
                    state='running', started_at=started, worker_pid=pid))

    def cancel(self, identity):
        with self.db.engine.begin() as connection:
            connection.execute(update(editor_tasks).where(*self._where(identity),
                editor_tasks.c.state != 'terminal').values(cancel_requested=True))

    def active(self, identity):
        with self.db.engine.connect() as connection:
            row = connection.execute(select(editor_tasks).where(*self._where(identity),
                editor_tasks.c.state != 'terminal')).mappings().first()
        return self.record(row) if row else None

    def unresolved(self, limit=1000):
        with self.db.engine.connect() as connection:
            rows = connection.execute(select(editor_tasks).where(editor_tasks.c.scope == self.scope,
                editor_tasks.c.token != self.token, editor_tasks.c.state != 'terminal').order_by(
                    editor_tasks.c.sequence).limit(limit)).mappings().all()
        return [self.record(row) for row in rows]

    @staticmethod
    def record(row):
        return dict(task=json.loads(row['payload']), state=row['state'].upper(), ts_submit=row['created_at'],
                    ts_dispatch=row.get('dispatched_at'), ts_start=row.get('started_at'),
                    worker_pid=row.get('worker_pid'), cancel_requested=bool(row['cancel_requested']))

    def recover(self, limit=100):
        if self.closed:
            return []
        from .control_guard import ControlGuard
        with self.db.engine.connect() as connection:
            rows = connection.execute(select(editor_tasks).where(editor_tasks.c.scope == self.scope,
                editor_tasks.c.state != 'terminal', editor_tasks.c.sequence > self.cursor).order_by(
                    editor_tasks.c.sequence).limit(limit)).mappings().all()
        self.cursor = rows[-1]['sequence'] if len(rows) == limit else -1
        recovered = []
        for row in rows:
            if row['token'] == self.token:
                recovered.append(dict(row))
                continue
            with ControlGuard.recover(self.directory, row['token'], row['holder']) as exited:
                if not exited:
                    continue
                with self.db.engine.begin() as connection:
                    locked = connection.execute(update(editor_tasks).where(*self._where(row['task_id'], row['token']),
                        editor_tasks.c.state != 'terminal').values(token=row['token']))
                    if locked.rowcount != 1:
                        continue
                    current = dict(connection.execute(select(editor_tasks).where(*self._where(row['task_id']))).mappings().one())
                    if current['state'] == 'pending' and not current['cancel_requested']:
                        try:
                            # Tasks accepted before input references existed
                            # must be validated and protected before dispatch.
                            self._hold_inputs(connection, row['task_id'], json.loads(current['payload']))
                        except ServerError as exc:
                            record = self.record(current)
                            record.update(state='FAILED', time_profile={},
                                status=dict(completed=False, str='failed', messages=[str(exc)]))
                            self._finish(connection, row['task_id'], record, row['token'])
                            continue
                        connection.execute(update(editor_tasks).where(*self._where(row['task_id'], row['token'])).values(
                            token=self.token, holder=self.holder))
                        current.update(token=self.token, holder=self.holder)
                        recovered.append(current)
                    else:
                        record = self.record(current)
                        cancelled = current['cancel_requested']
                        record.update(state='CANCELLED' if cancelled else 'FAILED', ts_finish=time.time(), time_profile={},
                            status=dict(completed=False, str='cancelled' if cancelled else 'failed',
                                        messages=['Previous editor exited before confirming a result']))
                        self._finish(connection, row['task_id'], record, row['token'])
        return recovered

    def _finish(self, connection, identity, value, token):
        # Retention and pagination use one database clock across Server hosts.
        value['ts_finish'] = self.db.clock(connection)
        if value['state'] == 'SUCCEEDED':
            value['output_files'] = self.db._editor_output_files(connection, self.scope, identity)
        else:
            identities = list(connection.execute(select(editor_file_refs.c.file_id).where(
                editor_file_refs.c.scope == self.scope, editor_file_refs.c.task_id == identity,
                editor_file_refs.c.kind == 'output')).scalars())
            connection.execute(delete(editor_file_refs).where(
                editor_file_refs.c.scope == self.scope, editor_file_refs.c.task_id == identity,
                editor_file_refs.c.kind == 'output'))
            if identities:
                connection.execute(update(files).where(files.c.file_id.in_(identities)).values(purge_after=self.db.clock(connection)))
            value['output_files'] = []
        connection.execute(insert(editor_history).values(scope=self.scope, task_id=identity,
            finished_at=value['ts_finish'], record=encode(value)))
        connection.execute(update(editor_tasks).where(*self._where(identity, token)).values(
            state='terminal', payload=encode({'request_sha256': self.request_hash(value['task'])})))

    def save(self, task_id, record):
        value = {key: item for key, item in record.items() if key != 'results'}
        data = encode(value)
        try:
            with self.db.engine.begin() as connection:
                locked = connection.execute(update(editor_tasks).where(*self._where(task_id, self.token)).values(token=self.token))
                if locked.rowcount != 1:
                    raise ServerError(409, 'editor_ownership_lost', 'Task belongs to another editor')
                receipt = json.loads(connection.execute(select(editor_tasks.c.payload).where(
                    *self._where(task_id, self.token))).scalar_one())
                expired = receipt.get('summary')
                if expired is not None:
                    if receipt.get('request_sha256') != self.request_hash(value['task']) or expired['state'] != value['state']:
                        raise ServerError(409, 'editor_history_conflict', 'Task already has a different recorded result')
                    return expired  # A late acknowledgement cannot recreate removed history.
                self._finish(connection, task_id, value, self.token)
        except IntegrityError:
            existing = self.get(task_id)
            # A commit receipt can be lost. Repeating that same completion
            # keeps the original finish time and never replaces another result.
            comparable = lambda row: {key: item for key, item in row.items() if key != 'ts_finish'}
            if existing is None or comparable(existing) != comparable(value):
                raise ServerError(409, 'editor_history_conflict', 'Task already has a different recorded result')
            return existing
        return value

    def get(self, task_id):
        with self.db.engine.connect() as connection:
            value = connection.execute(select(editor_history.c.record).where(
                editor_history.c.scope == self.scope, editor_history.c.task_id == task_id)).scalar_one_or_none()
            if value is None:
                receipt = connection.execute(select(editor_tasks.c.payload).where(*self._where(task_id),
                    editor_tasks.c.state == 'terminal')).scalar_one_or_none()
                return json.loads(receipt).get('summary') if receipt is not None else None
        return json.loads(value)

    @staticmethod
    def request_hash(payload):
        # Server-resolved execution settings belong to the admitted request,
        # not its client identity. A retry keeps the original settings even if
        # the sidecar was edited or removed after admission.
        return hashlib.sha256(encode({key: value for key, value in payload.items()
                                     if key != '_editor_execution'}).encode()).hexdigest()

    def retained(self, identities):
        if not identities:
            return set()
        with self.db.engine.connect() as connection:
            return set(connection.execute(select(editor_history.c.task_id).where(
                editor_history.c.scope == self.scope, editor_history.c.task_id.in_(identities))).scalars())

    def cleanup(self, retention_seconds, limit=100):
        self.db._validate_retention(retention_seconds)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('History cleanup limit must be between 1 and 1000')
        if retention_seconds is None:
            return []
        removed = []
        with self.db.engine.begin() as connection:
            # Admission and concurrent cleaners serialize on the existing
            # workspace sequence row, then lock individual task rows.
            connection.execute(update(server_meta).where(server_meta.c.key == self.counter_key).values(value=server_meta.c.value))
            now = self.db.clock(connection)
            candidates = connection.execute(select(editor_history.c.task_id).where(
                editor_history.c.scope == self.scope, editor_history.c.finished_at < now - retention_seconds)
                .order_by(editor_history.c.finished_at, editor_history.c.task_id).limit(limit)).scalars().all()
            for identity in candidates:
                self.db._lock_editor_output(connection, self.scope, identity)
                task = connection.execute(select(editor_tasks).where(*self._where(identity))).mappings().first()
                if task is not None and task['state'] != 'terminal':
                    continue
                record = connection.execute(select(editor_history.c.record).where(
                    editor_history.c.scope == self.scope, editor_history.c.task_id == identity)).scalar_one_or_none()
                if record is None:
                    continue
                record = json.loads(record)
                state = record['state']
                if state not in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
                    continue
                summary = dict(task={'id': identity}, state=state, ts_finish=record['ts_finish'],
                    status=dict(completed=state == 'SUCCEEDED', str={'SUCCEEDED': 'success', 'FAILED': 'failed', 'CANCELLED': 'cancelled'}[state],
                                messages=['Saved results were removed by the history retention policy']),
                    outputs={}, output_files=[], outputs_expired=True)
                receipt = encode(dict(request_sha256=self.request_hash(record['task']), summary=summary))
                if task is None:
                    # History written before durable admission also needs a
                    # receipt, otherwise its old identity could execute again.
                    connection.execute(insert(editor_tasks).values(scope=self.scope, task_id=identity,
                        token=self.token, holder=self.holder, state='terminal', created_at=record.get('ts_submit') or now,
                        sequence=0, priority=record['task'].get('priority', 100), payload=receipt, cancel_requested=False))
                else:
                    connection.execute(update(editor_tasks).where(*self._where(identity)).values(payload=receipt))
                identities = connection.execute(select(editor_file_refs.c.file_id).where(
                    editor_file_refs.c.scope == self.scope, editor_file_refs.c.task_id == identity,
                    editor_file_refs.c.kind == 'output')).scalars().all()
                connection.execute(delete(editor_file_refs).where(editor_file_refs.c.scope == self.scope,
                    editor_file_refs.c.task_id == identity))
                if identities:
                    connection.execute(update(files).where(files.c.file_id.in_(identities)).values(purge_after=now))
                connection.execute(delete(editor_history).where(editor_history.c.scope == self.scope,
                    editor_history.c.task_id == identity))
                removed.append(identity)
        return removed

    def latest(self, limit, *, before=None):
        if not 1 <= limit <= 1000:
            raise ValueError('Editor history limit must be between 1 and 1000')
        with self.db.engine.connect() as connection:
            conditions = [editor_history.c.scope == self.scope]
            if before is not None:
                finished = connection.execute(select(editor_history.c.finished_at).where(
                    editor_history.c.scope == self.scope, editor_history.c.task_id == before)).scalar_one_or_none()
                if finished is None:
                    raise ServerError(404, 'history_cursor_not_found', 'History page is no longer available; refresh the latest tasks')
                conditions.append((editor_history.c.finished_at < finished) |
                    ((editor_history.c.finished_at == finished) & (editor_history.c.task_id < before)))
            rows = connection.execute(select(editor_history.c.task_id, editor_history.c.record).where(
                *conditions).order_by(editor_history.c.finished_at.desc(), editor_history.c.task_id.desc()).limit(limit)).all()
        return {row.task_id: json.loads(row.record) for row in reversed(rows)}

    def close(self):
        self.closed = True
        self.guard.close()


class DB:
    """Every operation owns a short connection and transaction; no shared cursor."""
    def __init__(self, location, *, timeout_seconds=None, backup_dir=None):
        timeout = DATABASE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
            raise ValueError('database_timeout_seconds must be a finite positive number, at most 86400')
        self.timeout_seconds = float(timeout)
        if isinstance(location, Path) or '://' not in str(location):
            self.db_path = Path(location).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            url = URL.create('sqlite', database=str(self.db_path))
        else:
            url = make_url(location)
            self.db_path = Path(url.database) if url.get_backend_name() == 'sqlite' and url.database != ':memory:' else None
        if url.get_backend_name() not in ('sqlite', 'mysql', 'postgresql'):
            raise ValueError('Database must use SQLite, MySQL or PostgreSQL')
        if url.drivername == 'mysql':
            url = url.set(drivername='mysql+pymysql')
        elif url.drivername == 'postgresql':
            url = url.set(drivername='postgresql+psycopg')
        driver = url.get_driver_name()
        backend = url.get_backend_name()
        if driver not in {'sqlite': ('pysqlite',), 'mysql': ('pymysql',), 'postgresql': ('psycopg',)}[backend]:
            raise ValueError('Server databases use SQLite/pysqlite, MySQL/PyMySQL or PostgreSQL/psycopg')
        if backend == 'sqlite':
            connect_args = {'check_same_thread': False, 'timeout': timeout}
        elif backend == 'mysql':
            connect_args = {'connect_timeout': timeout, 'read_timeout': timeout, 'write_timeout': timeout}
        else:
            connect_args = {'connect_timeout': max(2, math.ceil(timeout))}
        pool_options = {} if backend == 'sqlite' and url.database in (None, '', ':memory:') else {'pool_timeout': timeout}
        self.engine = create_engine(url, pool_pre_ping=True, hide_parameters=True,
                                    connect_args=connect_args, **pool_options)
        self.url = url
        self.backup_dir = Path(backup_dir) if backup_dir is not None else Path.cwd() / 'resources' / 'db' / 'backups'
        if backend == 'postgresql':
            connection_type = _postgres_connection(timeout)
            @event.listens_for(self.engine, 'do_connect')
            def connect(dialect, record, args, params):
                return connection_type.connect(*args, **params)
        if backend == 'sqlite':
            @event.listens_for(self.engine, 'connect')
            def pragmas(connection, record):
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute(f'PRAGMA busy_timeout={max(1, math.ceil(timeout * 1000))}')
        self._migrated = False
        from .shared import SharedControl
        self.shared = SharedControl(self)

    def init_schema(self):
        from alembic import command
        from alembic.config import Config
        from alembic.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from .migrations.lock import schema_lock
        if not self._migrated:
            config = Config()
            config.set_main_option('script_location', str(Path(__file__).parent / 'migrations'))
            with schema_lock(self) as connection:
                current = set(MigrationContext.configure(connection).get_current_heads())
                expected = set(ScriptDirectory.from_config(config).get_heads())
                if current != expected:
                    if self.engine.dialect.name == 'sqlite' and self.db_path is not None:
                        self._backup_sqlite(connection)
                    elif self.engine.dialect.name != 'sqlite':
                        from sqlalchemy import inspect
                        # A fresh database has no data to protect. Existing
                        # databases must have a completed native backup before
                        # any DDL; missing clients or backup failures stop here.
                        if inspect(connection).get_table_names() or inspect(connection).get_view_names():
                            from .database_backup import backup, SUFFIXES
                            backend = self.engine.dialect.name
                            backup(self, connection, self.backup_dir / (backend + '-' + uuid.uuid4().hex + SUFFIXES[backend]))
                config.attributes['connection'] = connection
                command.upgrade(config, 'head')
            self._migrated = True
        return self

    def backup(self, destination, *, timeout_seconds=300):
        from .database_backup import backup
        from .migrations.lock import schema_lock
        with schema_lock(self) as connection:
            return backup(self, connection, destination, timeout_seconds=timeout_seconds)

    def restore(self, source, *, timeout_seconds=300):
        from .database_backup import restore
        from .migrations.lock import schema_lock
        with schema_lock(self) as connection:
            result = restore(self, connection, source, timeout_seconds=timeout_seconds)
        self._migrated = False
        self.engine.dispose()
        return result

    def export_data(self, destination, *, timeout_seconds=300):
        from .database_transfer import export_data
        return export_data(self, destination, timeout_seconds=timeout_seconds)

    def import_data(self, source, *, timeout_seconds=300):
        from .database_transfer import import_data
        return import_data(self, source, timeout_seconds=timeout_seconds)

    def _backup_sqlite(self, connection):
        from .database_backup import backup
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            return
        destination = self.db_path.with_name(self.db_path.name + '.backup-' + uuid.uuid4().hex + '.sqlite3')
        backup(self, connection, destination)

    def init_tasks(self):
        return self.init_schema()

    def ping(self):
        with self.engine.connect() as connection:
            return connection.execute(select(1)).scalar() == 1

    def clock(self, connection=None):
        """Use the database clock for cross-host deadlines and leases."""
        if connection is None:
            with self.engine.connect() as current:
                return self.clock(current)
        statements = {
            'sqlite': "SELECT (julianday('now') - 2440587.5) * 86400.0",
            'postgresql': 'SELECT EXTRACT(EPOCH FROM clock_timestamp())',
            'mysql': 'SELECT UNIX_TIMESTAMP(CURRENT_TIMESTAMP(6))',
        }
        return float(connection.exec_driver_sql(statements[self.engine.dialect.name]).scalar())

    @staticmethod
    def _stat(path):
        path = Path(path)
        return path.name, path.suffix.lower(), path.stat().st_size if path.exists() else None

    @staticmethod
    def _path(path):
        value = str(Path(path).resolve())
        if len(value) > 768:
            raise ValueError('Managed path exceeds 768 characters')
        return value

    def _one(self, statement):
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
            return dict(row) if row else None

    def insert_workflow(self, name, file_path, cover=None, requirements=None):
        identity = str(uuid.uuid4())
        _, ext, size = self._stat(file_path)
        with self.engine.begin() as connection:
            connection.execute(insert(workflows).values(id=identity, path=self._path(file_path), name=name,
                ext=ext, size=size, cover=cover, requirements=requirements))
        return identity

    def update_workflow_metadata(self, id, name, file_path, cover=None, requirements=None):
        _, ext, size = self._stat(file_path)
        with self.engine.begin() as connection:
            connection.execute(update(workflows).where(workflows.c.id == id).values(
                name=name, ext=ext, size=size, cover=cover, requirements=requirements))

    def upsert_workflow_by_path(self, file_path, name, cover=None, requirements=None):
        statement = select(workflows.c.id).where(workflows.c.path == self._path(file_path))
        row = self._one(statement)
        if row is None:
            try:
                return self.insert_workflow(name, file_path, cover, requirements)
            except IntegrityError:
                row = self._one(statement)
                if row is None:
                    raise
        self.update_workflow_metadata(row['id'], name, file_path, cover, requirements)
        return row['id']

    def get_workflow_path(self, id_):
        row = self._one(select(workflows.c.path).where(workflows.c.id == id_))
        return Path(row['path']) if row else None

    def get_workflow_name(self, id_):
        row = self._one(select(workflows.c.name).where(workflows.c.id == id_))
        return row['name'] if row else None

    def get_workflow_id_and_cover_by_path(self, file_path):
        row = self._one(select(workflows.c.id, workflows.c.cover).where(workflows.c.path == self._path(file_path)))
        return (row['id'], row['cover']) if row else None

    def delete_workflow(self, id_):
        with self.engine.begin() as connection:
            return connection.execute(delete(workflows).where(workflows.c.id == id_)).rowcount > 0

    def insert_or_ignore_template(self, file_path, cover=None, category=None, requirements=None):
        name, ext, size = self._stat(file_path)
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(templates).values(id=str(uuid.uuid4()), path=self._path(file_path),
                    name=name, ext=ext, size=size, cover=cover, category=category, requirements=requirements))
        except IntegrityError:
            if self.get_template_meta_by_path(file_path) is None:
                raise

    def get_template_path(self, id_):
        row = self._one(select(templates.c.path).where(templates.c.id == id_))
        return Path(row['path']) if row else None

    def get_template_meta_by_path(self, file_path):
        row = self._one(select(templates).where(templates.c.path == self._path(file_path)))
        return (row['id'], row['cover'], row['requirements'], row['category']) if row else None

    @staticmethod
    def _validate_retention(seconds):
        if seconds is not None and (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 60):
            raise ValueError('Task retention must be at least 60 seconds or None')

    @staticmethod
    def subject_key(owner):
        # Stable across drivers/collations, including case-sensitive identities
        # on a MySQL database whose default string collation is insensitive.
        return hashlib.sha256(json.dumps(owner, ensure_ascii=False).encode()).hexdigest()

    def _lock_subject(self, connection, owner):
        key = self.subject_key(owner)
        return self._lock_subject_key(connection, key)

    def _lock_subject_key(self, connection, key):
        dialect = connection.dialect.name
        if dialect == 'mysql':
            from sqlalchemy.dialects.mysql import insert as upsert
            statement = upsert(subject_locks).values(subject_key=key)
            statement = statement.on_duplicate_key_update(subject_key=statement.inserted.subject_key)
        else:
            if dialect == 'sqlite':
                from sqlalchemy.dialects.sqlite import insert as upsert
            else:
                from sqlalchemy.dialects.postgresql import insert as upsert
            statement = upsert(subject_locks).values(subject_key=key).on_conflict_do_nothing()
        connection.execute(statement)
        # A write locks on all three backends, including SQLite. Do this before
        # establishing a MySQL consistent-read snapshot or taking queue locks.
        connection.execute(update(subject_locks).where(subject_locks.c.subject_key == key).values(subject_key=key))
        return key

    def _admit_subject(self, connection, owner, limit):
        if limit is None:
            return self.subject_key(owner)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000_000:
            raise ValueError('Subject task limit must be an integer from 1 to 1000000')
        key = self._lock_subject(connection, owner)
        active = connection.execute(select(func.count()).select_from(tasks).where(
            tasks.c.owner_key == key, ~tasks.c.state.in_(TERMINAL))).scalar_one()
        if active >= limit:
            raise ServerError(429, 'subject_tasks_full', 'Identity task limit reached', retryable=True)
        return key

    def create_task(self, task_id, name, created_at, inputs=None, parameters=None, owner=None, idempotency_key=None,
                    *, retention_seconds=None, request_id=None, max_tasks_per_subject=None, input_state=None, callback_url=None, callback_scope=None):
        self._validate_retention(retention_seconds)
        if idempotency_key is not None and (not idempotency_key or len(idempotency_key) > 128):
            raise ValueError('Idempotency key must contain 1 to 128 characters')
        from .callbacks import fingerprint as task_fingerprint, task_values
        fingerprint = task_fingerprint(inputs, parameters, owner, callback_url)
        try:
            with self.engine.begin() as connection:
                owner_key = self._admit_subject(connection, owner, max_tasks_per_subject)
                connection.execute(insert(tasks).values(**task_values(callback_url, callback_scope), task_id=task_id, name=name, state='pending', created_at=created_at,
                    inputs=encode(inputs), parameters=encode(parameters), input_state=encode(input_state) if input_state else None,
                    owner=owner, owner_key=owner_key,
                    idempotency_key=idempotency_key, fingerprint=fingerprint, retention_seconds=retention_seconds, request_id=request_id))
                self._reference_files(connection, task_id, [inputs, parameters])
            return task_id
        except (IntegrityError, ServerError) as exc:
            if isinstance(exc, ServerError) and exc.code != 'subject_tasks_full':
                raise
            if idempotency_key is None:
                raise
            existing = self._one(select(tasks).where(tasks.c.name == name, tasks.c.idempotency_key == idempotency_key))
            if existing is None:
                raise
            if existing['fingerprint'] != fingerprint:
                raise ServerError(409, 'idempotency_conflict', 'Idempotency key was used with different inputs')
            return existing['task_id']

    def find_task(self, name, idempotency_key, inputs, parameters, owner=None, callback_url=None):
        row = self._one(select(tasks.c.task_id, tasks.c.fingerprint).where(
            tasks.c.name == name, tasks.c.idempotency_key == idempotency_key))
        if row:
            from .callbacks import fingerprint as task_fingerprint
            fingerprint = task_fingerprint(inputs, parameters, owner, callback_url)
            if row['fingerprint'] != fingerprint:
                raise ServerError(409, 'idempotency_conflict', 'Idempotency key was used with different inputs')
            return self.get_task(row['task_id'])
        return None

    def start_task(self, task_id, started_at, attempt=1):
        with self.engine.begin() as connection:
            result = connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.state == 'pending',
                tasks.c.attempt == attempt, tasks.c.cancel_requested.is_(False)).values(state='running', started_at=started_at,
                                                                                     event_seq=tasks.c.event_seq + 1))
            if result.rowcount != 1:
                record = connection.execute(select(tasks.c.state, tasks.c.cancel_requested).where(tasks.c.task_id == task_id)).first()
                if record and record.state == 'pending' and record.cancel_requested:
                    raise ServerError(409, 'task_cancelled', 'Task was cancelled before execution', task_id=task_id)
                raise RuntimeError('Task is not pending or ownership has changed')
            self._start_event(connection, task_id)

    def _start_event(self, connection, task_id):
        sequence = connection.execute(select(tasks.c.event_seq).where(tasks.c.task_id == task_id)).scalar_one()
        connection.execute(insert(task_events).values(task_id=task_id, sequence=sequence, kind='start',
            data=encode(dict(task_id=task_id, state='running'))))

    def finish_task(self, task_id, state, finished_at, outputs=None, error=None, attempt=1, worker_token=None):
        if state not in TERMINAL:
            raise ValueError('Invalid terminal task state')
        with self.engine.begin() as connection:
            routing = connection.execute(select(tasks.c.queue_key, tasks.c.group_key).where(tasks.c.task_id == task_id)).first()
            if routing and routing.queue_key:
                # Match admission's lock order before touching the task row.
                for key in (routing.group_key, routing.queue_key):
                    connection.execute(update(queue_limits).where(queue_limits.c.queue_key == key).values(inflight=queue_limits.c.inflight))
            conditions = [tasks.c.task_id == task_id, tasks.c.state.in_(('pending', 'running', 'dispatched')),
                          tasks.c.attempt == attempt, tasks.c.worker_token == worker_token]
            result = connection.execute(update(tasks).where(*conditions).values(
                    state=state, finished_at=finished_at, outputs=encode(outputs), error=encode(error), event_seq=tasks.c.event_seq + 2,
                    purge_after=self.clock(connection) + tasks.c.retention_seconds))
            if result.rowcount != 1:
                raise RuntimeError('Task has finished or ownership has changed')
            if outputs is not None:
                self._reference_files(connection, task_id, outputs)
            if state != 'succeeded':
                self._discard_task_outputs(connection, task_id)
            final = connection.execute(select(tasks.c.event_seq, tasks.c.name, tasks.c.request_id).where(tasks.c.task_id == task_id)).one()
            sequence = final.event_seq
            detail = dict(task_id=task_id, status_url=f'/v1/tasks/{task_id}')
            if state != 'succeeded':
                detail['error'] = dict(code=(error or {}).get('code', state),
                                       message=str((error or {}).get('message', state))[:2048],
                                       retryable=bool((error or {}).get('retryable', False)))
            connection.execute(insert(task_events).values(task_id=task_id, sequence=sequence - 1,
                kind='result' if state == 'succeeded' else 'error', data=encode(detail)))
            connection.execute(insert(task_events).values(task_id=task_id, sequence=sequence, kind='done',
                data=encode(dict(task_id=task_id, state=state, status_url=f'/v1/tasks/{task_id}'))))
            if routing and routing.queue_key:
                for key in (routing.group_key, routing.queue_key):
                    connection.execute(update(queue_limits).where(queue_limits.c.queue_key == key).values(inflight=queue_limits.c.inflight - 1))
        metrics = getattr(self, 'metrics', None)
        if metrics is not None:
            metrics.terminal.labels(final.name, state).inc()
        from .request import event as log_event
        log_event('task_finished', task_id=task_id, request_id=final.request_id, entry=final.name, state=state)

    def append_task_event(self, task_id, value, *, attempt=1, worker_token=None, max_events=10000, max_bytes=65536,
                          allow_cancel_requested=False):
        """Persist one bounded event only while its execution still owns the task."""
        data = encode(value)
        if len(data.encode('utf-8')) > max_bytes:
            raise ServerError(422, 'stream_event_too_large', 'Stream event exceeds the configured limit', task_id=task_id)
        with self.engine.begin() as connection:
            conditions = [tasks.c.task_id == task_id, tasks.c.state == 'running', tasks.c.attempt == attempt,
                          tasks.c.worker_token == worker_token, tasks.c.event_seq <= max_events]
            if not allow_cancel_requested:
                conditions.append(tasks.c.cancel_requested.is_(False))
            result = connection.execute(update(tasks).where(*conditions).values(event_seq=tasks.c.event_seq + 1))
            if result.rowcount != 1:
                row = connection.execute(select(tasks.c.cancel_requested, tasks.c.event_seq).where(tasks.c.task_id == task_id)).first()
                if row and row.cancel_requested:
                    raise ServerError(409, 'task_cancelled', 'Stream cancellation requested', task_id=task_id)
                if row and row.event_seq > max_events:
                    raise ServerError(422, 'stream_event_limit', 'Stream event count exceeds the configured limit', task_id=task_id)
                raise ServerError(409, 'execution_ownership_lost', 'Task execution no longer owns the stream', task_id=task_id)
            self._reference_files(connection, task_id, value)
            sequence = connection.execute(select(tasks.c.event_seq).where(tasks.c.task_id == task_id)).scalar_one()
            connection.execute(insert(task_events).values(task_id=task_id, sequence=sequence, kind='delta',
                data=encode(dict(task_id=task_id, data=value))))
            return sequence

    def read_task_events(self, task_id, after=0, limit=16):
        with self.engine.connect() as connection:
            rows = connection.execute(select(task_events).where(task_events.c.task_id == task_id,
                task_events.c.sequence > after).order_by(task_events.c.sequence).limit(min(limit, 16))).mappings()
            return [dict(sequence=row['sequence'], kind=row['kind'], data=json.loads(row['data'])) for row in rows]

    def database_identity(self):
        statement = select(server_meta.c.value).where(server_meta.c.key == 'identity')
        row = self._one(statement)
        if row is None:
            try:
                with self.engine.begin() as connection:
                    connection.execute(insert(server_meta).values(key='identity', value=uuid.uuid4().hex))
            except IntegrityError:
                pass
            row = self._one(statement)
        return row['value']

    def register_namespace_storage(self, namespace, identity):
        key = hashlib.sha256(('storage:' + namespace).encode()).hexdigest()
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(server_meta).values(key=key, value=identity))
        except IntegrityError:
            row = self._one(select(server_meta.c.value).where(server_meta.c.key == key))
            if row is None or row['value'] != identity:
                raise ServerError(409, 'shared_storage_mismatch',
                    'Shared workers require the same local file directory or object storage configuration')

    def register_queue(self, key, capacity):
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(queue_limits).values(queue_key=key, capacity=capacity))
        except IntegrityError:
            row = self._one(select(queue_limits).where(queue_limits.c.queue_key == key))
            if row is None:
                raise
            if row['capacity'] != capacity:
                raise ServerError(409, 'queue_configuration_conflict', 'Shared queue capacity differs from existing workers')

    def create_shared_task(self, task_id, name, queue_key, group_key, inputs, parameters,
                           queue_timeout, idempotency_key=None, owner=None, *, retention_seconds=None, request_id=None,
                           max_tasks_per_subject=None, input_state=None, callback_url=None, callback_scope=None):
        self._validate_retention(retention_seconds)
        from .callbacks import fingerprint as task_fingerprint, task_values
        fingerprint = task_fingerprint(inputs, parameters, owner, callback_url)
        try:
            with self.engine.begin() as connection:
                owner_key = self._admit_subject(connection, owner, max_tasks_per_subject)
                now = self.clock(connection)
                for key in (group_key, queue_key):
                    if key == queue_key:
                        self.shared.check_admission(connection, queue_key, group_key)
                    reserved = connection.execute(update(queue_limits).where(queue_limits.c.queue_key == key,
                        queue_limits.c.inflight < queue_limits.c.capacity, queue_limits.c.accepting.is_(True)).values(inflight=queue_limits.c.inflight + 1))
                    if reserved.rowcount != 1:
                        row = connection.execute(select(queue_limits.c.accepting).where(queue_limits.c.queue_key == key)).first()
                        if row and not row.accepting:
                            raise ServerError(503, 'not_ready', 'Shared entry is stopped', retryable=True)
                        raise ServerError(429, 'queue_full', 'Shared task capacity reached', retryable=True)
                connection.execute(insert(tasks).values(**task_values(callback_url, callback_scope), task_id=task_id, name=name, queue_key=queue_key, group_key=group_key,
                    state='pending', created_at=now, expires_at=now + queue_timeout, inputs=encode(inputs), parameters=encode(parameters),
                    fingerprint=fingerprint, owner=owner, owner_key=owner_key, idempotency_key=idempotency_key,
                    input_state=encode(input_state) if input_state else None,
                    retention_seconds=retention_seconds, request_id=request_id))
                self._reference_files(connection, task_id, [inputs, parameters])
            return task_id
        except (IntegrityError, ServerError) as exc:
            if isinstance(exc, ServerError) and exc.code not in ('queue_full', 'subject_tasks_full'):
                raise
            if idempotency_key is None:
                raise
            existing = self.find_task(name, idempotency_key, inputs, parameters, owner, callback_url)
            if existing is None:
                raise
            return existing['task_id']

    def pending_tasks(self, queue_key, limit=100):
        with self.engine.connect() as connection:
            # Oldest publication first ensures a lost Redis Stream is repaired
            # without starving later tasks behind repeatedly published early IDs.
            rows = connection.execute(select(tasks.c.task_id).where(tasks.c.queue_key == queue_key,
                tasks.c.state == 'pending').order_by(func.coalesce(tasks.c.published_at, 0), tasks.c.created_at).limit(limit))
            return [row.task_id for row in rows]

    def mark_published(self, task_id):
        with self.engine.begin() as connection:
            connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.state == 'pending').values(published_at=self.clock(connection)))

    def claim_task(self, task_id, queue_key, worker_token, lease_seconds, *, holder=None):
        with self.engine.begin() as connection:
            runtime = self.shared.claim_runtime(connection, queue_key, holder) if holder is not None else {}
            now = self.clock(connection)
            claimed = connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.queue_key == queue_key,
                tasks.c.state == 'pending', tasks.c.cancel_requested.is_(False), tasks.c.expires_at > now).values(
                    state='running', worker_token=worker_token, started_at=now, lease_expires_at=now + lease_seconds,
                    event_seq=tasks.c.event_seq + 1, **runtime))
            if claimed.rowcount == 1:
                self._start_event(connection, task_id)
        return self.get_task(task_id) if claimed.rowcount == 1 else None

    def check_shared_execution(self, task_id, worker_token, attempt):
        """Recheck a claimed task after dynamic batch collection using DB time."""
        with self.engine.connect() as connection:
            row = connection.execute(select(tasks).where(tasks.c.task_id == task_id)).mappings().first()
            if row is None or row['state'] != 'running' or row['worker_token'] != worker_token or row['attempt'] != attempt:
                raise ServerError(409, 'task_ownership_lost', 'Task execution ownership changed', task_id=task_id)
            if row['cancel_requested']:
                raise ServerError(409, 'task_cancelled', 'Task was cancelled before batch execution', task_id=task_id)
            remaining = row['expires_at'] - self.clock(connection)
            if remaining <= 0:
                raise ServerError(504, 'queue_timeout', 'Task expired before batch execution', task_id=task_id)
            return remaining

    def heartbeat_task(self, task_id, worker_token, attempt, lease_seconds):
        with self.engine.begin() as connection:
            changed = connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.worker_token == worker_token,
                tasks.c.attempt == attempt, tasks.c.state == 'running').values(lease_expires_at=self.clock(connection) + lease_seconds))
        return changed.rowcount == 1

    def stalled_tasks(self, queue_key, limit=100):
        with self.engine.connect() as connection:
            rows = connection.execute(select(tasks.c.task_id).where(tasks.c.queue_key == queue_key, tasks.c.state == 'running',
                tasks.c.lease_expires_at < self.clock(connection)).order_by(tasks.c.lease_expires_at).limit(limit))
            return [row.task_id for row in rows]

    def queue_usage(self, queue_key):
        return self._one(select(queue_limits).where(queue_limits.c.queue_key == queue_key))

    def set_queue_accepting(self, queue_key, accepting, *, expected_token=None, write_token=None):
        """Commit control intent, rejecting stale preparations and unsafe resume.

        Repeating an acknowledged-or-uncertain write token is idempotent until
        another control changes the row. Even repeated Stop invalidates an older
        Start still loading a model. Follower workers must not call this method.
        """
        if type(accepting) is not bool or any(value is not None and (
                not isinstance(value, str) or not 1 <= len(value) <= 32) for value in (expected_token, write_token)):
            raise ValueError('Invalid shared queue control')
        if write_token is not None and (expected_token is None or write_token == expected_token):
            raise ValueError('Retryable control writes require distinct expected and write tokens')
        write_token = write_token or uuid.uuid4().hex
        with self.engine.begin() as connection:
            self.shared.lock_control(connection, queue_key)
            self.shared.check_admission(connection, queue_key)
            conditions = [queue_limits.c.queue_key == queue_key]
            if expected_token is not None:
                conditions.append(queue_limits.c.control_token == expected_token)
            if accepting:
                conditions.append((queue_limits.c.accepting.is_(True)) | (queue_limits.c.inflight == 0))
            changed = connection.execute(update(queue_limits).where(*conditions)
                                         .values(accepting=accepting, control_token=write_token))
            if changed.rowcount == 1:
                return write_token
            row = connection.execute(select(queue_limits).where(queue_limits.c.queue_key == queue_key)).mappings().first()
            if row is None:
                raise ServerError(404, 'queue_not_found', 'Shared queue no longer exists')
            if row['control_token'] == write_token and row['accepting'] == accepting:
                return write_token
            if expected_token is not None and row['control_token'] != expected_token:
                raise ServerError(409, 'shared_control_changed', 'Shared entry control changed during preparation; retry using current state', retryable=True)
            raise ServerError(409, 'entry_busy', 'Shared entry is still draining', retryable=True)

    def _reference_files(self, connection, task_id, values):
        owner = connection.execute(select(tasks.c.owner).where(tasks.c.task_id == task_id)).scalar_one()
        for identity in sorted(_file_ids(values)):
            row = connection.execute(select(files.c.state, files.c.owner).where(files.c.file_id == identity).with_for_update()).first()
            if row is None or row.state != 'ready' or row.owner != owner:
                raise ServerError(422, 'invalid_file', 'Referenced file is unavailable')
            exists = connection.execute(select(file_refs.c.file_id).where(
                file_refs.c.file_id == identity, file_refs.c.task_id == task_id)).first()
            if exists is None:
                connection.execute(insert(file_refs).values(file_id=identity, task_id=task_id))

    def append_task_input(self, task_id, message, owner=None):
        from .protocol import TaskInput
        try:
            value = TaskInput.model_validate(message)
            payload = encode([value.end, value.data])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ServerError(422, 'invalid_input', 'Expected a finite JSON input with a positive sequence') from exc
        with self.engine.begin() as connection:
            # The task row serializes writers, consumers, cancellation and
            # completion on SQLite as well as databases with row locks.
            found = connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.owner == owner)
                                       .values(state=tasks.c.state))
            if found.rowcount != 1:
                raise ServerError(404, 'task_not_found', 'Task does not exist')
            task = connection.execute(select(tasks).where(tasks.c.task_id == task_id)).mappings().one()
            state = json.loads(task['input_state']) if task['input_state'] else None
            if state is None:
                raise ServerError(422, 'duplex_unsupported', 'Task does not declare duplex input')
            if len(payload.encode('utf-8')) > state['max_bytes']:
                raise ServerError(413, 'input_too_large', 'Input exceeds the configured event size')
            existing = connection.execute(select(task_inputs.c.data).where(
                task_inputs.c.task_id == task_id, task_inputs.c.sequence == value.sequence)).scalar_one_or_none()
            if existing is not None:
                if existing != payload:
                    raise ServerError(409, 'input_conflict', 'Input sequence was used with different data')
                return dict(task_id=task_id, sequence=value.sequence, end=value.end)
            if task['state'] in TERMINAL or task['cancel_requested'] or state['closed']:
                raise ServerError(409, 'input_closed', 'Task no longer accepts input')
            if value.sequence != state['received'] + 1:
                raise ServerError(409, 'input_sequence', 'Input sequence must follow the last accepted input')
            if not value.end:
                if state['received'] >= state['max_events']:
                    raise ServerError(422, 'input_event_limit', 'Task input event limit reached')
                if state['received'] - state['consumed'] >= state['max_pending']:
                    raise ServerError(429, 'input_queue_full', 'Task input buffer is full', retryable=True)
            self._reference_files(connection, task_id, value.data)
            connection.execute(insert(task_inputs).values(task_id=task_id, sequence=value.sequence, data=payload))
            state.update(received=value.sequence, closed=value.end)
            connection.execute(update(tasks).where(tasks.c.task_id == task_id).values(input_state=encode(state)))
            return dict(task_id=task_id, sequence=value.sequence, end=value.end)

    def read_task_input(self, task_id, after, *, attempt=1, worker_token=None):
        with self.engine.begin() as connection:
            found = connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.state == 'running',
                tasks.c.attempt == attempt, tasks.c.worker_token == worker_token,
                tasks.c.cancel_requested.is_(False)).values(state=tasks.c.state))
            if found.rowcount != 1:
                raise ServerError(409, 'task_cancelled', 'Task ended, was cancelled, or its Worker changed', task_id=task_id)
            raw = connection.execute(select(tasks.c.input_state).where(tasks.c.task_id == task_id)).scalar_one()
            if raw is None:
                raise ServerError(422, 'duplex_unsupported', 'Task does not declare duplex input')
            state = json.loads(raw)
            # Acknowledgement happens only when the algorithm requests its next
            # item. Retain data for idempotent client retries until task cleanup.
            if after > state['consumed']:
                state['consumed'] = min(after, state['received'])
                connection.execute(update(tasks).where(tasks.c.task_id == task_id).values(input_state=encode(state)))
            row = connection.execute(select(task_inputs.c.sequence, task_inputs.c.data).where(
                task_inputs.c.task_id == task_id, task_inputs.c.sequence == after + 1)).first()
            if row is None:
                return None
            end, data = json.loads(row.data)
            return dict(sequence=row.sequence, end=end, data=data)

    def runtime_tasks(self, queue_key, runtime_token, runtime_holder):
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(select(tasks.c.task_id, tasks.c.attempt,
                tasks.c.worker_token, tasks.c.cancel_requested).where(tasks.c.queue_key == queue_key,
                tasks.c.runtime_token == runtime_token, tasks.c.runtime_holder == runtime_holder,
                tasks.c.state == 'running').order_by(tasks.c.task_id).limit(100)).mappings()]

    def request_cancel(self, task_id):
        with self.engine.begin() as connection:
            connection.execute(update(tasks).where(tasks.c.task_id == task_id, ~tasks.c.state.in_(TERMINAL)).values(cancel_requested=True))
        return self.get_task(task_id)

    def get_task(self, task_id):
        result = self._one(select(tasks).where(tasks.c.task_id == task_id))
        if result:
            result['callback'] = ({'url': result['callback_url'], 'state': result['callback_state'],
                'attempts': result['callback_attempts'], 'status_code': result['callback_status_code'],
                'error': result['callback_error']} if result['callback_url'] else None)
            for field in ('inputs', 'parameters', 'outputs', 'error'):
                result[field] = json.loads(result[field]) if result[field] is not None else None
        return result

    def reserve_file(self, record, max_files):
        """Count an upload before any file content is written."""
        with self.engine.begin() as connection:
            key = self._lock_subject(connection, record.get('owner'))
            count = connection.execute(select(func.count()).select_from(files).where(files.c.owner_key == key)).scalar_one()
            if max_files is not None and count >= max_files:
                raise ServerError(429, 'subject_files_full', 'Identity file count limit reached', retryable=True)
            connection.execute(insert(files).values(**dict(record, owner_key=key, state='uploading', size=0)))

    def grow_file(self, identity, owner, size, max_bytes):
        """Reserve actual bytes before copying the next upload chunk."""
        with self.engine.begin() as connection:
            key = self._lock_subject(connection, owner)
            previous = connection.execute(select(files.c.size).where(files.c.file_id == identity,
                files.c.owner_key == key, files.c.state == 'uploading')).scalar_one_or_none()
            if previous is None or size < previous:
                raise ServerError(409, 'upload_changed', 'File upload ownership changed')
            if max_bytes is not None:
                used = connection.execute(select(func.coalesce(func.sum(files.c.size), 0)).where(
                    files.c.owner_key == key)).scalar_one()
                if used + size - previous > max_bytes:
                    raise ServerError(429, 'subject_storage_full', 'Identity storage byte limit reached', retryable=True)
            connection.execute(update(files).where(files.c.file_id == identity).values(size=size))

    def abort_file_upload(self, identity):
        # Call only after the associated content is deleted. Never release a
        # committed file just because its commit acknowledgement was lost.
        with self.engine.begin() as connection:
            return connection.execute(delete(files).where(files.c.file_id == identity,
                files.c.state == 'uploading')).rowcount == 1

    def reconcile_file_upload(self, identity, owner_key):
        """Wait for an uncertain reservation/final commit before inspecting it.

        A normal MVCC read can see 'uploading' while another transaction is
        committing 'ready'. Deleting bytes based on that read loses a committed
        file. Reservation holds the subject lock; finalization holds the file
        row lock. Acquire both, then read their settled outcome.
        """
        with self.engine.begin() as connection:
            self._lock_subject_key(connection, owner_key)
            connection.execute(update(files).where(files.c.file_id == identity,
                files.c.owner_key == owner_key).values(state=files.c.state))
            row = connection.execute(select(files).where(files.c.file_id == identity,
                files.c.owner_key == owner_key)).mappings().first()
            return dict(row) if row is not None else None

    @staticmethod
    def _lock_editor_output(connection, scope, task_id, token=None):
        conditions = [editor_tasks.c.scope == scope, editor_tasks.c.task_id == task_id]
        if token is not None:
            conditions.extend((editor_tasks.c.token == token,
                               editor_tasks.c.state.in_(('dispatched', 'running')),
                               editor_tasks.c.cancel_requested.is_(False)))
        changed = connection.execute(update(editor_tasks).where(*conditions).values(state=editor_tasks.c.state))
        if token is not None and changed.rowcount != 1:
            raise ServerError(409, 'editor_ownership_lost', 'Editor task cannot publish output files', task_id=task_id)

    def reconcile_editor_file(self, identity, scope, task_id):
        with self.engine.begin() as connection:
            # Reservation/growth owns the subject lock; final publication owns
            # the editor row. Wait for either uncertain transaction to settle.
            self._lock_subject_key(connection, self.subject_key('editor:' + scope))
            self._lock_editor_output(connection, scope, task_id)
            row = connection.execute(select(files).where(files.c.file_id == identity)).mappings().first()
            return dict(row) if row is not None else None

    @staticmethod
    def _editor_output_files(connection, scope, task_id):
        rows = connection.execute(select(editor_file_refs.c.node_path, editor_file_refs.c.field_path, files.c.file_id,
            files.c.media_type, files.c.size, files.c.sha256).join(files,
                files.c.file_id == editor_file_refs.c.file_id).where(
                    editor_file_refs.c.scope == scope, editor_file_refs.c.task_id == task_id,
                    editor_file_refs.c.kind == 'output')
                .order_by(editor_file_refs.c.node_key)).mappings()
        return [dict(node_path=json.loads(row['node_path']), file_id=row['file_id'],
                     **({'field_path': json.loads(row['field_path'])} if row['field_path'] is not None else {}),
                     media_type=row['media_type'], size=row['size'], sha256=row['sha256'],
                     url=f"/api/queue/task/{task_id}/files/{row['file_id']}") for row in rows]

    def editor_output_files(self, scope, task_id):
        with self.engine.connect() as connection:
            return self._editor_output_files(connection, scope, task_id)

    def create_file(self, record, *, task_id=None, attempt=1, worker_token=None, retention_seconds=None, reserved=False,
                    editor_task=None):
        self._validate_retention(retention_seconds)
        if task_id is not None and editor_task is not None:
            raise ValueError('A file has only one producing task')
        with self.engine.begin() as connection:
            record = dict(record)
            record['owner_key'] = self.subject_key(record.get('owner'))
            if retention_seconds is not None:
                record['purge_after'] = self.clock(connection) + retention_seconds
            if task_id is not None:
                changed = connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.state == 'running',
                    tasks.c.attempt == attempt, tasks.c.worker_token == worker_token, tasks.c.owner == record.get('owner'))
                    .values(state=tasks.c.state))
                if changed.rowcount != 1:
                    raise ServerError(409, 'execution_ownership_lost', 'Task cannot publish output files', task_id=task_id)
                record['producer_task_id'] = task_id
            if editor_task is not None:
                scope, identity, token, node_path = editor_task[:4]
                field_path = editor_task[4] if len(editor_task) > 4 else None
                self._lock_editor_output(connection, scope, identity, token)
                if record.get('owner') != 'editor:' + scope:
                    raise ServerError(409, 'editor_ownership_lost', 'Editor output owner does not match its workspace')
            if reserved:
                changed = connection.execute(update(files).where(files.c.file_id == record['file_id'],
                    files.c.owner_key == record['owner_key'], files.c.state == 'uploading',
                    files.c.size == record['size'], files.c.storage_scope == record['storage_scope'],
                    files.c.backend == record['backend']).values(**dict(record, state='ready')))
                if changed.rowcount != 1:
                    raise ServerError(409, 'upload_changed', 'File upload ownership changed')
            else:
                connection.execute(insert(files).values(**record))
            if task_id is not None:
                connection.execute(insert(file_refs).values(file_id=record['file_id'], task_id=task_id))
            if editor_task is not None:
                path = encode(node_path)
                key = path if field_path is None else encode([node_path, field_path])
                connection.execute(insert(editor_file_refs).values(scope=scope, task_id=identity,
                    node_key=hashlib.sha256(key.encode()).hexdigest(), node_path=path,
                    field_path=encode(field_path) if field_path is not None else None, file_id=record['file_id']))
        return record

    def _discard_task_outputs(self, connection, task_id, file_id=None):
        conditions = [files.c.producer_task_id == task_id]
        if file_id is not None:
            conditions.append(files.c.file_id == file_id)
        identities = list(connection.execute(select(files.c.file_id).where(*conditions)).scalars())
        if identities:
            connection.execute(delete(file_refs).where(file_refs.c.task_id == task_id, file_refs.c.file_id.in_(identities)))
            connection.execute(update(files).where(files.c.file_id.in_(identities)).values(purge_after=self.clock(connection)))
        return bool(identities)

    def discard_task_output(self, task_id, file_id, *, attempt=1, worker_token=None):
        with self.engine.begin() as connection:
            changed = connection.execute(update(tasks).where(tasks.c.task_id == task_id, tasks.c.state != 'succeeded',
                tasks.c.attempt == attempt, tasks.c.worker_token == worker_token).values(state=tasks.c.state))
            if changed.rowcount != 1:
                return False
            return self._discard_task_outputs(connection, task_id, file_id)

    def protect_download(self, file_id, scope, seconds):
        if type(seconds) is not int or not 1 <= seconds <= 3600:
            raise ValueError('Download URL expiry must be between 1 and 3600 seconds')
        with self.engine.begin() as connection:
            self._lock_available_file(connection, file_id)
            row = connection.execute(select(files).where(files.c.file_id == file_id)).mappings().one()
            if row['storage_scope'] not in (None, scope):
                raise ServerError(503, 'storage_mismatch', 'The file belongs to another storage location')
            until = max(row['protected_until'] or 0, self.clock(connection) + seconds)
            connection.execute(update(files).where(files.c.file_id == file_id).values(protected_until=until))

    def _cleanup_eligible(self, scope, now):
        return (files.c.storage_scope == scope,
                (files.c.cleanup_after.is_(None) | (files.c.cleanup_after <= now)),
                ((files.c.state == 'deleting') | ((files.c.state == 'ready') & (files.c.purge_after <= now) &
                 (files.c.protected_until.is_(None) | (files.c.protected_until <= now)))),
                *(~reference for reference in self._file_references(files.c.file_id)))

    def files_for_cleanup(self, scope, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Cleanup limit must be between 1 and 1000')
        with self.engine.connect() as connection:
            return list(connection.execute(select(files.c.file_id).where(*self._cleanup_eligible(scope, self.clock(connection)))
                .order_by(files.c.created_at, files.c.file_id).limit(limit)).scalars())

    def claim_file_cleanup(self, file_id, scope):
        with self.engine.begin() as connection:
            self._check_file_delete(connection, file_id)
            changed = connection.execute(update(files).where(files.c.file_id == file_id,
                *self._cleanup_eligible(scope, self.clock(connection))).values(state='deleting'))
            return changed.rowcount == 1

    def defer_file_cleanup(self, file_id, scope):
        with self.engine.begin() as connection:
            connection.execute(update(files).where(files.c.file_id == file_id, files.c.storage_scope == scope,
                files.c.state == 'deleting').values(cleanup_after=self.clock(connection) + 60))

    def cleanup_tasks(self, limit=100):
        """Delete one bounded batch of expired terminal records and their child rows.

        Retention is recorded at admission and starts at terminal commit using the
        DB clock. A janitor cannot shorten another Server's retention policy.
        Cascades release task file references, never the file content itself.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Cleanup limit must be between 1 and 1000')
        with self.engine.begin() as connection:
            eligible = (tasks.c.state.in_(TERMINAL), tasks.c.purge_after <= self.clock(connection),
                        (tasks.c.callback_state.is_(None) | (tasks.c.callback_state != 'pending')))
            identities = list(connection.execute(select(tasks.c.task_id).where(*eligible)
                .order_by(tasks.c.purge_after, tasks.c.task_id).limit(limit)).scalars())
            if not identities:
                return 0
            # Recheck in the DELETE: multiple cleaners may select the same rows.
            # FK cascades remove events/references in the same transaction, while
            # the unique idempotency key is released only by this committed delete.
            return connection.execute(delete(tasks).where(tasks.c.task_id.in_(identities), *eligible)).rowcount

    def get_file(self, file_id):
        return self._one(select(files).where(files.c.file_id == file_id))

    def bind_file_storage(self, file_id, scope):
        with self.engine.begin() as connection:
            connection.execute(update(files).where(files.c.file_id == file_id, files.c.storage_scope.is_(None))
                               .values(storage_scope=scope))
            current = connection.execute(select(files.c.storage_scope).where(files.c.file_id == file_id)).scalar()
            if current != scope:
                raise ServerError(503, 'storage_mismatch', 'The file belongs to another storage location')

    @staticmethod
    def _lock_available_file(connection, identity):
        # A write also serializes SQLite, which ignores SELECT FOR UPDATE.
        changed = connection.execute(update(files).where(files.c.file_id == identity, files.c.state == 'ready')
                                     .values(state=files.c.state))
        if changed.rowcount != 1:
            raise ServerError(422, 'invalid_file', 'Referenced file is unavailable')

    def hold_files(self, holder, values):
        """Protect declared resources until this specific runtime has closed."""
        identities = sorted(_file_ids(values))
        if not identities:
            return
        with self.engine.begin() as connection:
            if connection.execute(select(file_holders.c.holder).where(file_holders.c.holder == holder)).first() is None:
                connection.execute(insert(file_holders).values(holder=holder, host=socket.gethostname(), pid=os.getpid(),
                                                               created_at=self.clock(connection)))
            for identity in identities:
                self._lock_available_file(connection, identity)
                exists = connection.execute(select(file_holds.c.file_id).where(
                    file_holds.c.holder == holder, file_holds.c.file_id == identity)).first()
                if exists is None:
                    connection.execute(insert(file_holds).values(holder=holder, file_id=identity))

    def release_files(self, holder):
        with self.engine.begin() as connection:
            self.shared.lock_holder(connection, holder)
            connection.execute(delete(file_holders).where(file_holders.c.holder == holder))

    def save_server_config(self, scope, name, value, *, only_if_absent=False, write_token=None,
                           expected_token=None, checked=False):
        """Commit a workflow's restart configuration and file references together."""
        identity = hashlib.sha256(encode([scope, name]).encode()).hexdigest()
        data = encode(value)
        write_token = write_token or uuid.uuid4().hex
        # A concurrent first insert can win; retry the ordinary transactional
        # update. Legacy imports never overwrite an already committed config.
        for attempt in range(2):
            try:
                with self.engine.begin() as connection:
                    existing = connection.execute(select(server_configs).where(
                        server_configs.c.config_id == identity).with_for_update()).mappings().first()
                    if existing and only_if_absent:
                        return existing['write_token']
                    if existing and existing['write_token'] == write_token:
                        if existing['data'] != data:
                            raise ServerError(409, 'configuration_conflict', 'Configuration operation was reused with different content')
                        return write_token
                    if checked and (existing['write_token'] if existing else None) != expected_token:
                        raise ServerError(409, 'configuration_conflict', 'Saved configuration changed in another Server')
                    if existing:
                        conditions = [server_configs.c.config_id == identity]
                        if checked:
                            conditions.append(server_configs.c.write_token == expected_token)
                        changed = connection.execute(update(server_configs).where(*conditions).values(data=data, write_token=write_token))
                        if changed.rowcount != 1:
                            committed = connection.execute(select(server_configs.c.write_token, server_configs.c.data).where(
                                server_configs.c.config_id == identity)).first()
                            if committed and committed.write_token == write_token and committed.data == data:
                                return write_token
                            raise ServerError(409, 'configuration_conflict', 'Saved configuration changed in another Server')
                    else:
                        connection.execute(insert(server_configs).values(config_id=identity, scope=scope, name=name, data=data,
                                                                        write_token=write_token))
                    identities = sorted(_file_ids([value.get('config'), value.get('workflow'), value.get('resources')]))
                    for file in identities:
                        self._lock_available_file(connection, file)
                    connection.execute(delete(config_file_refs).where(config_file_refs.c.config_id == identity))
                    if identities:
                        connection.execute(insert(config_file_refs), [dict(config_id=identity, file_id=file) for file in identities])
                return write_token
            except IntegrityError:
                if attempt:
                    raise

    def get_server_config(self, scope, name):
        identity = hashlib.sha256(encode([scope, name]).encode()).hexdigest()
        row = self._one(select(server_configs.c.data, server_configs.c.write_token).where(server_configs.c.config_id == identity))
        return dict(data=json.loads(row['data']), write_token=row['write_token']) if row else None

    def load_server_configs(self, scope, *, with_tokens=False):
        with self.engine.connect() as connection:
            rows = connection.execute(select(server_configs.c.data, server_configs.c.write_token).where(server_configs.c.scope == scope)
                                      .order_by(server_configs.c.name))
            if with_tokens:
                return [dict(data=json.loads(row.data), write_token=row.write_token) for row in rows]
            return [json.loads(row.data) for row in rows]

    @staticmethod
    def _file_references(file_id):
        return [select(table.c.file_id).where(table.c.file_id == file_id).exists()
                for table in (file_refs, config_file_refs, file_holds, editor_file_refs)]

    def delete_file(self, file_id):
        with self.engine.begin() as connection:
            self._check_file_delete(connection, file_id)
            return connection.execute(delete(files).where(files.c.file_id == file_id)).rowcount == 1

    def _check_file_delete(self, connection, file_id):
        connection.execute(select(files.c.file_id).where(files.c.file_id == file_id).with_for_update()).first()
        if any(connection.execute(select(reference)).scalar() for reference in self._file_references(file_id)):
            raise ServerError(409, 'file_in_use', 'File is referenced by a task, saved workflow or running entry')

    def mark_file_deleting(self, file_id):
        with self.engine.begin() as connection:
            self._check_file_delete(connection, file_id)
            # SQLite ignores FOR UPDATE. This final conditional write also
            # closes its race between the reference read and the state change.
            result = connection.execute(update(files).where(files.c.file_id == file_id,
                *(~reference for reference in self._file_references(file_id))).values(state='deleting'))
            if result.rowcount != 1 and connection.execute(select(files.c.file_id).where(files.c.file_id == file_id)).first():
                raise ServerError(409, 'file_in_use', 'File is referenced by a task, saved workflow or running entry')

    def close(self):
        self.engine.dispose()

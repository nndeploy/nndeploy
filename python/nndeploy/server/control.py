"""Durable HTTP control receipts over the existing Server lifecycle methods."""
import asyncio
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import Table, Column, String, Integer, Double, Text, Index, select, insert, update, delete
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import IntegrityError, OperationalError, InterfaceError, DisconnectionError, TimeoutError as PoolTimeout
from pydantic import ValidationError
from starlette.exceptions import HTTPException

from .control_guard import ControlGuard
from .db import metadata, encode
from .errors import ServerError


receipts = Table('server_control_requests', metadata,
    Column('identity', String(64), primary_key=True), Column('request_id', String(36), nullable=False),
    Column('scope', String(64), nullable=False), Column('owner', String(128), nullable=False),
    Column('name', String(128), nullable=False), Column('action', String(16), nullable=False),
    Column('fingerprint', String(64), nullable=False), Column('state', String(16), nullable=False),
    Column('status_code', Integer), Column('response', Text().with_variant(LONGTEXT(), 'mysql')),
    Column('created_at', Double, nullable=False), Column('expires_at', Double),
    Index('ix_server_control_request', 'scope', 'owner', 'request_id'),
    Index('ix_server_control_expiry', 'expires_at'))

RETRY = (OperationalError, InterfaceError, DisconnectionError, PoolTimeout, OSError)
RETENTION_SECONDS = 7 * 24 * 3600
DATABASE_WAIT_SECONDS = 5.0
MAX_CONTROL_OPERATIONS = 64
MAX_CONTROL_QUERIES = 64
MAX_RESULT_BYTES = 32 * 1024 * 1024
RESULT_FIELDS = ('identity', 'request_id', 'scope', 'owner', 'name', 'action', 'fingerprint')


class Controls:
    def __init__(self, server):
        self.server, self.db = server, server.db
        scope = ['shared', server.redis_config.namespace] if server.redis_config else ['workspace', server.config_scope]
        self.scope = hashlib.sha256(encode(['http-control', scope]).encode()).hexdigest()
        self.database_id = self.db.database_identity()
        self.results = self.server.data_dir / 'control-results'
        self._result_scan = None
        self._scan_lock = threading.Lock()
        self.pending, self.reads = {}, {}
        # Receipt I/O has a bounded, separate pool: a slow management database
        # call must not consume every thread used by Deploy admission/results.
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='nndeploy-control-db')

    def close(self):
        # Server joins owned operations/queries before closing this pool or DB.
        self.pool.shutdown(wait=True)
        if self._result_scan is not None:
            self._result_scan.close()
            self._result_scan = None

    async def database(self, operation, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, operation, *args)

    async def wait_for(self, future, timeout):
        observer = self.server._observe_result(future)
        try:
            return await asyncio.wait_for(observer, timeout)
        finally:
            observer.cancel()

    def unavailable(self, request_id):
        return ServerError(503, 'control_unconfirmed',
            'Database confirmation is unavailable; the operation may still complete. Retry with the same Idempotency-Key or status URL',
            status_url=self.url(request_id), retryable=True)

    def admission_pending(self, pending):
        from fastapi.responses import JSONResponse
        # A delayed duplicate may eventually resolve to an older receipt ID.
        # Until reserve confirms that ID, only the original key is a safe retry
        # address. Give keyless clients the key generated for this operation.
        error = ServerError(503, 'control_unconfirmed',
            'Database acceptance is not confirmed; the operation may still complete. Retry the same operation with the returned Idempotency-Key',
            retryable=True)
        return JSONResponse(dict(request_id=pending['request_id'], **error.response()), status_code=503,
                            headers={'Idempotency-Key': pending['key'], 'Retry-After': '1', 'Cache-Control': 'no-store'})

    @staticmethod
    def url(request_id):
        return f'/api/server?request_id={request_id}'

    def identity(self, request_id, owner, name, action, body, key):
        identity = hashlib.sha256(encode([self.scope, owner, name,
            ['key', key] if key is not None else ['request', request_id]]).encode()).hexdigest()
        try:
            fingerprint = hashlib.sha256(encode([action, body]).encode()).hexdigest()
        except (TypeError, ValueError) as exc:
            raise ServerError(422, 'invalid_request', 'Control request must contain finite JSON values') from exc
        return identity, fingerprint

    def reserve(self, request_id, owner, name, action, body, key):
        identity, fingerprint = self.identity(request_id, owner, name, action, body, key)
        while True:
            try:
                with self.db.engine.begin() as connection:
                    now = self.db.clock(connection)
                    row = connection.execute(select(receipts).where(receipts.c.identity == identity).with_for_update()).mappings().first()
                    if row and row['expires_at'] is not None and row['expires_at'] <= now:
                        removed = connection.execute(delete(receipts).where(receipts.c.identity == identity,
                            receipts.c.request_id == row['request_id'], receipts.c.expires_at <= now))
                        if removed.rowcount != 1:
                            continue  # SQLite readers may have seen an older generation.
                        row = None
                    if row:
                        if row['fingerprint'] != fingerprint:
                            raise ServerError(409, 'idempotency_conflict', 'Control key was already used with a different request')
                        return dict(row)
                    value = dict(identity=identity, request_id=request_id, scope=self.scope, owner=owner,
                        name=name, action=action, fingerprint=fingerprint, state='running', status_code=None,
                        response=None, created_at=now, expires_at=None)
                    connection.execute(insert(receipts).values(**value))
                    return value
            except IntegrityError:
                # A competing replica committed the same identity. The next
                # transaction reads its receipt, never invokes its operation.
                continue

    def finish(self, record, status, response):
        result = encode(response)
        with self.db.engine.begin() as connection:
            changed = connection.execute(update(receipts).where(receipts.c.identity == record['identity'],
                receipts.c.request_id == record['request_id'], receipts.c.state.in_(('running', 'unconfirmed'))).values(
                state='succeeded' if status < 400 else 'failed', status_code=status, response=result,
                expires_at=self.db.clock(connection) + RETENTION_SECONDS))
            if changed.rowcount == 1:
                return
            row = connection.execute(select(receipts).where(receipts.c.identity == record['identity']).with_for_update()).mappings().first()
            if row is None or row['request_id'] != record['request_id']:
                raise RuntimeError('Control receipt no longer belongs to this request')
            if row['state'] in ('succeeded', 'failed'):
                if row['status_code'] != status or row['response'] != result:
                    raise RuntimeError('Control result changed after it was committed')
                return
            raise RuntimeError('Control receipt has an invalid state')

    def result_path(self, request_id):
        token = request_id.removeprefix('req_')
        if request_id != 'req_' + token or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('Invalid control result request ID')
        return self.results / (request_id + '.json')

    @staticmethod
    def _sync_directory(directory):
        if os.name != 'nt':
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def checkpoint(self, record, status, response):
        """Keep an already obtained response when its DB commit is unavailable."""
        value = dict(version=1, database=self.database_id,
                     record={key: record[key] for key in RESULT_FIELDS}, status=status, response=response)
        value['sha256'] = hashlib.sha256(encode(value).encode()).hexdigest()
        content = encode(value).encode()
        if len(content) > MAX_RESULT_BYTES:
            raise ValueError('Control result exceeds the recovery file size limit')
        path = self.result_path(record['request_id'])
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        pending = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.result-', delete=False) as output:
                pending = Path(output.name)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(pending, path)
            except FileExistsError:
                with path.open('rb') as existing:
                    if existing.read(MAX_RESULT_BYTES + 1) != content:
                        raise RuntimeError('Control result changed after it was checkpointed')
            self._sync_directory(path.parent)
            self._sync_directory(path.parent.parent)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)

    def read_result(self, path):
        try:
            with path.open('rb') as source:
                content = source.read(MAX_RESULT_BYTES + 1)
            if len(content) > MAX_RESULT_BYTES:
                raise ValueError('oversized')
            value = json.loads(content)
            if not isinstance(value, dict) or set(value) != {'version', 'database', 'record', 'status', 'response', 'sha256'}:
                raise ValueError('invalid fields')
            checksum = value.pop('sha256')
            record = value['record']
            if (type(value['version']) is not int or value['version'] != 1 or
                    value['database'] != self.database_id or
                    not isinstance(record, dict) or set(record) != set(RESULT_FIELDS) or
                    not all(isinstance(record[key], str) for key in RESULT_FIELDS) or
                    record['scope'] != self.scope or self.result_path(record['request_id']) != path or
                    type(value['status']) is not int or not 100 <= value['status'] <= 599 or
                    checksum != hashlib.sha256(encode(value).encode()).hexdigest()):
                raise ValueError('invalid identity or checksum')
            return value
        except FileNotFoundError:
            return None  # Another reader may already have confirmed this result.
        except (ValueError, TypeError, UnicodeError, RecursionError):
            logging.error('Rejected invalid Server control recovery file: %s', path.name)
            return None

    def discard_result(self, record):
        path = self.result_path(record['request_id'])
        value = self.read_result(path)
        if (value is not None and value['record'] == {key: record[key] for key in RESULT_FIELDS}
                and record['state'] in ('succeeded', 'failed')
                and value['status'] == record['status_code'] and encode(value['response']) == record['response']):
            path.unlink(missing_ok=True)
            self._sync_directory(path.parent)
        if record['state'] in ('succeeded', 'failed'):
            # HTTP receipt guards have no model children. Their known terminal
            # result is sufficient; runtime ownership guards are never touched.
            guard = self.server.data_dir / 'control-locks' / record['request_id'].removeprefix('req_')
            try:
                guard.unlink()
            except FileNotFoundError:
                pass
            else:
                self._sync_directory(guard.parent)

    def reconcile(self, limit=100):
        """Bounded maintenance over local results, independent of model loading."""
        checked = 0
        with self._scan_lock:
            if self._result_scan is None:
                try:
                    self._result_scan = os.scandir(self.results)
                except FileNotFoundError:
                    return 0
            while checked < limit:
                try:
                    entry = next(self._result_scan)
                except StopIteration:
                    self._result_scan.close()
                    self._result_scan = None
                    break
                checked += 1
                if not entry.name.endswith('.json') or not entry.is_file(follow_symlinks=False):
                    continue
                value = self.read_result(Path(entry.path))
                if value is None:
                    continue
                try:
                    self.get(value['record']['request_id'], value['record']['owner'])
                except ServerError as error:
                    if error.status_code != 404:
                        raise
                    # An absent receipt may indicate a restored older database;
                    # do not invent acceptance or discard the remaining evidence.
        return checked

    def get(self, request_id, owner):
        with self.db.engine.connect() as connection:
            row = connection.execute(select(receipts).where(receipts.c.scope == self.scope,
                receipts.c.owner == owner, receipts.c.request_id == request_id)).mappings().first()
            if row is None:
                raise ServerError(404, 'request_not_found', 'Unknown Server control request')
            record = dict(row)
        active = any(pending['request_id'] == request_id for pending in list(self.pending.values()))
        if record['state'] in ('running', 'unconfirmed') and not active:
            value = self.read_result(self.result_path(request_id))
            if value is not None and value['record'] == {key: record[key] for key in RESULT_FIELDS}:
                # This proves the old response, not that a model is still ready
                # or that its Worker has exited. Never invoke the action again.
                self.finish(record, value['status'], value['response'])
                with self.db.engine.connect() as connection:
                    row = connection.execute(select(receipts).where(receipts.c.identity == record['identity'],
                        receipts.c.request_id == request_id)).mappings().first()
                    if row is None:
                        raise ServerError(404, 'request_not_found', 'Unknown Server control request')
                    record = dict(row)
        if record['state'] == 'running':
            token = request_id.removeprefix('req_')
            with ControlGuard.recover(self.server.data_dir, token, token) as exited:
                if exited:
                    with self.db.engine.begin() as connection:
                        connection.execute(update(receipts).where(receipts.c.identity == record['identity'],
                            receipts.c.request_id == request_id, receipts.c.state == 'running').values(state='unconfirmed'))
                        record = dict(connection.execute(select(receipts).where(receipts.c.identity == record['identity'])).mappings().one())
        if record['state'] in ('succeeded', 'failed'):
            try:
                self.discard_result(record)
            except OSError:
                # The database result is authoritative even if local cleanup
                # is temporarily unavailable. Maintenance retries it later.
                logging.warning('Server control recovery file cleanup awaits storage recovery')
        return record

    def cleanup(self, limit=100):
        with self.db.engine.begin() as connection:
            ids = list(connection.execute(select(receipts.c.identity).where(receipts.c.scope == self.scope,
                receipts.c.expires_at <= self.db.clock(connection)).order_by(receipts.c.expires_at)
                .limit(limit).with_for_update()).scalars())
            if ids:
                return connection.execute(delete(receipts).where(receipts.c.identity.in_(ids),
                    receipts.c.expires_at <= self.db.clock(connection))).rowcount
            return 0

    async def write(self, operation, *args, finishing=False):
        # A completed action still gets one commit attempt during shutdown.
        # Close joins this operation before disposing the database; it must
        # not turn an otherwise known result into an orphan just by stopping.
        while finishing or not self.server.shutting_down:
            try:
                return await self.database(operation, *args)
            except RETRY:
                if self.server.shutting_down:
                    break
                logging.warning('Server control receipt awaits database recovery')
                try:
                    await asyncio.wait_for(self.server.maintenance_stop.wait(), .5)
                except asyncio.TimeoutError:
                    pass
        raise ServerError(503, 'control_unconfirmed', 'Control result is not confirmed; query the original request', retryable=True)

    async def run(self, accepted, request_id, owner, name, action, body, key, operation):
        guard = None
        confirmed = False
        try:
            token = request_id.removeprefix('req_')
            guard = ControlGuard.create(self.server.data_dir, token, token)
            record = await self.write(self.reserve, request_id, owner, name, action, body, key)
            accepted.set_result(record)
            if record['request_id'] != request_id or record['state'] in ('succeeded', 'failed', 'unconfirmed'):
                confirmed = True
                return
            try:
                response, status = await operation(), 200
            except ServerError as exc:
                response, status = exc.response(), exc.status_code
            except ValidationError:
                response, status = ServerError(422, 'invalid_configuration', 'Server configuration is invalid').response(), 422
            except HTTPException as exc:
                status = exc.status_code
                response = ({'detail': exc.detail} if status < 500 else
                    ServerError(status, 'internal_error', 'Server control request failed').response())
            except Exception:
                logging.exception('Server control request failed')
                response, status = ServerError(500, 'internal_error', 'Server control request failed').response(), 500
            # describe() includes mutable schema/config values. Keep the exact
            # returned response before yielding to storage or another control.
            response = copy.deepcopy(response)
            try:
                await self.database(self.checkpoint, record, status, response)
            except (OSError, ValueError):
                # Disk failure must not prevent committing an already known
                # result to a healthy DB. If both fail, it remains unconfirmed.
                logging.error('Server control response could not be checkpointed')
            await self.write(self.finish, record, status, response, finishing=True)
            confirmed = True
            try:
                terminal = dict(record, state='succeeded' if status < 400 else 'failed',
                                status_code=status, response=encode(response))
                await self.database(self.discard_result, terminal)
            except OSError:
                logging.warning('Server control recovery file cleanup awaits storage recovery')
        except BaseException as exc:
            if not accepted.done():
                if isinstance(exc, ServerError) and exc.code in ('idempotency_conflict', 'invalid_request'):
                    confirmed = True  # admission rejected; this request owns no receipt
                accepted.set_exception(exc)
            raise
        finally:
            if guard is not None:
                guard.close()
                if confirmed:
                    # HTTP receipt guards are never inherited by model children.
                    # Once terminal (or a duplicate), no late join is possible.
                    # Runtime guards retain their separate revocation evidence.
                    guard.path.unlink(missing_ok=True)

    async def request(self, request_id, owner, name, action, body, key, operation, wait):
        if self.server.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down')
        if not isinstance(name, str) or not 1 <= len(name) <= 128:
            raise ServerError(422, 'invalid_request', 'Server control target must have 1 to 128 characters')
        if key is not None and (not isinstance(key, str) or not 1 <= len(key) <= 128 or
                                any(not 32 <= ord(character) <= 126 for character in key)):
            raise ServerError(422, 'invalid_request', 'Idempotency-Key must contain 1 to 128 printable ASCII characters')
        key = request_id if key is None else key
        identity, fingerprint = self.identity(request_id, owner, name, action, body, key)
        pending = self.pending.get(identity)
        if pending is not None:
            if pending['fingerprint'] != fingerprint:
                raise ServerError(409, 'idempotency_conflict', 'Control key was already used with a different request')
        else:
            if len(self.pending) >= MAX_CONTROL_OPERATIONS:
                raise ServerError(503, 'control_capacity_exceeded', 'Too many Server control requests await completion',
                                  retryable=True, accepted=False)
            accepted = asyncio.get_running_loop().create_future()
            accepted.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            pending = dict(request_id=request_id, key=key, fingerprint=fingerprint, accepted=accepted)
            self.pending[identity] = pending
            task = self.server.own(self.run(accepted, request_id, owner, name, action, body, key, operation))
            task.add_done_callback(lambda done: self.pending.pop(identity, None))
        try:
            record = await self.wait_for(pending['accepted'], DATABASE_WAIT_SECONDS)
        except asyncio.TimeoutError:
            # A timeout cannot prove whether a transaction committed. Keep
            # ownership and coalesce retries until its outcome is confirmed.
            return self.admission_pending(pending)
        deadline = time.monotonic() + wait
        while record['state'] == 'running' and time.monotonic() < deadline:
            try:
                record = await self.lookup(record['request_id'], owner,
                                           timeout=min(DATABASE_WAIT_SECONDS, max(0, deadline - time.monotonic())))
            except ServerError as exc:
                if exc.status_code == 503 and exc.retryable:
                    break  # Acceptance is durable; return its original 202.
                raise
            if record['state'] != 'running':
                break
            await asyncio.sleep(min(.1, max(0, deadline - time.monotonic())))
        response = self.response(record)
        response.headers['Idempotency-Key'] = key
        return response

    async def lookup(self, request_id, owner, *, timeout=None):
        # A query may reconcile an orphaned receipt. Join this transaction on
        # shutdown even if the HTTP reader disconnects while it is in flight.
        if self.server.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down', retryable=True)
        key = (request_id, owner)
        query = self.reads.get(key)
        if query is None:
            if len(self.reads) >= MAX_CONTROL_QUERIES:
                raise self.unavailable(request_id)
            query = self.server.own(self.database(self.get, request_id, owner))
            self.reads[key] = query
            query.add_done_callback(lambda done: self.reads.pop(key, None))
        try:
            return await self.wait_for(query, DATABASE_WAIT_SECONDS if timeout is None else timeout)
        except (asyncio.TimeoutError, *RETRY):
            raise self.unavailable(request_id) from None

    def response(self, record):
        from fastapi.responses import JSONResponse
        request_id, state = record['request_id'], record['state']
        headers = {'Location': self.url(request_id), 'Cache-Control': 'no-store',
                   'Idempotency-Retention-Seconds': str(RETENTION_SECONDS)}
        if state in ('succeeded', 'failed'):
            return JSONResponse(json.loads(record['response']), status_code=record['status_code'], headers=headers)
        if state == 'unconfirmed':
            error = ServerError(503, 'control_unconfirmed',
                'Original control request ended without a confirmed result; inspect current Server state before a new action',
                status_url=self.url(request_id), accepted=True)
            return JSONResponse(dict(request_id=request_id, **error.response()), status_code=503, headers=headers)
        headers['Retry-After'] = '1'
        return JSONResponse(dict(request_id=request_id, status='running', status_url=self.url(request_id)), status_code=202, headers=headers)

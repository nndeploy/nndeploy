"""Durable, metadata-only Server audit records using the existing DB."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import logging
import re
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Table, Column, String, Integer, BigInteger, Double, Boolean, Index, select, insert, update, delete
from starlette.routing import Match

from .db import metadata, encode
from .errors import ServerError


_active = ContextVar('nndeploy_server_audit_operation', default=None)


def audited(function):
    """Record a direct public call; nested calls reuse their existing operation."""
    signature = inspect.signature(function)
    @wraps(function)
    async def call(server, *args, **kwargs):
        audit = server.audit
        current = _active.get()
        if audit.config is None or (current is not None and current[0] is audit and current[1]['active']):
            return await function(server, *args, **kwargs)
        arguments = signature.bind(server, *args, **kwargs).arguments
        return await audit.python_call(function.__name__, arguments, lambda: function(server, *args, **kwargs))
    return call


records = Table('server_audit', metadata,
    Column('id', BigInteger().with_variant(Integer, 'sqlite'), primary_key=True, autoincrement=True),
    Column('scope', String(64), nullable=False), Column('request_id', String(36), nullable=False, unique=True),
    Column('created_at', Double, nullable=False), Column('finished_at', Double), Column('expires_at', Double),
    Column('subject', String(128)), Column('authenticated', Boolean, nullable=False, server_default='0'),
    Column('administrator', Boolean, nullable=False, server_default='0'),
    Column('method', String(16), nullable=False), Column('route', String(512)), Column('entry', String(128)),
    Column('resource_id', String(64)), Column('state', String(16), nullable=False),
    Column('status_code', Integer), Column('duration_seconds', Double),
    Index('ix_server_audit_scope', 'scope', 'id'), Index('ix_server_audit_expiry', 'expires_at', 'id'))


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    retention_seconds: float = Field(default=30 * 86400.0, ge=60, le=365 * 86400)
    max_pending: int = Field(default=64, ge=1, le=4096)
    wait_seconds: float = Field(default=5.0, gt=0, le=30)


class Audit:
    def __init__(self, server, config):
        self.server, self.db = server, server.db
        self.config = AuditConfig.model_validate(config) if config is not None else None
        scope = ['shared', server.redis_config.namespace] if server.redis_config else ['workspace', server.config_scope]
        self.scope = hashlib.sha256(encode(['http-audit', scope]).encode()).hexdigest()
        self.pending = set()
        self.pool = None
        self.closed = False

    @contextmanager
    def track(self):
        # Only a small marker crosses task boundaries, never HTTP headers or
        # request bodies. A detached child cannot reuse a completed operation.
        marker = {'active': True}
        token = _active.set((self, marker))
        try:
            yield marker
        finally:
            marker['active'] = False
            _active.reset(token)

    async def background(self, action, operation):
        if self.config is None:
            return await operation()
        # Periodic work is independent of the caller that started Server,
        # even when its first batch overlaps the startup audit context.
        return await self.python_call(action, {}, operation, method='SYSTEM')

    async def python_call(self, action, arguments, operation, *, method='PYTHON'):
        from .logging_taskid import set_request_id, reset_request_id

        identity = 'req_' + uuid.uuid4().hex
        state = dict(request_id=identity, audit_route='Server.' + action,
                     audit_entry=arguments.get('name'),
                     audit_resource=arguments.get('identity') or arguments.get('task_id'))
        scope = dict(type='python', state=state)
        token = set_request_id(identity)
        started, status, complete = time.perf_counter(), 500, False
        try:
            # Python code is trusted in-process; an owner argument is not proof
            # of a remotely authenticated identity. Record no fabricated actor.
            await self.database(self._begin, identity, method, self.context(scope))
            state['audit_started'] = True
            with self.track():
                result = await operation()
            if action == 'submit':
                state['audit_resource'] = result[0]
            elif isinstance(result, dict):
                state['audit_resource'] = result.get('task_id', state['audit_resource'])
                state['audit_entry'] = result.get('name', state['audit_entry'])
            status = 202 if action == 'submit' or (action == 'deploy_batch' and arguments.get('mode') == 'async') else 200
            complete = True
            return result
        except asyncio.CancelledError:
            status = 499
            raise
        except ServerError as exc:
            status, complete = exc.status_code, True
            state['audit_resource'] = exc.task_id or state['audit_resource']
            raise
        except Exception:
            complete = True
            raise
        finally:
            try:
                await asyncio.shield(self.server.own(self.finish(scope, status, complete, time.perf_counter() - started)))
            finally:
                reset_request_id(token)

    def covers(self, scope):
        return self.config is not None and scope['type'] in ('http', 'websocket') and (
            scope['path'] in ('/api', '/v1') or scope['path'].startswith(('/api/', '/v1/')))

    async def database(self, operation, *args):
        if self.config is None or self.closed:
            raise ServerError(503, 'audit_unavailable', 'Audit storage is unavailable', retryable=True)
        if len(self.pending) >= self.config.max_pending:
            self.server.metrics.audit_errors.labels('capacity').inc()
            raise ServerError(503, 'audit_busy', 'Audit storage is busy', retryable=True)
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='nndeploy-audit-db')
        future = asyncio.get_running_loop().run_in_executor(self.pool, operation, *args)
        self.pending.add(future)
        def complete(done):
            self.pending.discard(done)
            if not done.cancelled():
                done.exception()
        future.add_done_callback(complete)
        try:
            # A disconnected caller must not abandon the SQL transaction or
            # let Server.close dispose its DB while the operation is running.
            return await asyncio.wait_for(asyncio.shield(future), self.config.wait_seconds)
        except Exception:
            self.server.metrics.audit_errors.labels(operation.__name__.lstrip('_')).inc()
            raise ServerError(503, 'audit_unavailable', 'Audit storage is unavailable', retryable=True) from None

    @staticmethod
    def context(scope):
        state = scope.get('state', {})
        caller = state.get('audit_identity')
        route = getattr(scope.get('route'), 'path', state.get('audit_route', '__unmatched__'))
        params = scope.get('path_params') or state.get('audit_params', {})
        entry = params.get('name') or state.get('audit_entry')
        if entry is None and state.get('server_entry'):
            entry = state['server_entry'][0]
        if not isinstance(entry, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', entry):
            entry = None
        resource = state.get('audit_resource') or params.get('identity') or params.get('task_id')
        if not isinstance(resource, str) or not re.fullmatch(r'(?:task|file)_[0-9a-f]{32}', resource):
            resource = None
        # Never accept raw URL/query/header/body/exception fields here.
        return dict(subject=caller.subject if caller is not None else None,
                    authenticated=caller is not None and (caller.subject is not None or caller.administrator),
                    administrator=caller.administrator if caller is not None else False,
                    route=route[:512], entry=entry, resource_id=resource)

    def _begin(self, request_id, method, context=None):
        with self.db.engine.begin() as connection:
            connection.execute(insert(records).values(scope=self.scope, request_id=request_id,
                created_at=self.db.clock(connection), method=method, state='started', **(context or {})))

    async def begin(self, scope):
        # Editor authorization runs before Router dispatch. Resolve only the
        # registered route template here, without executing endpoint code or
        # storing raw paths, so a crash still leaves the intended operation.
        for route in self.server.app.routes:
            match, child = route.matches(scope)
            if match == Match.FULL:
                scope['state']['audit_route'] = getattr(route, 'path', '__unmatched__')
                scope['state']['audit_params'] = child.get('path_params', {})
                break
        if scope['type'] == 'websocket' and 'method' not in scope:
            method = 'WEBSOCKET'
        else:
            method = scope.get('method')
            method = method if method in ('GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS') else 'OTHER'
        await self.database(self._begin, scope['state']['request_id'], method, self.context(scope))
        scope['state']['audit_started'] = True

    def _identify(self, request_id, context):
        with self.db.engine.begin() as connection:
            result = connection.execute(update(records).where(records.c.request_id == request_id,
                records.c.scope == self.scope, records.c.state.in_(('started', 'authorized'))).values(
                    **context, state='authorized'))
            if result.rowcount != 1:
                raise RuntimeError('Audit request is not active')

    async def identify(self, scope, caller):
        if self.config is None:
            return
        state = scope.setdefault('state', {})
        state['audit_identity'] = caller
        if not state.get('audit_started') or state.get('audit_authorized'):
            return
        await self.database(self._identify, state['request_id'], self.context(scope))
        state['audit_authorized'] = True

    def _finish(self, request_id, context, status, complete, duration):
        with self.db.engine.begin() as connection:
            now = self.db.clock(connection)
            connection.execute(update(records).where(records.c.request_id == request_id, records.c.scope == self.scope).values(
                **context, state='complete' if complete else 'interrupted', status_code=status,
                duration_seconds=duration, finished_at=now, expires_at=now + self.config.retention_seconds))

    async def finish(self, scope, status, complete, duration):
        if not scope.get('state', {}).get('audit_started'):
            return
        try:
            await self.database(self._finish, scope['state']['request_id'], self.context(scope), status, complete, duration)
        except ServerError:
            # The operation may already have committed and its response may
            # already be sent. Preserve the durable intent as unconfirmed;
            # never rewrite a successful operation as failed or auto-repeat it.
            logging.getLogger('nndeploy.server.audit').error('Audit outcome remains unconfirmed: %s', scope['state']['request_id'])

    def _read(self, limit, before, request_id):
        conditions = [records.c.scope == self.scope]
        if before is not None:
            conditions.append(records.c.id < before)
        if request_id is not None:
            conditions.append(records.c.request_id == request_id)
        with self.db.engine.connect() as connection:
            rows = list(connection.execute(select(records).where(*conditions).order_by(records.c.id.desc()).limit(limit + 1)).mappings())
        items = [{k: v for k, v in row.items() if k != 'scope'} for row in rows[:limit]]
        return {'items': items, 'next_before': items[-1]['id'] if len(rows) > limit else None}

    async def read(self, limit=100, before=None, request_id=None):
        return await self.database(self._read, limit, before, request_id)

    def _cleanup(self):
        with self.db.engine.begin() as connection:
            now = self.db.clock(connection)
            expired = list(connection.execute(select(records.c.id).where(records.c.scope == self.scope,
                records.c.expires_at <= now).order_by(records.c.expires_at, records.c.id).limit(100)).scalars())
            if not expired:
                return 0
            return connection.execute(delete(records).where(records.c.id.in_(expired), records.c.expires_at <= now)).rowcount

    async def cleanup(self):
        return await self.database(self._cleanup) if self.config is not None else 0

    async def close(self):
        self.closed = True
        if self.pending:
            await asyncio.gather(*list(self.pending), return_exceptions=True)
        if self.pool is not None:
            self.pool.shutdown(wait=True)

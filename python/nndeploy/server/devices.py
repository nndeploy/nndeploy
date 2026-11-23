"""Resident model budgets and request concurrency on declared devices.

These are admission budgets, not GPU allocators or hardware isolation. No
heartbeat, wall-clock expiry, or controller PID can free a resident claim.
"""
import asyncio
import hashlib
import logging
import time
import uuid
from typing import Literal

from pydantic import Field, model_validator
from sqlalchemy import (Table, Column, String, Text, Integer, BigInteger, Double,
                        ForeignKey, Index, Boolean, and_, or_, select, insert, update, delete, func)
from sqlalchemy.exc import IntegrityError, OperationalError, InterfaceError, DisconnectionError, TimeoutError as PoolTimeout
from sqlalchemy.dialects.mysql import VARCHAR

from .config import StrictModel
from .control_guard import ControlGuard
from .db import metadata, encode, tasks, editor_tasks
from .errors import ServerError


class DeviceBudget(StrictModel):
    identity: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_.:-]{1,128}$')
    max_workers: int = Field(ge=1, le=65536)
    memory_bytes: int | None = Field(default=None, ge=1, le=2**63 - 1)
    max_concurrency: int | None = Field(default=None, ge=1, le=65536)
    reserved_slots: dict[str, int] = Field(default_factory=dict, max_length=128)

    @model_validator(mode='after')
    def check_slots(self):
        import re
        if self.reserved_slots and self.max_concurrency is None:
            raise ValueError('reserved_slots requires max_concurrency')
        if (any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name) or type(value) is not int or value < 1
                for name, value in self.reserved_slots.items()) or
            sum(self.reserved_slots.values()) > (self.max_concurrency or 0)):
            raise ValueError('Reserved entry slots must fit within device concurrency')
        return self


class DeviceLimits(StrictModel):
    scope: Literal['server', 'host'] = 'server'
    namespace: str = Field(default='nndeploy', pattern=r'^[A-Za-z0-9_.-]{1,64}$')
    host_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_.:-]{1,128}$')
    devices: dict[str, DeviceBudget] = Field(min_length=1, max_length=64)

    @model_validator(mode='after')
    def check(self):
        import re
        if any(not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', name) for name in self.devices):
            raise ValueError('Device names must be bounded identifiers')
        if self.scope == 'host' and (self.host_id is None or any(value.identity is None for value in self.devices.values())):
            raise ValueError('Host coordination requires a stable host_id and physical identity for every device')
        identities = [value.identity or name for name, value in self.devices.items()]
        if len(set(identities)) != len(identities):
            raise ValueError('Device aliases must identify different physical devices')
        return self


IDENTIFIER = String(128).with_variant(VARCHAR(128, charset='utf8mb4', collation='utf8mb4_bin'), 'mysql')

scopes = Table('server_device_limits', metadata,
    Column('scope', String(64), primary_key=True), Column('policy', Text, nullable=False))
claims = Table('server_device_claims', metadata,
    Column('token', String(32), primary_key=True), Column('device', IDENTIFIER, primary_key=True),
    Column('scope', String(64), ForeignKey('server_device_limits.scope'), nullable=False),
    Column('name', IDENTIFIER, nullable=False), Column('directory', Text, nullable=False),
    Column('workers', Integer, nullable=False), Column('memory_bytes', BigInteger, nullable=False),
    Column('peak_memory_bytes', BigInteger, nullable=False, server_default='0'),
    Column('created_at', Double, nullable=False), Index('ix_server_device_scope', 'scope', 'token'))
runs = Table('server_device_runs', metadata,
    Column('token', String(32), primary_key=True), Column('scope', String(64), nullable=False),
    Column('runtime_token', String(32), nullable=False), Column('name', IDENTIFIER, nullable=False),
    Column('members', Text, nullable=False), Column('running', Boolean, nullable=False, server_default='0'),
    Column('created_at', Double, nullable=False), Column('rank', Double, nullable=False),
    Column('expires_at', Double, nullable=False),
    Index('ix_server_device_wait', 'scope', 'running', 'rank'),
    Index('ix_server_device_runtime', 'runtime_token'))
run_devices = Table('server_device_run_devices', metadata,
    Column('token', String(32), ForeignKey('server_device_runs.token', ondelete='CASCADE'), primary_key=True),
    Column('memory_bytes', BigInteger, nullable=False, server_default='0'),
    Column('device', IDENTIFIER, primary_key=True), Index('ix_server_device_run_device', 'device', 'token'))

RETRY = (OperationalError, InterfaceError, DisconnectionError, PoolTimeout, OSError)


class Devices:
    def __init__(self, server, settings):
        self.server, self.db = server, server.db
        self.settings = DeviceLimits.model_validate(settings)
        group = (['host', self.settings.namespace, self.settings.host_id] if self.settings.scope == 'host'
                 else ['server', server.config_scope])
        self.scope = hashlib.sha256(encode(['resident-devices', group]).encode()).hexdigest()
        self.policy = {budget.identity or alias: dict(max_workers=budget.max_workers, memory_bytes=budget.memory_bytes)
                       for alias, budget in self.settings.devices.items()}
        for alias, budget in self.settings.devices.items():
            if budget.max_concurrency is not None:
                self.policy[budget.identity or alias].update(max_concurrency=budget.max_concurrency,
                                                           reserved_slots=budget.reserved_slots)
        self.cursor = ''

    def requirements(self, config):
        execution = config.execution
        if not execution.devices:
            raise ServerError(422, 'device_declaration_required', 'Configure execution.devices before enabling device budgets')
        requirements = {}
        for alias in execution.devices:
            budget = self.settings.devices.get(alias)
            if budget is None:
                raise ServerError(422, 'device_not_configured', 'Entry declares a device outside the Server budget')
            amount = execution.memory_bytes.get(alias, 0)
            if budget.memory_bytes is not None and amount == 0:
                raise ServerError(422, 'memory_budget_required', 'Declare per-worker memory_bytes for each memory-limited device')
            temporary = execution.temporary_memory_bytes.get(alias, 0)
            if temporary and budget.memory_bytes is None:
                raise ServerError(422, 'memory_limit_required', 'Temporary memory requires a configured device memory budget')
            peak = temporary * (execution.batching.max_size if execution.batching.enabled else 1)
            requirements[budget.identity or alias] = dict(workers=execution.workers, memory_bytes=amount * execution.workers,
                                                          peak_memory_bytes=peak)
        return requirements

    def _usage(self, connection):
        usage = {}
        def device_usage(device):
            return usage.setdefault(device, dict(workers=0, resident=0, peaks={}, calls={}, temporary={}))
        for row in connection.execute(select(claims.c.device, claims.c.name, func.sum(claims.c.workers).label('workers'),
                func.sum(claims.c.memory_bytes).label('resident'), func.max(claims.c.peak_memory_bytes).label('peak'))
                .where(claims.c.scope == self.scope).group_by(claims.c.device, claims.c.name)):
            current = device_usage(row.device)
            current['workers'] += int(row.workers)
            current['resident'] += int(row.resident)
            current['peaks'][row.name] = int(row.peak)
        for row in connection.execute(select(run_devices.c.device, runs.c.name, func.count().label('allocated'),
                func.sum(run_devices.c.memory_bytes).label('temporary')).select_from(run_devices.join(runs))
                .where(runs.c.scope == self.scope, runs.c.running.is_(True)).group_by(run_devices.c.device, runs.c.name)):
            current = device_usage(row.device)
            current['calls'][row.name] = row.allocated
            current['temporary'][row.name] = int(row.temporary)
        return usage

    def _headroom(self, device, peaks):
        reserved = self.policy[device].get('reserved_slots', {})
        return (sum(count * peaks.get(name, 0) for name, count in reserved.items()) +
                max((peak for name, peak in peaks.items() if name not in reserved), default=0))

    def _register(self):
        try:
            with self.db.engine.begin() as connection:
                connection.execute(insert(scopes).values(scope=self.scope, policy=encode(self.policy)))
        except IntegrityError:
            pass

    def _lock(self, connection):
        # A write locks the row on SQLite as well as MySQL/PostgreSQL. Nothing
        # doing model initialization, shutdown, OS locking or I/O runs here.
        connection.execute(update(scopes).where(scopes.c.scope == self.scope).values(scope=self.scope))
        return connection.execute(select(scopes.c.policy).where(scopes.c.scope == self.scope)).scalar_one_or_none()

    def _reserve(self, token, name, requirements):
        requirements = {device: dict(value, peak_memory_bytes=value.get('peak_memory_bytes', 0))
                        for device, value in requirements.items()}
        self._register()
        with self.db.engine.begin() as connection:
            policy = self._lock(connection)
            existing = list(connection.execute(select(claims).where(claims.c.token == token)).mappings())
            if existing:
                if (any(row['scope'] != self.scope or row['name'] != name for row in existing) or
                    {row['device']: dict(workers=row['workers'], memory_bytes=row['memory_bytes'],
                                        peak_memory_bytes=row['peak_memory_bytes']) for row in existing} != requirements):
                    raise ServerError(409, 'device_claim_conflict', 'Resident resource claim changed before confirmation')
                return  # Includes a committed reservation whose acknowledgement was lost.
            usage = self._usage(connection)
            if policy != encode(self.policy):
                if usage:
                    raise ServerError(409, 'device_policy_conflict', 'Participating Servers must use the same device budgets')
                connection.execute(update(scopes).where(scopes.c.scope == self.scope).values(policy=encode(self.policy)))
            for device, required in requirements.items():
                current = usage.get(device, dict(workers=0, resident=0, peaks={}, temporary={}))
                peaks = dict(current['peaks'])
                peaks[name] = max(peaks.get(name, 0), required['peak_memory_bytes'])
                budget = self.policy[device]
                # Loading a resident model must not consume unused reservations
                # while ordinary calls occupy more than their single-call peak.
                protected = sum(max(0, count * peaks.get(other, 0) - current['temporary'].get(other, 0))
                                for other, count in budget.get('reserved_slots', {}).items())
                additional = max(sum(current['temporary'].values()) + protected, self._headroom(device, peaks))
                if (current['workers'] + required['workers'] > budget['max_workers'] or
                    current['resident'] + required['memory_bytes'] + additional > (budget['memory_bytes'] or 2**63 - 1)):
                    raise ServerError(503, 'device_capacity_exceeded', 'Resident model budget is exhausted; existing models remain loaded', retryable=True)
            now = self.db.clock(connection)
            for device, required in requirements.items():
                connection.execute(insert(claims).values(token=token, device=device, scope=self.scope, name=name,
                    directory=str(self.server.data_dir), created_at=now, **required))

    def _release(self, token):
        with self.db.engine.begin() as connection:
            self._lock(connection)
            # All physical Workers have exited. Even a lost per-request release
            # acknowledgement can now be reconciled without guessing task state.
            connection.execute(delete(runs).where(runs.c.scope == self.scope, runs.c.runtime_token == token))
            connection.execute(delete(claims).where(claims.c.scope == self.scope, claims.c.token == token))

    async def _write(self, operation, *args, closing=False):
        while closing or not self.server.shutting_down:
            try:
                return await asyncio.to_thread(operation, *args)
            except RETRY:
                if self.server.shutting_down:
                    break
                logging.warning('Resident device budget awaits database recovery')
                try:
                    await asyncio.wait_for(self.server.maintenance_stop.wait(), .5)
                except asyncio.TimeoutError:
                    pass
        raise ServerError(503, 'device_claim_unconfirmed', 'Resident resource release or reservation is not confirmed', retryable=True)

    async def acquire(self, name, entry):
        requirements = self.requirements(entry['config'])
        await asyncio.to_thread(self.recover)
        token = uuid.uuid4().hex
        guard = ControlGuard.create(self.server.data_dir, token, token)
        entry['device_guard'] = guard
        await self._write(self._reserve, token, name, requirements)

    async def release(self, entry):
        guard = entry.get('device_guard')
        if guard is None:
            return
        # Caller has joined all physical Workers. Closing this controller's
        # guard also permits independent reconciliation if the DB is offline.
        guard.close()
        await self._write(self._release, guard.path.name, closing=True)
        entry.pop('device_guard', None)
        # Retain the revocation inode: a child created just before controller
        # exit must never join a replacement file after recovery.

    def recover(self, limit=100):
        with self.db.engine.connect() as connection:
            query = select(claims.c.token, claims.c.directory).where(claims.c.scope == self.scope).distinct()
            rows = list(connection.execute(query.where(claims.c.token > self.cursor)
                                           .order_by(claims.c.token).limit(limit)))
            if not rows and self.cursor:
                # Wrap in this call: a physical-exit notification must not
                # become an empty recovery pass just because the previous
                # scan reached the last still-live owner.
                rows = list(connection.execute(query.order_by(claims.c.token).limit(limit)))
        self.cursor = rows[-1].token if rows else ''
        recovered = 0
        for row in rows:
            try:
                with ControlGuard.recover(row.directory, row.token, row.token) as exited:
                    if exited:
                        self._release(row.token)
                        recovered += 1
            except Exception:
                # One inaccessible proof or failed transaction must not starve
                # unrelated, verifiably closed models in the same scope.
                logging.warning('Resident device recovery remains unconfirmed')
        return recovered

    def snapshot(self):
        with self.db.engine.connect() as connection:
            usage = self._usage(connection)
            now = self.db.clock(connection)
            activity = list(connection.execute(select(run_devices.c.device, runs.c.running, func.count().label('allocated'))
                .select_from(run_devices.join(runs)).where(runs.c.scope == self.scope,
                    or_(runs.c.running.is_(True), runs.c.expires_at > now)).group_by(run_devices.c.device, runs.c.running)))
        result = {}
        for alias, budget in self.settings.devices.items():
            device = budget.identity or alias
            current = usage.get(device, dict(workers=0, resident=0, peaks={}, temporary={}))
            temporary = sum(current['temporary'].values())
            result[alias] = dict(**self.policy[device], used=dict(workers=current['workers'], memory_bytes=current['resident']))
            if budget.memory_bytes is not None:
                result[alias]['memory'] = dict(resident_bytes=current['resident'], temporary_bytes=temporary,
                    used_bytes=current['resident'] + temporary, available_bytes=budget.memory_bytes - current['resident'] - temporary,
                    required_headroom_bytes=self._headroom(device, current['peaks']))
            if budget.max_concurrency is not None or any(current['peaks'].values()):
                result[alias]['execution'] = {
                    'running': sum(row.allocated for row in activity if row.device == device and row.running),
                    'waiting': sum(row.allocated for row in activity if row.device == device and not row.running)}
        return result

    def execution_devices(self, entry):
        return [self.settings.devices[alias].identity or alias for alias in entry['config'].execution.devices
                if self.settings.devices[alias].max_concurrency is not None or entry['config'].execution.temporary_memory_bytes.get(alias, 0)]

    def _valid_members(self, connection, values, config, now):
        rows = {row['task_id']: row for row in connection.execute(select(
            tasks.c.task_id, tasks.c.state, tasks.c.attempt, tasks.c.worker_token, tasks.c.cancel_requested,
            tasks.c.expires_at, tasks.c.created_at).where(tasks.c.task_id.in_([item['_task_id'] for item in values]))).mappings()}
        remaining, failures, members = [], [], []
        for item in values:
            identity = item['_task_id']
            row = rows.get(identity)
            if (row is None or row['attempt'] != item.get('_attempt', 1) or
                row['worker_token'] != item.get('_worker_token') or row['state'] not in ('pending', 'dispatched', 'running')):
                failures.append(ServerError(409, 'execution_ownership_lost', 'Task no longer belongs to this execution', task_id=identity))
            elif row['cancel_requested']:
                failures.append(ServerError(409, 'task_cancelled', 'Task was cancelled while waiting for a device', task_id=identity))
            else:
                deadline = item.get('_queue_deadline')
                seconds = (deadline - time.monotonic() if deadline is not None else
                           (row['expires_at'] or row['created_at'] + config.execution.queue_timeout_seconds) - now)
                if seconds <= 0:
                    failures.append(ServerError(504, 'queue_timeout', 'Task expired while waiting for a device', task_id=identity))
                else:
                    remaining.append(seconds)
                    members.append(item)
        if not remaining:
            raise failures[0] if failures else ServerError(422, 'invalid_request', 'Device execution requires an existing task')
        return max(remaining), members

    def _begin(self, token, entry, values, devices, *, startup_deadline=None, editor=None):
        runtime = entry['device_guard'].path.name
        name = entry['config'].name
        members = encode({'editor': editor} if editor is not None else {'startup': True} if startup_deadline is not None else
                         [{key: value[key] for key in ('_task_id', '_attempt', '_worker_token') if key in value} for value in values])
        active_values = [{}]  # Warmup/editor runs reserve the configured ceiling.
        with self.db.engine.begin() as connection:
            if self._lock(connection) != encode(self.policy):
                raise ServerError(409, 'device_policy_conflict', 'Device execution policy changed')
            held = set(connection.execute(select(claims.c.device).where(claims.c.scope == self.scope, claims.c.token == runtime)).scalars())
            if not set(devices) <= held:
                raise ServerError(409, 'device_claim_lost', 'Resident device ownership is no longer available')
            current = connection.execute(select(runs).where(runs.c.token == token)).mappings().first()
            if current is not None:
                if current['runtime_token'] != runtime or current['members'] != members or current['scope'] != self.scope:
                    raise ServerError(409, 'device_claim_conflict', 'Execution ownership changed')
                if current['running']:
                    return True  # Commit acknowledgement may have been lost; do not reserve again.
            now = self.db.clock(connection)
            if editor is not None:
                row = connection.execute(select(editor_tasks).where(
                    editor_tasks.c.scope == editor['scope'], editor_tasks.c.task_id == editor['task_id'],
                    editor_tasks.c.token == editor['token'])).mappings().first()
                if row is None or row['state'] not in ('dispatched', 'running'):
                    raise ServerError(409, 'execution_ownership_lost', 'Editor task no longer belongs to this Worker')
                if row['cancel_requested']:
                    raise ServerError(409, 'task_cancelled', 'Editor task was cancelled while waiting for a device')
                remaining, units = row['created_at'] + entry['config'].execution.queue_timeout_seconds - now, 1
                if remaining <= 0:
                    raise ServerError(504, 'queue_timeout', 'Editor task expired while waiting for a device')
            elif startup_deadline is not None:
                remaining, units = startup_deadline - time.monotonic(), 1
                if remaining <= 0:
                    raise ServerError(504, 'startup_timeout', 'Worker expired while waiting for a device')
            else:
                remaining, active_values = self._valid_members(connection, values, entry['config'], now)
            if current is None:
                # One priority point advances one second. A waiting request
                # eventually precedes all newly arriving requests at any priority.
                rank = now - entry['config'].execution.priority
                current = dict(token=token, scope=self.scope, runtime_token=runtime, name=name, members=members,
                               running=False, created_at=now, rank=rank, expires_at=now + remaining)
                connection.execute(insert(runs).values(**current))
                for device in devices:
                    connection.execute(insert(run_devices).values(token=token, device=device))
            elif current['expires_at'] <= now:
                raise ServerError(504, 'startup_timeout' if startup_deadline is not None else 'queue_timeout', 'Device wait expired')
            usage = self._usage(connection)
            needed = {self.settings.devices[alias].identity or alias:
                      sum((item.get('_service') or {}).get('temporary_memory_bytes', {}).get(alias, amount)
                          for item in active_values)
                      for alias, amount in entry['config'].execution.temporary_memory_bytes.items()}
            for device in devices:
                policy, current_usage = self.policy[device], usage[device]
                active = current_usage['calls']
                reserved = policy.get('reserved_slots', {})
                protected = sum(max(0, amount - active.get(other, 0)) for other, amount in reserved.items() if other != name)
                if policy.get('max_concurrency') is not None and sum(active.values()) + protected >= policy['max_concurrency']:
                    return False
                if policy['memory_bytes'] is not None:
                    protected_memory = sum(max(0, count * current_usage['peaks'].get(other, 0) - current_usage['temporary'].get(other, 0))
                                           for other, count in reserved.items() if other != name)
                    if (current_usage['resident'] + sum(current_usage['temporary'].values()) + protected_memory +
                        needed.get(device, 0) > policy['memory_bytes']):
                        return False
                earlier = [runs.c.scope == self.scope, runs.c.running.is_(False), runs.c.expires_at > now,
                           run_devices.c.device == device,
                           or_(runs.c.rank < current['rank'], and_(runs.c.rank == current['rank'], runs.c.token < token))]
                # An entry may use its own reserved share even while earlier
                # requests from another entry are waiting for general capacity.
                if active.get(name, 0) < reserved.get(name, 0):
                    earlier.append(runs.c.name == name)
                if connection.execute(select(runs.c.token).select_from(runs.join(run_devices))
                                      .where(*earlier).limit(1)).first() is not None:
                    return False
            for device in devices:
                connection.execute(update(run_devices).where(run_devices.c.token == token, run_devices.c.device == device)
                                   .values(memory_bytes=needed.get(device, 0)))
            connection.execute(update(runs).where(runs.c.token == token).values(running=True))
            return True

    def _end(self, token):
        with self.db.engine.begin() as connection:
            self._lock(connection)
            connection.execute(delete(runs).where(runs.c.scope == self.scope, runs.c.token == token))

    async def _end_later(self, token):
        try:
            await self._write(self._end, token, closing=True)
        except Exception:
            # The resident guard remains the fallback proof. Closing all of
            # its Workers or recovering their exit removes any remaining runs.
            logging.warning('Device execution release remains unconfirmed')

    async def _run(self, token, entry, values, devices, operation, *, startup_deadline=None):
        invoked, uncertain = False, False
        try:
            # Bound DB failures by the same queue deadline; no computation is
            # allowed until the durable slot reservation is acknowledged.
            local_deadline = max((value.get('_queue_deadline', float('inf')) for value in values), default=float('inf'))
            fallback_deadline = time.monotonic() + entry['config'].execution.queue_timeout_seconds
            if startup_deadline is not None:
                local_deadline = fallback_deadline = startup_deadline
            with self.server.metrics.measure('device_startup_wait' if startup_deadline is not None else 'device_wait', entry['config'].name):
                while True:
                    try:
                        options = {} if startup_deadline is None else {'startup_deadline': startup_deadline}
                        if await asyncio.to_thread(self._begin, token, entry, values, devices, **options):
                            break
                    except RETRY:
                        if self.server.shutting_down or time.monotonic() >= min(local_deadline, fallback_deadline):
                            raise ServerError(503, 'device_coordination_unavailable', 'Could not confirm device execution ownership')
                    await asyncio.sleep(.05)
            invoked = True
            return await operation()
        except BaseException as exc:
            # A cancelled coroutine is not proof that a native thread stopped.
            # Keep its slot until the model's physical shutdown can be proved.
            # Startup failure may include failed cleanup. The entry's physical
            # Worker shutdown, not the exception, releases that reservation.
            uncertain = invoked and (startup_deadline is not None or isinstance(exc, asyncio.CancelledError))
            raise
        finally:
            if not uncertain:
                try:
                    await asyncio.to_thread(self._end, token)
                except Exception:
                    self.server.own(self._end_later(token))

    async def run(self, entry, values, operation):
        devices = self.execution_devices(entry)
        if not devices:
            return await operation()
        # Caller/HTTP cancellation must not cancel native execution or its slot.
        return await asyncio.shield(self.server.own(self._run(uuid.uuid4().hex, entry, values, devices, operation)))

    async def initialize(self, entry, worker):
        devices = self.execution_devices(entry)
        if not devices:
            return await worker.start()
        deadline = time.monotonic() + entry['config'].execution.startup_timeout_seconds
        async def start():
            if time.monotonic() >= deadline:
                raise ServerError(504, 'startup_timeout', 'Worker expired before initialization')
            return await worker.start(deadline=deadline)
        return await asyncio.shield(self.server.own(self._run(uuid.uuid4().hex, entry, [], devices, start,
                                                             startup_deadline=deadline)))


class EditorDeviceRun:
    """Use the same budgets for a one-shot editor Worker process.

    The Worker exits after this Run. Keep its physical guard until OS exit,
    including failed initialization, deinit and native result serialization.
    Existing device recovery releases both claims and execution slots only
    after that proof; a successful Task result alone cannot release them.
    """
    def __init__(self, payload, options):
        from pathlib import Path
        from types import SimpleNamespace
        from .config import ServerConfig
        from .db import DB
        from .control_guard import _boot_id
        self.db = None
        if not _boot_id():
            raise ServerError(422, 'editor_device_platform_unsupported',
                              'Editor Run device budgets require verifiable local process-exit evidence')
        if not options or not options.get('devices') or not payload.get('_editor_execution'):
            raise ServerError(422, 'editor_device_config_required',
                              'Editor Run has no confirmed device configuration; submit it again with workflow_id')
        settings = options['devices']
        config = ServerConfig.model_validate(payload['_editor_execution'])
        self.db = DB(options['database_url'], timeout_seconds=options['timeout_seconds'])
        server = SimpleNamespace(db=self.db, config_scope=settings['config_scope'], data_dir=Path(settings['data_dir']))
        self.devices = Devices(server, settings['settings'])
        self.entry = dict(config=config)
        self.editor = dict(scope=payload['_editor_scope'], task_id=payload['id'], token=payload['_editor_token'])

    def acquire(self, cancelled):
        requirements = self.devices.requirements(self.entry['config'])
        self.devices.recover()
        token = uuid.uuid4().hex
        guard = ControlGuard.create(self.devices.server.data_dir, token, token)
        self.entry['device_guard'] = guard
        deadline = time.monotonic() + self.entry['config'].execution.queue_timeout_seconds
        execution_token = uuid.uuid4().hex
        reserved = False
        while True:
            if cancelled():
                raise ServerError(409, 'task_cancelled', 'Editor task was cancelled before device execution')
            if time.monotonic() >= deadline:
                raise ServerError(504, 'queue_timeout', 'Editor task expired before device execution')
            try:
                if not reserved:
                    self.devices._reserve(token, self.entry['config'].name, requirements)
                    reserved = True
                # Validate ownership/deadline even when no per-device
                # concurrency or temporary memory limit was configured.
                if self.devices._begin(execution_token, self.entry, [],
                                       self.devices.execution_devices(self.entry), editor=self.editor):
                    return
            except RETRY:
                pass  # Same tokens retry a possibly committed reservation.
            time.sleep(.05)

    def close(self):
        if self.db is not None:
            self.db.close()

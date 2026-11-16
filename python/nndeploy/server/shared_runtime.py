"""Connect shared configuration transactions to the existing Server lifecycle."""
import asyncio
import copy
import hashlib
import logging
import uuid

from sqlalchemy.exc import OperationalError, InterfaceError, DisconnectionError, TimeoutError as PoolTimeout

from .config import ServerConfig
from .db import encode
from .errors import ServerError
from .redis import entry_fingerprint
from .task_queue import RedisTaskQueue


class SharedRuntime:
    def __init__(self, server):
        self.server = server
        self.control = server.db.shared
        self.runtime_cursors = {}

    async def write(self, operation, *args, **kwargs):
        """Resolve an uncertain commit with the same operation identity."""
        while not self.server.shutting_down:
            try:
                return await asyncio.to_thread(operation, *args, **kwargs)
            except (OperationalError, InterfaceError, DisconnectionError, PoolTimeout, OSError):
                logging.warning('Shared configuration write awaits database recovery')
                try:
                    await asyncio.wait_for(self.server.maintenance_stop.wait(), .5)
                except asyncio.TimeoutError:
                    pass
        raise ServerError(503, 'shared_control_unconfirmed', 'Shared configuration write is not confirmed; inspect shared state after recovery', retryable=True)

    @staticmethod
    def kind(source):
        return next(key for key in ('workflow', 'graph_factory', 'handler_factory', 'deploy') if source.get(key) is not None)

    async def snapshot(self, entry):
        source = entry['source']
        if source['workflow'] is not None:
            from .resources import prepare_workflow
            source['workflow'] = await asyncio.to_thread(prepare_workflow, source['workflow'], entry['config'],
                self.server.data_dir, source['base_dir'], lease=await self.server._assets_for(entry))
        return dict(kind=self.kind(source), identity=source['identity'], workflow=source['workflow'],
                    config=entry['config'].model_dump(), autodetect=source['autodetect'], untyped=entry['untyped'],
                    schemas=[source[key].model_json_schema() if source.get(key) is not None else None
                             for key in ('input_model', 'output_model')])

    def apply(self, entry, state):
        """Only reuse already registered code; DB data never imports a factory."""
        data, source = state['data'], entry['source']
        schemas = [source[key].model_json_schema() if source.get(key) is not None else None
                   for key in ('input_model', 'output_model')]
        if data['kind'] != self.kind(source) or data['schemas'] != schemas or (
                data['kind'] != 'workflow' and data['identity'] != source['identity']):
            raise ServerError(409, 'shared_configuration_changed', 'Active configuration requires different registered code or schemas')
        config = ServerConfig.model_validate(data['config'])
        # HTTP connection policy belongs to each API replica, not its workers.
        config.execution.http_wait_timeout_seconds = entry['config'].execution.http_wait_timeout_seconds
        source.update(identity=data['identity'], workflow=copy.deepcopy(data['workflow']),
                      autodetect=data['autodetect'], untyped=data['untyped'])
        entry.update(config=config, schema=config.public_schema(), untyped=data['untyped'], shared_key=state['entry_key'])

    def queue(self, name, entry, *, key=None, salt=''):
        value = RedisTaskQueue(self.server.db, self.server.redis_config, name,
            entry_fingerprint(entry['source'], entry['config']) + salt, entry['config'].execution.capacity, queue_key=key)
        self.server.metrics.attach_redis(value.client, name)
        return value

    async def attach_workflow(self, name):
        """An explicit editor Update can control a persisted entry without loading it first."""
        group = hashlib.sha256(encode([self.server.redis_config.namespace, 'group']).encode()).hexdigest()
        try:
            state = await asyncio.to_thread(self.control.get, self.control.entry_key(group, name))
        except ServerError as exc:
            if exc.code == 'entry_not_found':
                return False
            raise
        if state['data']['kind'] != 'workflow':
            raise ServerError(409, 'shared_configuration_changed', 'This shared name belongs to registered code')
        previous = self.server.entries.get(name)
        if previous and (previous['workers'] or previous.get('shared_holders')):
            raise ServerError(409, 'entry_busy', 'Previous workers have not confirmed release', retryable=True)
        if previous and previous.get('queue'):
            await previous['queue'].close()
        self.server.entries.pop(name, None)
        if previous is not None:
            self.server._release_assets(previous)
        self.server.add(name=name, workflow=state['data']['workflow'], config=state['data']['config'])
        entry = self.server.entries[name]
        self.apply(entry, state)
        await self.prepare(name, entry, resume=False)
        return True

    async def prepare(self, name, entry, *, loaded=False, resume=True):
        if entry.get('shared_candidate') or entry.get('shared_draining'):
            return
        previous = entry.get('queue')
        if previous is not None:
            await previous.close()
        before = entry_fingerprint(entry['source'], entry['config'])
        entry['queue'] = self.queue(name, entry)
        await entry['queue'].start()
        data = await self.snapshot(entry)
        async with self.server._publishing_assets(entry):
            state = await self.write(self.control.register, entry['queue'].group_key, name, entry['queue'].key,
                before, data, count=entry['config'].execution.workers)
        if state['pending_token']:
            await asyncio.to_thread(self.control.recover_pending, state['entry_key'], self.server.data_dir)
            state = await asyncio.to_thread(self.control.get, state['entry_key'])
        if before != state['fingerprint'] and state['load_failed'] and not state['initialized']:
            entry['shared_key'] = state['entry_key']
            if not state['accepting'] and not resume:
                raise ServerError(409, 'shared_control_changed', 'Initial entry is stopped; explicitly Start to repair it', retryable=True)
            token = uuid.uuid4().hex
            await entry['queue'].close()
            entry['queue'] = self.queue(name, entry, salt=token)
            await entry['queue'].start()
            entry['shared_candidate'] = token
            entry['shared_start_repair'] = True
            await self.guard(entry, token)
            await self.write(self.control.begin_reload, state['entry_key'], state['control_token'], entry['queue'].key,
                before, await self.snapshot(entry), count=entry['config'].execution.workers, write_token=token,
                repair=True, expected_queue_token=state['queue_control_token'], pending_holder=entry['file_holder'])
            return
        self.apply(entry, state)
        if entry['queue'].key != state['active_queue']:
            await entry['queue'].close()
            entry['queue'] = self.queue(name, entry, key=state['active_queue'])
            await entry['queue'].start()
        if loaded and before != state['fingerprint']:
            raise ServerError(409, 'shared_control_changed', 'Factory schema was discovered; retry Start with the active configuration', retryable=True)

    async def guard(self, entry, token):
        from .control_guard import ControlGuard
        entry.setdefault('file_holder', uuid.uuid4().hex)
        entry['control_guard'] = ControlGuard.create(self.server.data_dir, token, entry['file_holder'])
        group = hashlib.sha256(encode([self.server.redis_config.namespace, 'group']).encode()).hexdigest()
        key = self.control.entry_key(group, entry['config'].name)
        await self.write(self.control.bind_runtime, key, token, entry['file_holder'])

    async def hold(self, entry, *, ready=False, holder=None):
        if not entry.get('shared_key'):
            return None
        if holder is None:
            holder = uuid.uuid4().hex
            entry.setdefault('shared_holders', []).append(holder)
        if entry.get('control_guard'):
            await self.write(self.control.bind_runtime, entry['shared_key'], entry['control_guard'].path.name,
                             entry['file_holder'], entry['queue'].key)
        await self.write(self.control.hold_worker, entry['shared_key'], entry['queue'].key, holder,
                         token=entry.get('shared_candidate'), ready=ready, draining=entry.get('shared_draining', False),
                         owner_token=entry['control_guard'].path.name if entry.get('control_guard') else None,
                         runtime_holder=entry['file_holder'] if entry.get('control_guard') else None)
        return holder

    async def release(self, entry):
        # Called only after all local Worker.close() calls confirm physical exit.
        for holder in list(entry.get('shared_holders', [])):
            await asyncio.to_thread(self.server.db.release_files, holder)
            entry['shared_holders'].remove(holder)

    async def started(self, entry):
        if entry.get('shared_start_repair'):
            await self.write(self.control.activate, entry['shared_key'], entry['shared_candidate'])
            entry.pop('shared_candidate', None)
            entry.pop('shared_start_repair', None)
            await self.cleanup(entry)
        elif not entry.get('shared_candidate') and not entry.get('shared_draining'):
            await self.write(self.control.confirm_ready, entry['shared_key'], entry['queue'].key, entry['shared_holders'])

    async def failed(self, entry):
        if not entry.get('shared_key') or entry.get('shared_holders'):
            return
        if entry.get('shared_start_repair'):
            try:
                await asyncio.to_thread(self.control.discard, entry['shared_key'], entry['queue'].key,
                                        token=entry['shared_candidate'])
            finally:
                entry.pop('shared_candidate', None)
                entry.pop('shared_start_repair', None)
        else:
            await asyncio.to_thread(self.control.initialization_failed, entry['shared_key'], entry['queue'].key)

    async def current(self, entry):
        return await asyncio.to_thread(self.control.get, entry['shared_key'])

    async def status(self, entry):
        return await asyncio.to_thread(self.control.status, entry['shared_key'])

    async def check(self, entry):
        state = await self.status(entry)
        if entry['queue'].key != state['active_queue']:
            raise ServerError(409, 'entry_changed', 'Shared configuration changed; this Server is loading the active configuration', retryable=True)
        return state

    @staticmethod
    def replacement(old):
        candidate = dict(old, source=dict(old['source']), workers=[], consumers=[], futures={}, admitted=set(),
                         status='stopped', error=None, closing=None, shared_holders=[])
        for key in ('queue', 'maintenance', 'wake', 'file_holder', 'control_guard', 'device_guard', 'release_pending', 'shared_candidate', 'shared_start_repair', 'shared_draining'):
            candidate.pop(key, None)
        return candidate

    async def cleanup(self, entry):
        # Shutdown must join DB reconciliation before disposing dependencies.
        # Cancelling the polling task does not cancel a thread doing recovery.
        await asyncio.shield(self.server.own(self._cleanup(entry)))

    async def _cleanup(self, entry):
        await asyncio.to_thread(self.control.recover_pending, entry['shared_key'], self.server.data_dir)
        key = entry['shared_key']
        runtimes = await asyncio.to_thread(self.control.runtimes, key, self.runtime_cursors.get(key, ''))
        self.runtime_cursors[key] = runtimes[-1]['token'] if runtimes else ''
        for runtime in runtimes:
            try:
                await asyncio.to_thread(self.control.recover_runtime, key, runtime, self.server.data_dir)
            except Exception:
                logging.exception('Shared runtime recovery remains unconfirmed')
        for key in await asyncio.to_thread(self.control.draining, entry['shared_key']):
            if await self.has_results(key):
                continue
            try:
                await asyncio.to_thread(self.control.discard, entry['shared_key'], key)
            except ServerError as exc:
                if exc.code != 'entry_busy':
                    raise

    async def has_results(self, key):
        return await asyncio.to_thread(lambda: next((self.server.data_dir / 'queue-results' / key).glob('*.json'), None) is not None)

    async def recoverable(self, entry):
        result_queues = []
        for key in await asyncio.to_thread(self.control.draining, entry['shared_key']):
            if await self.has_results(key):
                result_queues.append(key)
        return await asyncio.to_thread(self.control.recoverable, entry['shared_key'], result_queues)

    async def switch_local(self, name, old, candidate):
        server = self.server
        server.entries[name] = candidate
        candidate.pop('shared_candidate', None)
        candidate['status'] = 'reloading' if candidate['workers'] else 'stopped'
        old['status'] = 'ready' if old['workers'] else 'stopped'
        server.retiring_entries.setdefault(name, []).append(old)
        try:
            await server._persist_entry(name, candidate, bool(candidate['workers']))
            await server._stop_entry(name, old, persist=False, shared_scope=False, drain_shared=True)
            if old.get('queue') is not None:
                await old['queue'].close()
            server.retiring_entries[name].remove(old)
            server._release_assets(old)
            await self.cleanup(candidate)
        except BaseException:
            candidate['error'] = 'Previous worker release or shared configuration cleanup was not confirmed'
            raise
        finally:
            candidate['status'] = 'ready' if candidate['workers'] else 'stopped'

    async def reject_candidate(self, name, old, candidate):
        # A committed-but-unacknowledged switch may already have accepted work.
        # Closing only local workers is safe in either state; never stop its queue.
        try:
            if candidate['workers']:
                candidate['status'] = 'ready'
                await self.server._stop_entry(name, candidate, persist=False, shared_scope=False)
            if candidate.get('queue') is not None:
                await candidate['queue'].close()
        finally:
            old['status'] = 'ready' if old['workers'] else 'stopped'
            if candidate['workers'] or candidate.get('shared_holders'):
                self.server.retiring_entries.setdefault(name, []).append(candidate)
            elif candidate.get('control_guard') or candidate.get('release_pending'):
                try:
                    await self.server._release_entry_files(candidate)
                except BaseException:
                    # Physical Workers can be gone while their database
                    # cleanup is still unconfirmed. Keep the original holder
                    # and token reachable through the existing Stop path.
                    self.server.retiring_entries.setdefault(name, []).append(candidate)
                    raise
            self.server._release_assets(candidate)

    async def reload(self, name, old, candidate):
        state = await self.check(old)
        await self.cleanup(old)
        token = uuid.uuid4().hex
        candidate = self.replacement(candidate)
        candidate['shared_candidate'] = token
        candidate['shared_key'] = old['shared_key']
        old['status'] = 'reloading'
        begun = False
        try:
            await self.guard(candidate, token)
            data = await self.snapshot(candidate)
            candidate['queue'] = self.queue(name, candidate, salt=token)
            await candidate['queue'].start()
            async with self.server._publishing_assets(candidate):
                await self.write(self.control.begin_reload, old['shared_key'], state['control_token'], candidate['queue'].key,
                    entry_fingerprint(candidate['source'], candidate['config']), data,
                    count=candidate['config'].execution.workers, write_token=token, expected_queue_token=state['queue_control_token'],
                    pending_holder=candidate['file_holder'])
            begun = True
            await self.server._start_entry(name, candidate, persist=False, resume=False)
            await self.write(self.control.activate, old['shared_key'], token)
        except BaseException:
            await self.reject_candidate(name, old, candidate)
            if begun and not candidate['workers'] and not candidate.get('shared_holders'):
                try:
                    await asyncio.to_thread(self.control.discard, old['shared_key'], candidate['queue'].key, token=token)
                except Exception:
                    logging.warning('Shared candidate state requires reconciliation')
            raise
        await self.switch_local(name, old, candidate)
        return self.server.describe(name)

    async def follow(self, name, old, state):
        if self.server.entries.get(name) is not old:
            return
        candidate = self.replacement(old)
        self.apply(candidate, state)
        old['status'] = 'reloading' if old['workers'] else 'stopped'
        try:
            await self.server._start_entry(name, candidate, persist=False, resume=False)
        except BaseException:
            await self.reject_candidate(name, old, candidate)
            raise
        await self.switch_local(name, old, candidate)

    async def recover(self, name, current, states):
        """Recreate old workers solely to finish already accepted queued tasks."""
        if self.server.entries.get(name) is not current:
            return
        for state in states:
            entry = self.replacement(current)
            self.apply(entry, state)
            entry['shared_draining'] = True
            entry['queue'] = self.queue(name, entry, key=state['active_queue'])
            self.server.retiring_entries.setdefault(name, []).append(entry)
            try:
                await entry['queue'].start()
                await self.server._reconcile_shared_results(entry)
                if await asyncio.to_thread(self.server.db.pending_tasks, entry['queue'].key, 1):
                    await self.server._start_entry(name, entry, persist=False, resume=False)
                    await self.server._stop_entry(name, entry, persist=False, shared_scope=False, drain_shared=True)
                    await self.server._reconcile_shared_results(entry)
            except Exception:
                # Another replica can finish and retire this queue between our
                # snapshot and holder acquisition. There is then nothing left
                # to recover, provided our own physical cleanup is confirmed.
                if (entry['workers'] or entry.get('shared_holders')
                        or await self.has_results(entry['queue'].key)
                        or await asyncio.to_thread(self.server.db.pending_tasks, entry['queue'].key, 1)):
                    raise
            finally:
                if not entry['workers'] and not entry.get('shared_holders'):
                    await entry['queue'].close()
                    self.server.retiring_entries[name].remove(entry)
                    self.server._release_assets(entry)
        await self.cleanup(current)
        if current['error'] == 'Old queued tasks could not be resumed':
            current['error'] = None

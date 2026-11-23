"""Entry start, replacement and drain; Server retains transport/task orchestration."""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .entry import Entry, replacement
from .task_queue import TaskQueue

if TYPE_CHECKING:
    from .server import Server


class EntryLifecycle:
    def __init__(self, server: Server):
        self.server = server

    async def start(self, name: str, entry: Entry, *, persist=True, resume=True):
        server = self.server
        from .worker import Worker, ProcessWorker, EngineWorker, ConcurrentWorker
        from .errors import ServerError
        if server.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down')
        if entry['status'] == 'ready':
            if entry.get('consumer_failed'):
                raise ServerError(503, 'consumer_failed', 'Stop the entry before restarting its failed consumer')
            if server.redis_config is not None and resume:
                await server.shared_runtime.check(entry)
                queue = entry['queue']
                usage = await asyncio.to_thread(server.db.queue_usage, queue.key)
                if not usage['accepting']:
                    await server._write_shared_control(entry, True, usage['control_token'])
            return server.describe(name)
        if entry['status'] == 'failed' and entry['workers']:
            raise ServerError(409, 'entry_busy', 'Previous worker release was not confirmed')
        if entry['status'] == 'failed' and entry.get('release_pending'):
            # Failed initialization may have closed every Worker while its
            # resource transaction failed. Preserve the old holder until that
            # cleanup succeeds; a new load must not orphan the old references.
            await server._release_entry_files(entry)
        if entry['status'] == 'failed' and entry.get('device_guard') and server.devices is not None:
            await server.devices.release(entry)
        if entry['status'] in ('starting', 'stopping', 'reloading'):
            raise ServerError(409, 'entry_busy', 'Entry is changing state', retryable=True)
        entry['status'], entry['error'] = 'starting', None
        config = entry['config']
        initial_config = config
        directory = server.data_dir / 'tmp' / name
        directory.mkdir(parents=True, exist_ok=True)
        entry['workers'] = []
        entry['consumers'], entry['consumer_runs'] = [], []
        entry['consumer_failed'] = False
        if not entry.get('control_guard'):
            entry['file_holder'] = uuid.uuid4().hex
        entry['shared_holders'] = []
        candidate = bool(entry.get('shared_candidate') or entry.get('shared_draining'))
        queue_prepared = False
        control_token = None
        prepared_fingerprint = None
        try:
            if (entry['source']['workflow'] is not None or entry['source'].get('graph_factory') is not None) and 'saved_token' not in entry:
                saved = await asyncio.to_thread(server.db.get_server_config, server.config_scope, name)
                entry['saved_token'] = saved['write_token'] if saved else None
                if saved and saved['data'].get('kind') == 'graph_factory' and entry['source'].get('graph_factory') is not None:
                    entry['source']['resource_fallback'] = saved['data']['resources']
            if server.redis_config is not None and (not entry['source']['autodetect'] or entry.get('queue') is not None):
                # A known stopped shared configuration must not initialize or
                # execute warmup just because another API process restarted.
                await server._prepare_shared_queue(name, entry, resume=resume)
                candidate = bool(entry.get('shared_candidate') or entry.get('shared_draining'))
                config = initial_config = entry['config']
                queue_prepared = True
                usage = await asyncio.to_thread(server.db.queue_usage, entry['queue'].key)
                control_token = usage['control_token']
                from .redis import entry_fingerprint
                prepared_fingerprint = entry_fingerprint(entry['source'], config)
                if not candidate and not resume and not usage['accepting']:
                    entry['status'] = 'stopped'
                    return server.describe(name)
                if not candidate and resume and not usage['accepting'] and usage['inflight']:
                    raise ServerError(409, 'entry_busy', 'Shared entry is still draining', retryable=True)
            if server.shared_runtime is not None and not entry.get('control_guard'):
                await server.shared_runtime.guard(entry, uuid.uuid4().hex)
            if server.devices is not None:
                await server.devices.acquire(name, entry)
            elif config.execution.devices:
                raise ServerError(422, 'device_limits_required', 'Configure Server device_limits to enforce declared resident budgets')
            await asyncio.to_thread(server.db.hold_files, entry['file_holder'],
                                    [entry['source']['workflow'], config.model_dump()])
            if entry['source']['workflow'] is not None:
                from .resources import prepare_workflow
                entry['source']['workflow'] = await asyncio.to_thread(prepare_workflow,
                    entry['source']['workflow'], config, server.data_dir, entry['source']['base_dir'],
                    lease=await server._assets_for(entry))
            for _ in range(config.execution.workers):
                worker_class = {'engine': EngineWorker, 'concurrent': ConcurrentWorker}.get(
                    config.execution.scheduling, ProcessWorker if config.execution.isolation == 'process' else Worker)
                if config.execution.batching.enabled:
                    from .batching import BatchWorker
                    worker_class = BatchWorker
                # Existing editor plugins are trusted installation code. Each
                # isolated Worker must register them before loading the Graph.
                # Derive this locally; it is not a workflow field or HTTP input.
                plugin_root = Path(server.args.resources) if getattr(server, 'args', None) is not None else server.data_dir
                worker_source = dict(entry['source'], plugin_dir=str(plugin_root / 'plugin'), resource_dir=str(server.data_dir),
                                     resource_holder=entry['file_holder'])
                worker = worker_class(worker_source, config, server.storage, directory)
                if entry.get('control_guard'):
                    worker.control_guard_path = str(entry['control_guard'].path)
                if entry.get('device_guard'):
                    worker.device_guard_path = str(entry['device_guard'].path)
                    worker.device_run = lambda values, operation, entry=entry: server.devices.run(entry, values, operation)
                    if config.execution.temporary_memory_bytes:
                        worker.device_prepare = worker.device_run
                entry['workers'].append(worker)
                holder = await server.shared_runtime.hold(entry) if server.shared_runtime is not None else None
                worker.shared_holder = holder
                with server.metrics.measure('worker_initialize', name):
                    if server.devices is not None:
                        await server.devices.initialize(entry, worker)
                    else:
                        await worker.start()
                if getattr(worker, 'asset_paths', None):
                    lease = await server._assets_for(entry)
                    await asyncio.to_thread(lease.hold_values, worker.asset_paths)
                if entry['source'].get('graph_factory') is not None:
                    entry['source']['resource_snapshot'] = worker.resource_snapshot
                if holder is not None:
                    await server.shared_runtime.hold(entry, holder=holder, ready=True)
                if entry['source']['autodetect']:
                    config = worker.config
                    entry['config'], entry['schema'] = config, config.public_schema()
            await asyncio.to_thread(server.db.hold_files, entry['file_holder'], config.model_dump())
            if server.redis_config is not None:
                from .redis import entry_fingerprint
                if queue_prepared and prepared_fingerprint != entry_fingerprint(entry['source'], config):
                    entry['config'], entry['schema'] = initial_config, initial_config.public_schema()
                    raise ServerError(409, 'shared_configuration_changed', 'Discovered workflow configuration changed; register its explicit configuration before starting')
                if not queue_prepared:
                    await server._prepare_shared_queue(name, entry, loaded=True, resume=resume)
                    candidate = bool(entry.get('shared_candidate') or entry.get('shared_draining'))
                    for worker in entry['workers']:
                        worker.shared_holder = await server.shared_runtime.hold(entry, ready=True)
                if resume and not candidate:
                    usage = await asyncio.to_thread(server.db.queue_usage, entry['queue'].key)
                    if control_token is None:
                        # First-time factory discovery cannot observe this queue
                        # before loading its schema. Join an active queue, but
                        # never resume a stopped queue from that late snapshot.
                        if not usage['accepting']:
                            raise ServerError(409, 'shared_control_changed', 'Shared entry is stopped; retry Start after schema discovery', retryable=True)
                    else:
                        await server._write_shared_control(entry, True, control_token)
                usage = await asyncio.to_thread(server.db.queue_usage, entry['queue'].key)
                if not candidate and not usage['accepting']:
                    await asyncio.gather(*(worker.close() for worker in entry['workers']))
                    await server._release_entry_files(entry)
                    entry['workers'], entry['status'] = [], 'stopped'
                    return server.describe(name)
            if server.shared_runtime is not None:
                await server.shared_runtime.started(entry)
            if persist:
                await server._persist_entry(name, entry, True)
            if server.redis_config is None:
                # Deploy history lives in DB with its retention policy. Keep
                # the editor's legacy in-memory history only on its own queue.
                entry['queue'] = TaskQueue(capacity=config.execution.capacity, history_limit=0)
            entry['wake'] = asyncio.Event()
            entry['futures'] = {}
            entry['closing'] = None
            entry['status'] = 'ready'
            entry['local_stop'] = False
            consume = server._consume if server.redis_config is None else server._consume_shared
            # Execution/claim/finalization remain owned even if an internal
            # consumer waiter is cancelled before or during its first await.
            entry['consumer_runs'] = [asyncio.create_task(consume(entry, worker)) for worker in entry['workers']
                                      for _ in range(config.execution.request_slots)]
            for run in entry['consumer_runs']:
                run.add_done_callback(lambda task, entry=entry: server._consumer_finished(entry, task))
            entry['consumers'] = [asyncio.create_task(server._join_consumer(run)) for run in entry['consumer_runs']]
            if server.redis_config is not None:
                entry['maintenance'] = asyncio.create_task(server._maintain_shared(entry))
        except Exception as exc:
            closed = await asyncio.gather(*(worker.close() for worker in entry['workers']), return_exceptions=True)
            if not any(isinstance(result, BaseException) for result in closed):
                try:
                    await server._release_entry_files(entry)
                    if server.shared_runtime is not None:
                        await server.shared_runtime.failed(entry)
                except Exception:
                    logging.exception('Could not release failed entry resource references')
            if not any(isinstance(result, BaseException) for result in closed):
                entry['workers'] = []
            superseded = isinstance(exc, ServerError) and exc.code in ('shared_control_changed', 'entry_busy') and not entry['workers']
            entry['status'], entry['error'] = 'stopped' if superseded else 'failed', str(exc)
            if server.redis_config is not None and entry.get('queue') is not None:
                await entry['queue'].close()
            raise
        return server.describe(name)


    async def reload(self, name: str, *, workflow=None, config=None, start_stopped=False):
        """Prepare replacement workers before changing the accepting entry."""
        server = self.server
        import copy
        from .config import ServerConfig, load_json
        from .errors import ServerError
        old = server._entry(name)
        stopped_update = start_stopped and server.shared_runtime is not None and old['status'] in ('stopped', 'failed') and not old['workers']
        failed_update = server.shared_runtime is not None and old.get('shared_key') and old['status'] == 'failed' and not old['workers']
        if (old['status'] != 'ready' and not stopped_update and not failed_update) or server.shutting_down:
            raise ServerError(409, 'entry_busy', 'Entry must be ready before reload', retryable=True)
        if old['source']['graph'] is not None:
            raise ServerError(409, 'reload_unsupported', 'Reload requires a workflow or a factory')
        source = dict(old['source'])
        if workflow is not None:
            if source['workflow'] is None:
                raise ServerError(422, 'invalid_request', 'This entry does not load a workflow')
            source['workflow'] = load_json(workflow) if isinstance(workflow, (str, Path)) else copy.deepcopy(workflow)
            if server.redis_config is not None:
                from .redis import source_identity
                source['identity'] = source_identity(source)
        value = old['config']
        if config is not None:
            if isinstance(config, (str, Path)):
                config = load_json(config)
            value = ServerConfig.model_validate(config)
            value = value.model_copy(update={'name': name})
            source['autodetect'] = False
            source['untyped'] = False
        candidate = replacement(old, source=source, config=value)
        if server.shared_runtime is not None:
            return await server.shared_runtime.reload(name, old, candidate)
        old['status'] = 'reloading'
        try:
            await server._start_entry(name, candidate, persist=False)
            await server._persist_entry(name, candidate, True)
        except BaseException:
            try:
                if candidate['status'] == 'ready':
                    await server._stop_entry(name, candidate, persist=False)
            finally:
                old['status'] = 'ready'
                if candidate['workers'] or candidate.get('release_pending'):
                    server.retiring_entries.setdefault(name, []).append(candidate)
                else:
                    server._release_assets(candidate)
            raise
        # There is no await between the pointer change and stopping old admission.
        server.entries[name] = candidate
        candidate['status'] = 'reloading'
        old['status'] = 'ready'
        server.retiring_entries.setdefault(name, []).append(old)
        try:
            await server._stop_entry(name, old, persist=False)
        except BaseException:
            candidate['error'] = 'Previous worker release was not confirmed'
            raise
        finally:
            candidate['status'] = 'ready'
        server.retiring_entries[name].remove(old)
        server._release_assets(old)
        return server.describe(name)


    async def stop(self, name: str, entry: Entry, *, persist=True, shared_scope=True, drain_shared=None):
        server = self.server
        from .errors import ServerError
        if entry['status'] in ('starting', 'reloading'):
            raise ServerError(409, 'entry_busy', 'Entry is changing state', retryable=True)
        if shared_scope and server.shared_runtime is not None and entry.get('shared_key'):
            await server.shared_runtime.check(entry)
        if entry['status'] in ('stopped', 'failed') and not entry['workers']:
            if entry.get('release_pending'):
                await server._release_entry_files(entry)
                entry['status'], entry['error'] = 'stopped', None
            if server.devices is not None:
                await server.devices.release(entry)
            if server.redis_config is not None and entry.get('queue') is not None and shared_scope:
                usage = await asyncio.to_thread(server.db.queue_usage, entry['queue'].key)
                await server._write_shared_control(entry, False, usage['control_token'])
            if persist and entry['status'] == 'stopped':
                await server._persist_entry(name, entry, False)
            return server.describe(name)
        if entry['closing'] is None:
            async def drain():
                usage = None
                if server.redis_config is not None and entry.get('queue') is not None and shared_scope:
                    usage = await asyncio.to_thread(server.db.queue_usage, entry['queue'].key)
                if persist:
                    await server._persist_entry(name, entry, False)
                if usage is not None:
                    await server._write_shared_control(entry, False, usage['control_token'])
                entry['status'] = 'stopping'
                entry['local_stop'] = not (shared_scope if drain_shared is None else drain_shared)
                if entry.get('wake') is not None:
                    entry['wake'].set()
                consumer_results = await asyncio.gather(*entry.get('consumer_runs', entry['consumers']), return_exceptions=True)
                await asyncio.gather(*entry['consumers'], return_exceptions=True)
                if entry.get('maintenance'):
                    entry['maintenance'].cancel()
                    await asyncio.gather(entry['maintenance'], return_exceptions=True)
                failure_message = 'Worker release was not confirmed'
                try:
                    closed = await asyncio.gather(*(worker.close() for worker in entry['workers']), return_exceptions=True)
                    for result in closed:
                        if isinstance(result, BaseException):
                            raise result
                    failure_message = 'Resource cleanup was not confirmed'
                    if entry.get('consumer_failed'):
                        await server._finish_abandoned_consumer_tasks(entry)
                    await server._release_entry_files(entry)
                except BaseException:
                    entry['status'], entry['error'] = 'failed', failure_message
                    raise
                else:
                    entry['workers'] = []
                    entry['status'], entry['error'] = 'stopped', None
                if entry.get('consumer_failed') or any(isinstance(result, BaseException) for result in consumer_results):
                    entry['status'], entry['error'] = 'failed', 'Task consumer exited unexpectedly'
                    raise ServerError(500, 'consumer_failed', 'Task consumer exited unexpectedly; Workers have been closed')
            entry['closing'] = asyncio.create_task(drain())
        closing = entry['closing']
        try:
            await asyncio.shield(closing)
        except BaseException:
            if entry['status'] in ('ready', 'failed') and closing.done() and entry['closing'] is closing:
                # A later Stop may retry idempotent database cleanup. Workers
                # remember actual close failures, so an unconfirmed physical
                # release cannot be turned into success by this retry.
                entry['closing'] = None
            raise
        return server.describe(name)


"""Demand-driven shared result readers; execution remains owned by the DB/worker."""
import asyncio
import logging

from .errors import ServerError


class SharedResults:
    def __init__(self, server):
        self.server = server
        self.readers = {}
        self.retiring = set()

    def record(self, record, *, observe=True):
        if record is None:
            raise ServerError(404, 'task_not_found', 'Task record is no longer available')
        server, identity = self.server, record['task_id']
        if not observe:
            receipt = asyncio.get_running_loop().create_future()
            receipt.set_result({'task_id': identity})
            return identity, receipt
        future = server.task_futures.get(identity)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            if record['state'] == 'succeeded':
                future.set_result(dict(task_id=identity, outputs=record['outputs']))
            elif record['queue_key'] and record['state'] in ('pending', 'running', 'dispatched') and server.redis_config is not None:
                future.shared_result = (self, identity)
                server.task_futures[identity] = future
                server.waiters[identity] = asyncio.create_task(self.wait(identity, future))
            else:
                future.set_exception(ServerError.from_task(record))
        return identity, future

    def attach(self, identity, future, observer):
        readers = self.readers.setdefault(future, set())
        readers.add(observer)

        def detached(done):
            readers.discard(done)
            if readers:
                return
            self.readers.pop(future, None)
            server = self.server
            if server.task_futures.get(identity) is future:
                # Remove synchronously before cancellation: a new subscriber must
                # never attach to a waiter which is already being retired.
                server.task_futures.pop(identity, None)
                task = server.waiters.pop(identity, None)
                if task is not None:
                    self.retiring.add(task)
                    task.add_done_callback(self.retiring.discard)
                    task.cancel()
        observer.add_done_callback(detached)

    async def close(self):
        pending = set(self.server.waiters.values()) | self.retiring
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # Cancellation before a wait coroutine's first instruction skips its
        # finally block. Settle those observers explicitly as well.
        for identity, future in list(self.server.task_futures.items()):
            shared = getattr(future, 'shared_result', None)
            if shared is not None and shared[0] is self:
                if not future.done():
                    future.set_exception(ServerError(503, 'server_stopping',
                        'Task remains recorded; query its status after reconnecting',
                        task_id=identity, retryable=True))
                self.server.task_futures.pop(identity, None)
                self.server.waiters.pop(identity, None)

    async def wait(self, identity, future):
        server = self.server
        delay = .5
        sequence = 0
        try:
            while not future.done():
                try:
                    record = await asyncio.to_thread(server.db.get_task, identity)
                    if record is None:
                        raise ServerError(404, 'task_not_found', 'Task record is no longer available', task_id=identity)
                    sequence = record.get('event_seq', 0)
                    if record['state'] in ('succeeded', 'failed', 'cancelled', 'expired'):
                        if record['state'] == 'succeeded':
                            future.set_result(dict(task_id=identity, outputs=record['outputs']))
                        else:
                            future.set_exception(ServerError.from_task(record))
                        return
                except ServerError as exc:
                    future.set_exception(exc)
                    return
                except Exception:
                    logging.warning('Waiting for shared task database recovery: %s', identity)
                # Redis is only a wakeup. A bounded fallback repairs lost signals
                # and DB commits made by a process without a notification client.
                if server.event_notifications is not None:
                    await server.event_notifications.wait(identity, sequence, timeout_seconds=delay, terminal=True)
                else:
                    await asyncio.sleep(delay)
                delay = min(5.0, delay * 2)
        finally:
            if not future.done():
                future.set_exception(ServerError(503, 'server_stopping' if server.shutting_down else 'reader_detached',
                    'Task remains recorded; query its status after reconnecting', task_id=identity, retryable=True))
            if server.task_futures.get(identity) is future:
                server.task_futures.pop(identity, None)
                server.waiters.pop(identity, None)

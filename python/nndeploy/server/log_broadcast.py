"""Bounded task log history and best-effort editor WebSocket delivery."""
import asyncio
from collections import OrderedDict, deque
import logging
import re

from .log_policy import DiagnosticFormatter


class LogBroadcaster(logging.Handler):
    max_tasks = 64
    max_lines = 200
    max_bytes = 4 * 1024 * 1024
    max_pending = 128
    send_timeout = 1.0

    def __init__(self, get_loop, get_ws_map, metrics=None):
        super().__init__()
        self.get_loop, self.get_ws_map = get_loop, get_ws_map
        self.task_log_map = OrderedDict()
        self.cached_bytes = 0
        self.pending = set()
        self.dropped_sends = 0
        self.failed_sends = 0
        self.closed = False
        self.metrics = metrics
        self.setFormatter(DiagnosticFormatter('%(asctime)s %(message)s'))

    def _remember(self, identity, message):
        lines = self.task_log_map.setdefault(identity, deque())
        self.task_log_map.move_to_end(identity)
        lines.append(message)
        self.cached_bytes += len(message.encode('utf-8'))
        if len(lines) > self.max_lines:
            self.cached_bytes -= len(lines.popleft().encode('utf-8'))
        while len(self.task_log_map) > self.max_tasks:
            _, old = self.task_log_map.popitem(last=False)
            self.cached_bytes -= sum(len(item.encode('utf-8')) for item in old)
        while self.cached_bytes > self.max_bytes:
            oldest = next(iter(self.task_log_map))
            old = self.task_log_map[oldest]
            self.cached_bytes -= len(old.popleft().encode('utf-8'))
            if not old:
                del self.task_log_map[oldest]

    def _drop(self, failed=False):
        if failed:
            self.failed_sends += 1
        else:
            self.dropped_sends += 1
        if self.metrics is not None:
            self.metrics.log_dropped.labels('websocket').inc()

    def emit(self, record):
        # Handler.handle and get_logs/close share this reentrant lock. Never
        # report sink failures by logging back into this handler.
        with self.lock:
            if self.closed:
                return
            try:
                message = self.format(record)
                identity = getattr(record, 'task_id', None)
                if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', identity) or identity == '0':
                    match = re.search(r'\[task_id=([A-Za-z0-9_-]{1,128})\]', message)
                    identity = match.group(1) if match else None
                if identity is None:
                    return
                self._remember(identity, message)
                loop = self.get_loop()
                sockets = self.get_ws_map().get(identity, ())
                if loop is None or not loop.is_running():
                    return
                for ws in list(sockets):
                    if len(self.pending) >= self.max_pending:
                        self._drop()
                        continue
                    coroutine = self._send_log(ws, identity, message)
                    try:
                        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                    except BaseException:
                        coroutine.close()
                        raise
                    self.pending.add(future)
                    future.add_done_callback(self._completed)
            except Exception:
                self._drop(failed=True)

    def _completed(self, future):
        with self.lock:
            self.pending.discard(future)
        if not future.cancelled():
            future.exception()

    async def _send_log(self, ws, task_id, message):
        try:
            await asyncio.wait_for(ws.send_json({
                'flag': 'success', 'message': 'log update',
                'result': {'task_id': task_id, 'type': 'log', 'log': message},
            }), self.send_timeout)
        except Exception:
            with self.lock:
                self._drop(failed=True)

    def get_logs(self, task_id):
        with self.lock:
            return list(self.task_log_map.get(task_id, ()))

    def close(self):
        with self.lock:
            self.closed = True
            pending = list(self.pending)
            self.task_log_map.clear()
            self.cached_bytes = 0
        for future in pending:
            future.cancel()
        super().close()

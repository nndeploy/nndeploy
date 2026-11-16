# app.py

import argparse
import asyncio
from contextlib import asynccontextmanager
import threading
import logging
import os
import multiprocessing as mp
from multiprocessing.managers import MakeProxyType, SyncManager
from queue import Empty, Full, Queue
import time
import uuid
import sys
from pathlib import Path
from logging.handlers import QueueListener
from nndeploy.dag.node import add_global_import_lib, import_global_import_lib

from .task_queue import TaskQueue, ExecutionStatus
from .server import NnDeployServer
from .worker import run as worker_run, watch_editor_parent
from .log_broadcast import LogBroadcaster
from .logging_taskid import install_taskid_logrecord_factory, EditorQueueHandler

WORKER_RESTART_DELAY_SECONDS = 2
WORKER_MONITOR_INTERVAL_SECONDS = 1
EDITOR_IPC_TIMEOUT_SECONDS = 30


class _EditorIpcWatchdog:
    """Break stalled local RPCs without acquiring queue or logging locks."""

    def __init__(self):
        self.timeout = EDITOR_IPC_TIMEOUT_SECONDS
        self.lock = threading.Lock()
        self.pending = {}
        self.timed_out = None
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(name='EditorIpcWatchdogThread', target=self._run, daemon=True)
        self.thread.start()

    def begin(self, owner, wait):
        token = object()
        with self.lock:
            self.pending[token] = (owner, time.monotonic() + self.timeout + wait)
        return token

    def end(self, token):
        with self.lock:
            self.pending.pop(token, None)

    def _run(self):
        while not self.stop.wait(.1):
            now = time.monotonic()
            with self.lock:
                expired = {owner for owner, deadline in self.pending.values() if now >= deadline}
            for owner in expired:
                # During start(), the Process is assigned before its address
                # handshake completes. Keep the manager identity until then.
                process = getattr(owner, '_process', None) if isinstance(owner, SyncManager) else owner
                if process is None:
                    continue
                try:
                    if process.is_alive():
                        self.timed_out = process
                        # SIGTERM does not wake a SIGSTOP'ed process. Killing
                        # this exact manager also releases blocked RPC clients;
                        # the runtime still reaps the Worker before recovery.
                        process.kill()
                except (OSError, ValueError):
                    pass  # Concurrent cleanup may already have reaped it.

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()


class _EditorTimedCalls:
    def _callmethod(self, methodname, args=(), kwds=None):
        kwds = {} if kwds is None else kwds
        watch = getattr(self, '_editor_watch', None)
        wait = 0
        if methodname in ('get', 'put'):
            offset = 0 if methodname == 'get' else 1
            block = kwds.get('block', args[offset] if len(args) > offset else True)
            timeout = kwds.get('timeout', args[offset + 1] if len(args) > offset + 1 else None)
            if block and timeout is not None:
                wait = max(0, timeout)
            elif block:
                watch = None  # A declared unbounded queue wait is not an IPC deadline.
        elif methodname == 'join':
            watch = None
        if watch is None:
            return super()._callmethod(methodname, args, kwds)
        watchdog, process = watch
        token = watchdog.begin(process, wait)
        try:
            return super()._callmethod(methodname, args, kwds)
        finally:
            watchdog.end(token)


_EDITOR_QUEUE_METHODS = ('get', 'get_nowait', 'put', 'put_nowait', 'empty', 'full',
                         'qsize', 'task_done', 'join')


class _EditorQueueProxy(_EditorTimedCalls, MakeProxyType('_EditorQueueMethods', _EDITOR_QUEUE_METHODS)):
    pass


class _EditorJobProxy(_EditorTimedCalls, MakeProxyType('_EditorJobMethods', _EDITOR_QUEUE_METHODS + (
        'new_reader', 'claim', 'cancel', 'cancellation_requested', 'finish', 'outcomes',
        'acknowledge', 'retire_reader', 'begin_execution', 'execution_expired'))):
    pass


class _EditorJobQueue(Queue):
    """Invalidate blocked reads from a dead Worker before its replacement starts."""

    def __init__(self, maxsize=0):
        super().__init__(maxsize)
        self.reader = None
        self.deliveries = {}
        self.active = {}
        self.completed = {}
        self.one_shot = False
        self.exit_deadline = None
        self.retired = (None, False)

    def _qsize(self):
        return len(self.queue) + len(self.deliveries)

    def new_reader(self):
        with self.not_empty:
            if self.active:
                raise RuntimeError('Previous Worker has not confirmed exit')
            # A dead client's outstanding get may already have removed a job.
            # The Worker must confirm receipt before executing it; only these
            # unconfirmed deliveries can safely return to the pending queue.
            for item in reversed(list(self.deliveries.values())):
                self.queue.appendleft(item)
            self.deliveries.clear()
            self.reader = uuid.uuid4().hex
            self.one_shot = False
            self.exit_deadline = None
            self.not_empty.notify_all()
            return self.reader

    def claim(self, idx, reader):
        with self.not_full:
            if reader != self.reader or idx not in self.deliveries:
                return False
            item = self.deliveries.pop(idx)
            payload = item[1] if isinstance(item[1], dict) else {}
            settings = payload.get('_editor_execution', {}).get('execution', {})
            self.one_shot |= bool(payload.get('_editor_execution'))
            self.active[idx] = dict(reader=reader, cancel_requested=False, deadline=None,
                timeout=settings.get('execution_timeout_seconds'), termination_reason=None,
                queue_expires=(payload['_editor_submitted'] + settings.get('queue_timeout_seconds', 30)
                               if settings and '_editor_submitted' in payload else None))
            self.not_full.notify()
            return True

    def cancel(self, idx, only_pending=False):
        with self.not_full:
            if idx in self.completed:
                return 'finished'
            if idx in self.active:
                if only_pending:
                    return 'running'
                self.active[idx]['cancel_requested'] = True
                return 'requested'
            item = self.deliveries.pop(idx, None)
            if item is None:
                for queued in self.queue:
                    if queued[0] == idx:
                        item = queued
                        self.queue.remove(queued)
                        break
            if item is None:
                return 'absent'
            self.completed[idx] = (idx, ExecutionStatus(False, 'Cancelled before execution', 'cancelled'), {}, {})
            self.not_full.notify()
            return 'cancelled'

    def cancellation_requested(self, idx, reader):
        with self.mutex:
            record = self.active.get(idx)
            return record is not None and record['reader'] == reader and record['cancel_requested']

    def begin_execution(self, idx, reader):
        """Arm the external deadline only after resource waiting has ended."""
        with self.mutex:
            record = self.active.get(idx)
            if record is None or record['reader'] != reader or record['termination_reason']:
                return False
            if record['cancel_requested']:
                record['termination_reason'] = 'Execution cancelled'
                return False
            if record['queue_expires'] is not None and time.time() >= record['queue_expires']:
                record['termination_reason'] = 'Queue timed out before execution'
                return False
            if record['timeout'] is not None and record['deadline'] is None:
                record['deadline'] = time.monotonic() + record['timeout']
            return True

    @staticmethod
    def _expired(record):
        if record['deadline'] is not None and time.monotonic() >= record['deadline']:
            record['termination_reason'] = 'Execution timed out'
        return bool(record['termination_reason'])

    def execution_expired(self, reader):
        with self.mutex:
            if reader != self.reader:
                return False
            if self.exit_deadline is not None and time.monotonic() >= self.exit_deadline:
                return True
            return any(self._expired(record) for record in self.active.values() if record['reader'] == reader)

    def finish(self, idx, reader, status, results, profile):
        with self.mutex:
            record = self.active.get(idx)
            if record is None or record['reader'] != reader:
                return False
            if self._expired(record):
                return False  # Keep ownership until the parent confirms exit.
            self.completed[idx] = (idx, status, results, profile)
            # An accepted result is immutable, but its one-shot process may
            # still hang in finalizers or while sending the completion hint.
            self.exit_deadline = record['deadline']
            del self.active[idx]
            return True

    def outcomes(self, limit=16, idx=None):
        with self.mutex:
            if idx is not None:
                return [self.completed[idx]] if idx in self.completed else []
            return list(self.completed.values())[:limit]

    def acknowledge(self, idx):
        with self.mutex:
            self.completed.pop(idx, None)

    def retire_reader(self, reader):
        """Called only after the owning process has physically exited."""
        with self.not_empty:
            if self.reader != reader:
                return self.retired[1] if self.retired[0] == reader else False
            self.retired = (reader, self.one_shot)
            self.reader = None
            for item in reversed(list(self.deliveries.values())):
                self.queue.appendleft(item)
            self.deliveries.clear()
            for idx, record in list(self.active.items()):
                if record['reader'] == reader:
                    cancelled = record['cancel_requested']
                    status = ExecutionStatus(False, record['termination_reason'] or 'Worker exited before reporting a result',
                                             'cancelled' if cancelled else 'failed')
                    self.completed[idx] = (idx, status, {}, {})
                    del self.active[idx]
            self.not_empty.notify_all()
            return self.one_shot

    def get(self, block=True, timeout=None, *, reader=None):
        if timeout is not None and timeout < 0:
            raise ValueError('timeout must be nonnegative')
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.not_empty:
            while True:
                if reader is not None and reader != self.reader:
                    raise Empty
                if self.queue:
                    item = self._get()
                    if reader is None:
                        self.not_full.notify()
                    else:
                        self.deliveries[item[0]] = item
                    return item
                if not block:
                    raise Empty
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise Empty
                self.not_empty.wait(remaining)


class _EditorManager(SyncManager):
    pass


_EditorManager.register('JobQueue', _EditorJobQueue, proxytype=_EditorJobProxy)
_EditorManager.register('Queue', Queue, proxytype=_EditorQueueProxy)

def cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resources", default="./resources")
    ap.add_argument("--log", default="./logs/nndeploy_server.log")
    ap.add_argument("--front-end-version", default="!", help="GitHub frontend, as owner/repo@0.0.1 or @latest,default latest")
    ap.add_argument("--front-end-dir", help="Serve a local frontend build instead of downloading a release")
    ap.add_argument("--debug", dest="debug", action="store_true",
                    help="enable debug mode")
    ap.add_argument("--no-debug", dest="debug", action="store_false",
                    help="disable debug mode")
    ap.add_argument("--plugin", type=str, nargs='*', default=[], required=False)
    ap.add_argument("--api-only", action="store_true", help="Start the Deploy API without the editor")
    ap.add_argument("--workflow", help="Workflow JSON to run through the Deploy API")
    ap.add_argument("--config", help="Optional workflow.server.json path")
    ap.add_argument("--name", help="Name used in the Deploy URL")
    ap.add_argument("--admin-token-env", help="Environment variable containing the management token")
    ap.add_argument("--database-url-env", help="Environment variable containing a SQLite, MySQL or PostgreSQL URL")
    database_action = ap.add_mutually_exclusive_group()
    database_action.add_argument('--database-backup', metavar='PATH', help='Write a native database backup and exit without starting Server')
    database_action.add_argument('--database-restore', metavar='PATH', help='Restore a trusted native backup into an empty database and exit')
    database_action.add_argument('--database-export', metavar='PATH', help='Export Server records to portable JSONL and exit')
    database_action.add_argument('--database-import', metavar='PATH', help='Import portable Server records into an empty database and exit')
    ap.add_argument('--database-operation-timeout-seconds', type=float, default=300,
                    help='Database backup/restore or transfer execution deadline (default: 300 seconds)')
    ap.add_argument("--editor-history-retention-seconds", type=float, default=None,
                    help="Optional completed editor history retention; at least 60 seconds, default: keep history")
    ap.add_argument("--database-timeout-seconds", type=float, default=None, help="Database connection, pool and I/O timeout (default: 30 seconds)")
    ap.add_argument("--storage-config", help="JSON storage settings; credentials are referenced by environment variable name")
    ap.add_argument("--redis-config", help="Optional JSON settings for a shared Redis namespace")
    ap.add_argument("--device-limits", help="JSON resident model budgets; does not change algorithm device placement")
    ap.add_argument("--auth-config", help="JSON API key permissions; token values are read from environment variables")
    ap.add_argument("--audit-config", help="Optional JSON HTTP audit settings; an empty object enables default retention")
    ap.add_argument("--cors-origin", action="append", default=[], help="Explicit browser origin; repeat for multiple origins")
    ap.add_argument("--max-sync-connections", type=int, default=None, help="Maximum simultaneous synchronous Deploy HTTP responses")
    ap.add_argument(
        "--json_file",
        type=str,
        nargs='*',
        default=[],
        help="Path(s) to one or more JSON files to copy into resources/workflow directory"
    )
    ap.set_defaults(debug=False)
    args = ap.parse_args(argv)
    if not 0 < args.database_operation_timeout_seconds <= 86400:
        ap.error('--database-operation-timeout-seconds must be finite, positive and at most 86400')
    if (args.database_backup or args.database_restore or args.database_export or args.database_import) and (not args.database_url_env or args.api_only or args.workflow or args.config or args.name):
        ap.error('Database operations require --database-url-env and cannot start a workflow or API server')
    if args.database_timeout_seconds is not None and not 0 < args.database_timeout_seconds <= 86400:
        ap.error('--database-timeout-seconds must be finite, positive and at most 86400')
    if args.max_sync_connections is not None and args.max_sync_connections < 1:
        ap.error('--max-sync-connections must be positive')
    if not args.api_only and any((args.workflow, args.config, args.name)):
        ap.error("--workflow, --config and --name require --api-only")
    if args.api_only and not args.workflow:
        ap.error("--api-only requires --workflow")
    args.host = args.host or ("127.0.0.1" if args.api_only else "0.0.0.0")
    return args

from .logging_taskid import BoundedLogListener


class _EditorLogListener(BoundedLogListener):
    pass


def configure_root_logger(log_q: mp.Queue, log_file: str, server) -> QueueListener:
    from .request import StructuredFormatter
    from .log_policy import QueuedDiagnosticStreamHandler, DiagnosticRotatingFileHandler
    log_fmt = "%(asctime)s %(processName)s %(message)s"
    root = logging.getLogger()
    log_broadcaster = LogBroadcaster(
        get_loop=lambda: server.loop,
        get_ws_map=lambda: server.task_ws_map,
        metrics=getattr(server, 'metrics', None),
    )

    handlers = [
        QueuedDiagnosticStreamHandler(sys.stdout),
        DiagnosticRotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        ),
        log_broadcaster
    ]
    for h in handlers:
        h.setFormatter(StructuredFormatter(log_fmt))
    for handler, sink in zip(handlers[:2], ('console', 'file')):
        handler.metrics, handler.log_sink = getattr(server, 'metrics', None), sink
    
    previous_handlers, previous_level = root.handlers[:], root.level
    listener = _EditorLogListener(log_q, *handlers, respect_handler_level=True)
    listener.metrics = getattr(server, 'metrics', None)
    try:
        root.setLevel(logging.INFO)
        root.handlers = [EditorQueueHandler(log_q)]
        listener.start()
        return listener
    except BaseException:
        root.handlers, root.level = previous_handlers, previous_level
        for handler in handlers:
            handler.close()
        raise

def _close_worker(worker: mp.Process | None) -> None:
    if worker is None or getattr(worker, '_closed', False):
        return
    if worker.pid is not None:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=3)
        if worker.is_alive():
            worker.kill()
            worker.join()
    worker.close()


def _editor_storage_options(server):
    storage = getattr(server, 'storage', None)
    if storage is None:
        return None
    options = dict(database_url=storage.db.url.render_as_string(hide_password=False),
                timeout_seconds=storage.db.timeout_seconds, root=str(storage.root),
                max_bytes=storage.max_bytes, config=storage.config.model_dump())
    if getattr(server, 'devices', None) is not None:
        options['devices'] = dict(settings=server.devices.settings.model_dump(),
                                  config_scope=server.config_scope, data_dir=str(server.data_dir))
    return options


def start_worker(
        task_q: "mp.queues.Queue",
        result_q: "mp.queues.Queue",
        progress_q: "mp.queues.Queue",
        log_q: "mp.queues.Queue",
        plugin_update_q: "mp.queues.Queue",
        cancel_event_q: "mp.queues.Queue",
        resources, *, storage_options=None) -> mp.Process:
    reader = task_q.new_reader() if hasattr(task_q, 'new_reader') else None
    p = mp.Process(
        target=worker_run,
        name="WorkerProcess",
        args=(task_q, result_q, progress_q, log_q, plugin_update_q, cancel_event_q, resources),
        kwargs=dict(reader=reader, storage_options=storage_options),
        daemon=True,
    )
    p._editor_reader = reader
    try:
        p.start()
        logging.info("Worker spawned, pid=%s", p.pid)
        return p
    except BaseException:
        _close_worker(p)
        if reader is not None:
            try:
                task_q.retire_reader(reader)
            except (EOFError, OSError):
                pass  # The parent handles loss of the entire queue process.
        raise

def monitor_worker(
    worker: mp.Process,
    task_q: "mp.queues.Queue",
    result_q: "mp.queues.Queue",
    progress_q: "mp.queues.Queue",
    log_q: "mp.queues.Queue",
    plugin_update_q: "mp.queues.Queue",
    cancel_event_q: "mp.queues.Queue",
    resources,
    server: NnDeployServer,
    stop_event: threading.Event,
    *,
    on_worker=None,
) -> None:
    restart_count = 0
    restart_delay = WORKER_RESTART_DELAY_SECONDS
    stopped_by_policy = False
    try:
        while not stop_event.is_set():
            if worker is not None and worker.is_alive():
                reader = getattr(worker, '_editor_reader', None)
                if reader is not None and task_q.execution_expired(reader):
                    stopped_by_policy = True
                    try:
                        worker.kill()
                    except ProcessLookupError:
                        pass
                    worker.join(timeout=.1)
            if worker is not None and not worker.is_alive():
                exitcode = worker.exitcode
                reader = getattr(worker, '_editor_reader', None)
                one_shot = False
                if reader is not None:
                    try:
                        one_shot = task_q.retire_reader(reader)
                    except Exception:
                        logging.exception('[monitor_worker] task recovery failed; will retry before restart')
                        stop_event.wait(WORKER_MONITOR_INTERVAL_SECONDS)
                        continue
                _close_worker(worker)
                worker = None
                if getattr(server, 'devices', None) is not None:
                    try:
                        # A one-shot editor Worker retains its device guard
                        # until physical exit. Reconcile now so queued Deploy
                        # calls need not wait for the maintenance interval.
                        server.devices.recover()
                    except Exception:
                        logging.exception('Editor device release awaits database recovery')
                normal_exit = stopped_by_policy or exitcode == 0 and (one_shot or getattr(server, 'devices', None) is not None)
                restart_delay = 0 if normal_exit else WORKER_RESTART_DELAY_SECONDS
                report = logging.info if normal_exit else logging.error
                report("Worker exited (exitcode=%s). Restarting in %s seconds...", exitcode, restart_delay)
                try:
                    if not normal_exit:
                        server.notify_system_event(
                            "worker_died",
                            {"exitcode": exitcode, "restart_in_sec": restart_delay}
                        )
                except Exception:
                    logging.exception("[monitor_worker] notify worker_died failed")
            if worker is None:
                if stop_event.wait(restart_delay):
                    break
                try:
                    worker = start_worker(task_q, result_q, progress_q, log_q,
                                          plugin_update_q, cancel_event_q, resources,
                                          **({'storage_options': _editor_storage_options(server)} if hasattr(server, 'storage') else {}))
                    stopped_by_policy = False
                    if on_worker is not None:
                        on_worker(worker)
                except Exception:
                    restart_delay = WORKER_RESTART_DELAY_SECONDS
                    logging.exception("[monitor_worker] restart failed; will retry")
                    continue
                # A spawn may finish after shutdown began. Keep ownership until
                # finally reaps it, without announcing a running replacement.
                if stop_event.is_set():
                    break
                restart_count += 1
                logging.info("Worker restarted, new pid=%s (restart_count=%s)",
                             worker.pid, restart_count)
                try:
                    server.notify_system_event("worker_restarted", {"pid": worker.pid})
                except Exception:
                    logging.exception("[monitor_worker] notify worker_restarted failed")
            stop_event.wait(WORKER_MONITOR_INTERVAL_SECONDS)
    finally:
        reader = getattr(worker, '_editor_reader', None)
        _close_worker(worker)
        if getattr(server, 'devices', None) is not None:
            try:
                server.devices.recover()
            except Exception:
                logging.exception('Editor device release awaits database recovery')
        if reader is not None:
            try:
                task_q.retire_reader(reader)
            except (EOFError, OSError):
                pass  # The parent handles loss of the entire queue process.

def start_scheduler(queue: TaskQueue, job_q: mp.Queue, stop_event=None):
    from sqlalchemy.exc import SQLAlchemyError
    from .errors import ServerError
    stop_event = stop_event if stop_event is not None else threading.Event()
    def _loop():
        while not stop_event.is_set():
            try:
                item = queue.get(timeout=.1)
            except (SQLAlchemyError, ServerError):
                stop_event.wait(.1)
                continue
            if item is None:
                continue
            idx, payload = item
            while not stop_event.is_set():
                try:
                    queue.dispatch(idx, payload, job_q)
                except Full:
                    stop_event.wait(.1)
                    continue
                except (SQLAlchemyError, ServerError):
                    # Keep this task owned locally. Publishing without a
                    # confirmed durable dispatch could duplicate it on restart.
                    stop_event.wait(.1)
                    continue
                except (EOFError, OSError):
                    queue.pause_editor()
                    return
                break
            else:
                if not queue._editor_paused:
                    queue.task_done(idx, ExecutionStatus(False, 'Server stopped before dispatch', 'cancelled'), {}, {})

    th = threading.Thread(name="SchedulerThread", target=_loop, daemon=True)
    th.start()
    return th

def start_finisher(queue: TaskQueue, result_q: mp.Queue, stop_event=None):
    stop_event = stop_event if stop_event is not None else threading.Event()
    def _loop():
        while not stop_event.is_set():
            try:
                queue.collect_results()
            except Exception:
                logging.exception('[FinisherThread] pending result recovery failed')
            try:
                idx, status, results, time_profile_map = result_q.get(timeout=.1)
            except Empty:
                continue
            except (EOFError, OSError, ValueError):
                if not stop_event.is_set():
                    logging.exception('[FinisherThread] result queue closed')
                return
            try:
                queue.finish_result((idx, status, results, time_profile_map))
            except Exception:
                logging.exception('[FinisherThread] task notification failed')

    th = threading.Thread(name="FinisherThread", target=_loop, daemon=True)
    th.start()
    return th

def start_progress_listener(server: NnDeployServer, progress_q: mp.Queue, stop_event=None):
    stop_event = stop_event if stop_event is not None else threading.Event()
    def _on_started(task_id: str, d: dict):
        server.queue.mark_started(task_id, worker_pid=d.get("pid"))
    def _on_progress(task_id: str, d: dict):
        server.notify_task_progress(task_id, d.get("status"))
    def _on_finished(task_id: str, d: dict):
        server.notify_task_progress(task_id, d.get("status"))

    handlers = {
        "started": _on_started,
        "progress": _on_progress,
        "finished": _on_finished
    }

    def _loop():
        while not stop_event.is_set():
            try:
                idx, task_id, result = progress_q.get(timeout=.1)
            except Empty:
                continue
            except (EOFError, OSError, ValueError):
                if not stop_event.is_set():
                    logging.exception('[ProgressThread] progress queue closed')
                return
            d = result or {}
            evt = d.get("event")
            handler = handlers.get(evt)
            if handler:
                try:
                    handler(task_id, d)
                except Exception:
                    logging.exception("[ProgressThread] handler failed: evt=%s task_id=%s", evt, task_id)
            else:
                logging.debug("[ProgressThread] ignore unknown event: %s for task %s", evt, task_id)
    th = threading.Thread(name="ProgressThread", target=_loop, daemon=True)
    th.start()
    return th

def load_existing_plugins(plugin_dir: Path):
    for f in plugin_dir.iterdir():
        if f.suffix in {".py", ".so"} and f.is_file():
            add_global_import_lib(str(f.resolve()))
    import_global_import_lib()


class _EditorRuntime:
    """Own editor IPC and its workers for the lifetime of one HTTP Server.

    Queue process or helper exit pauses admission and rebuilds the local cycle.
    Stop computation before finalizing uncertain tasks or replacing channels;
    preserve the parent queue and collect known results before discarding IPC.
    """

    def __init__(self, args):
        self.args = args
        self.server = None
        self.manager = None
        self.worker = None
        self.monitor = None
        self.listener = None
        self.threads = []
        self.supervisor = None
        self.stop = threading.Event()
        self.cycle_stop = threading.Event()
        self.available = threading.Event()
        self.close_lock = threading.Lock()
        self.closed = False
        root = logging.getLogger()
        self.previous_logging = root.handlers[:], root.level
        self.ipc_watchdog = _EditorIpcWatchdog()

    @property
    def ready(self):
        return (self.available.is_set() and not self.stop.is_set()
                and self._cycle_failure() is None)

    def _cycle_failure(self):
        """Check owned components without making an IPC or database call."""
        manager = self.manager
        if manager is not None and self.ipc_watchdog.timed_out is manager._process:
            return 'queue IPC timed out'
        if manager is None or not manager._process.is_alive():
            return 'queue process exited'
        for thread in (*self.threads, self.monitor):
            if thread is None or not thread.is_alive():
                return f'{thread.name if thread is not None else "Worker monitor"} exited'
        listener = self.listener
        listener_thread = listener._thread if listener is not None else None
        if listener_thread is None or not listener_thread.is_alive():
            return 'log listener exited'
        return None

    def _channels(self):
        self.manager = _EditorManager()
        token = self.ipc_watchdog.begin(self.manager, 0)
        try:
            self.manager.start(initializer=watch_editor_parent)
            self.job_q = self.manager.JobQueue(maxsize=256)
            self.result_q = self.manager.Queue(maxsize=256)
            self.progress_q = self.manager.Queue(maxsize=1024)
            from .log_policy import LOG_QUEUE_SIZE
            self.log_q = self.manager.Queue(maxsize=LOG_QUEUE_SIZE)
            # Files in resources/plugin are authoritative; one pending hint is
            # enough to represent any number of completed uploads.
            self.plugin_q = self.manager.Queue(maxsize=1)
            self.cancel_q = self.manager.Queue()
            for channel in (self.job_q, self.result_q, self.progress_q, self.log_q, self.plugin_q, self.cancel_q):
                # Child proxy reconstruction omits this parent-local watchdog.
                channel._editor_watch = (self.ipc_watchdog, self.manager._process)
        except Exception as error:
            if self.ipc_watchdog.timed_out is not None and self.ipc_watchdog.timed_out is getattr(self.manager, '_process', None):
                raise TimeoutError('Editor queue startup timed out') from error
            raise
        finally:
            self.ipc_watchdog.end(token)
        if self.ipc_watchdog.timed_out is self.manager._process:
            raise TimeoutError('Editor queue startup timed out')

    def _start_cycle(self):
        self.cycle_stop = threading.Event()
        server = self.server
        server.plugin_update_q, server.cancel_event_queue = self.plugin_q, self.cancel_q
        # No old helper is alive when this reference changes.
        server.queue._job_q = self.job_q
        self.threads.append(start_scheduler(server.queue, self.job_q, self.cycle_stop))
        self.threads.append(start_finisher(server.queue, self.result_q, self.cycle_stop))
        self.worker = start_worker(self.job_q, self.result_q, self.progress_q, self.log_q,
                                   self.plugin_q, self.cancel_q, self.args.resources,
                                   storage_options=_editor_storage_options(server))
        self.monitor = threading.Thread(name='WorkerMonitorThread', target=monitor_worker,
            args=(self.worker, self.job_q, self.result_q, self.progress_q, self.log_q,
                  self.plugin_q, self.cancel_q, self.args.resources, server, self.cycle_stop),
            kwargs={'on_worker': self._own_worker}, daemon=True)
        self.monitor.start()
        self.listener = configure_root_logger(self.log_q, self.args.log, server)
        self.threads.append(start_progress_listener(server, self.progress_q, self.cycle_stop))
        if not self.stop.is_set():
            server.queue.resume_editor(self.job_q)
            self.available.set()

    def _own_worker(self, worker):
        self.worker = worker

    def notify_plugin(self, path):
        # Plugins are saved on disk before this notification. Every new Worker
        # scans that directory, including files uploaded during recovery.
        if not self.ready:
            return False
        try:
            self.plugin_q.put_nowait(path)
            return True
        except Full:
            return True
        except (EOFError, OSError):
            return False

    def start(self, server_factory):
        try:
            self.ipc_watchdog.start()
            self._channels()
            self.server = server_factory(self.job_q, self.plugin_q, self.cancel_q)
            self.server._editor_runtime = self
            self.server.queue.pause_editor()
            self._start_cycle()
            self.supervisor = threading.Thread(name='EditorQueueMonitorThread', target=self._supervise, daemon=True)
            self.supervisor.start()
            return self.server
        except BaseException:
            self.close()
            raise

    def _stop_cycle(self, *, recovering=False):
        self.available.clear()
        if self.server is not None:
            self.server.queue.pause_editor()
        # Detach a dead IPC logger before reporting any recovery errors.
        root = logging.getLogger()
        root.handlers, root.level = self.previous_logging
        self.cycle_stop.set()
        if self.monitor is not None and self.monitor.ident is not None:
            self.monitor.join()  # Owns and reaps replacements as well as the first Worker.
        _close_worker(self.worker)
        for thread in self.threads:
            thread.join()
        self.threads.clear()
        self.worker = self.monitor = None
        reconciled = False
        try:
            if self.server is not None:
                try:
                    while self.server.queue.collect_results():
                        pass
                except (EOFError, OSError):
                    pass
                self.server.queue.fail_editor_active()
            reconciled = True
        finally:
            try:
                if self.listener is not None:
                    try:
                        self.listener.stop()
                    finally:
                        for handler in getattr(self.listener, 'handlers', ()):
                            handler.close()
                        self.listener = None
            finally:
                # A live manager may still hold results beyond the first
                # collection batch. Retry history writes before discarding it.
                # On final shutdown, release IPC even if the database is down.
                if self.manager is not None and (reconciled or not recovering):
                    manager = self.manager
                    token = self.ipc_watchdog.begin(manager, 0)
                    try:
                        if hasattr(manager, 'shutdown'):
                            manager.shutdown()
                    finally:
                        try:
                            # start() may fail before registering shutdown().
                            process = getattr(manager, '_process', None)
                            if process is not None and process.pid is not None:
                                if process.is_alive():
                                    process.kill()
                                process.join()
                        finally:
                            self.ipc_watchdog.end(token)
                            self.manager = None

    def _supervise(self):
        while not self.stop.wait(WORKER_MONITOR_INTERVAL_SECONDS):
            failure = self._cycle_failure()
            if failure is None:
                continue
            while not self.stop.is_set():
                try:
                    self._stop_cycle(recovering=True)
                except Exception:
                    logging.exception('Editor recovery cleanup failed; will retry')
                    self.stop.wait(WORKER_RESTART_DELAY_SECONDS)
                    continue
                logging.error('Editor %s; rebuilding local queues', failure)
                if self.stop.wait(WORKER_RESTART_DELAY_SECONDS):
                    break
                try:
                    self._channels()
                    if self.stop.is_set():
                        break
                    self._start_cycle()
                except Exception:
                    logging.exception('Editor queue restart failed; will retry')
                    continue
                if self.stop.is_set():
                    break
                try:
                    self.server.notify_system_event('worker_restarted', {'pid': self.worker.pid})
                except Exception:
                    logging.exception('Editor restart notification failed')
                break

    def close(self):
        with self.close_lock:
            if self.closed:
                return
            self.stop.set()
            if self.supervisor is not None and self.supervisor.ident is not None:
                self.supervisor.join()
            try:
                self._stop_cycle()
                self.closed = True
            finally:
                self.ipc_watchdog.close()

def main() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    
    mp.set_start_method("spawn", force=True)

    args = cli()
    token = None
    if args.admin_token_env:
        token = os.environ.get(args.admin_token_env)
        if not token:
            raise ValueError("The management token environment variable is empty or missing")
    database_url = None
    if args.database_url_env:
        database_url = os.environ.get(args.database_url_env)
        if not database_url:
            raise ValueError("The database URL environment variable is empty or missing")
    if args.database_backup or args.database_restore or args.database_export or args.database_import:
        from .db import DB
        database = DB(database_url, timeout_seconds=args.database_timeout_seconds)
        try:
            if args.database_backup:
                path = database.backup(args.database_backup, timeout_seconds=args.database_operation_timeout_seconds)
            elif args.database_restore:
                path = database.restore(args.database_restore, timeout_seconds=args.database_operation_timeout_seconds)
            elif args.database_export:
                path = database.export_data(args.database_export, timeout_seconds=args.database_operation_timeout_seconds)
            else:
                path = database.import_data(args.database_import, timeout_seconds=args.database_operation_timeout_seconds)
            label = 'backup saved: ' if args.database_backup else 'restored from: ' if args.database_restore else 'operation completed: '
            print('Database ' + label + str(path))
        finally:
            database.close()
        return
    storage_config = None
    if args.storage_config:
        from .config import load_json
        storage_config = load_json(args.storage_config)
    redis_config = None
    if args.redis_config:
        from .config import load_json
        redis_config = load_json(args.redis_config)
    auth_config = None
    if args.auth_config:
        from .config import load_json
        auth_config = load_json(args.auth_config)
    audit_config = None
    if args.audit_config:
        from .config import load_json
        audit_config = load_json(args.audit_config)
    device_limits = None
    if args.device_limits:
        from .config import load_json
        device_limits = load_json(args.device_limits)
    if args.api_only:
        for plugin in args.plugin:
            add_global_import_lib(str(Path(plugin).resolve()))
        if args.plugin:
            import_global_import_lib()
        server = NnDeployServer(workflow=args.workflow, config=args.config, name=args.name,
                                data_dir=Path(args.resources) / "server", admin_token=token, database_url=database_url,
                                database_timeout_seconds=args.database_timeout_seconds,
                                storage_config=storage_config, redis_config=redis_config, auth_config=auth_config, audit_config=audit_config,
                                cors_origins=args.cors_origin, max_sync_connections=args.max_sync_connections, device_limits=device_limits)
        server.run(host=args.host, port=args.port)
        return
    install_taskid_logrecord_factory()

    # copy json file to workflow directory
    json_files = []
    for item in args.json_file:
        for part in item.split(","):
            part = part.strip()
            if part:
                json_files.append(part)

    if json_files:
        import shutil
        workflow_dir = Path(args.resources) / "workflow"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        for json_path in json_files:
            try:
                src = Path(json_path)
                if not src.exists():
                    logging.warning(f"JSON file not found: {src}")
                    continue
                shutil.copy(src, workflow_dir)
                logging.info(f"Copied JSON file {src} to {workflow_dir}")
            except Exception as e:
                logging.error(f"Failed to copy JSON file {json_path}: {e}")

    # load plugin
    plugin_dir = Path(args.resources) / "plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    if args.plugin != []:
        for plugin_path in args.plugin:
            import shutil
            shutil.copy(plugin_path, plugin_dir)
    if plugin_dir.exists():
        load_existing_plugins(plugin_dir)

    # dictConfig closes existing handlers globally. Configure HTTP logging
    # before the editor owns its queue sinks, then preserve both lifecycles.
    from logging.config import dictConfig
    from .request import logging_config
    dictConfig(logging_config(editor=True))
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: NnDeployServer(
        args, jobs, plugins, cancellations, admin_token=token,
        database_timeout_seconds=args.database_timeout_seconds,
        editor_history_retention_seconds=args.editor_history_retention_seconds,
        database_url=database_url, storage_config=storage_config, redis_config=redis_config,
        auth_config=auth_config, audit_config=audit_config, cors_origins=args.cors_origin,
        max_sync_connections=args.max_sync_connections, device_limits=device_limits))
    original_lifespan = server.app.router.lifespan_context

    @asynccontextmanager
    async def editor_lifespan(application):
        try:
            async with original_lifespan(application):
                try:
                    yield
                finally:
                    await asyncio.to_thread(runtime.close)
        finally:
            # Also close on ASGI startup failure, before Uvicorn re-raises SIGTERM.
            await asyncio.to_thread(runtime.close)

    server.app.router.lifespan_context = editor_lifespan
    try:
        server.run(host=args.host, port=args.port, loop="asyncio", log_config=None)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: shutting down...")
    finally:
        runtime.close()

if __name__ == "__main__":
    main()

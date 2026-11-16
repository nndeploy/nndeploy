import gc
import logging
import os
import threading
import traceback
import pickle
from pathlib import Path
from .logging_taskid import EditorQueueHandler
from .executor import GraphExecutor
from queue import Empty
from .task_queue import ExecutionStatus
import nndeploy
from nndeploy.dag.node import add_global_import_lib, import_global_import_lib

from .logging_taskid import (
    install_taskid_logrecord_factory,
    redirect_python_stdio,
    redirect_fd_to_logger_once,
    set_task_id_fallback,
    reset_task_id,
)

# import resource
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    _PROC = None

import ctypes, ctypes.util
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c"))
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except Exception:
    _malloc_trim = None

def malloc_trim():
    if _malloc_trim:
        try:
            _malloc_trim(0)
        except Exception:
            pass

PROGRESS_INTERVAL_SEC = 0.5

def configure_worker_logger(log_q):
    """
      - root → QueueHandler(log_q)
      - Python stdout/err → logging(print/traceback)
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(EditorQueueHandler(log_q))

    redirect_python_stdio()
    return root

def load_existing_plugins(plugin_dir: Path):
    for f in plugin_dir.iterdir():
        if f.suffix in {".py", ".so"} and f.is_file():
            add_global_import_lib(str(f.resolve()))
    import_global_import_lib()

def poll_plugin_updates(plugin_update_q, resources):
    # Legacy callers can notify paths outside resources/plugin. Bound their
    # drain too, so a continuous producer cannot starve the next Graph.
    paths = set()
    for _ in range(64):
        if plugin_update_q is None:
            break
        try:
            paths.add(plugin_update_q.get_nowait())
        except Empty:
            break
    for plugin_path in paths:
        try:
            if os.path.exists(plugin_path):
                add_global_import_lib(plugin_path)
            else:
                logging.warning("[Plugin] Plugin path not found: %s", plugin_path)
        except Exception:
            logging.exception("[Plugin] Import failed for: %s", plugin_path)
    # Always scan, even if hints were coalesced or IPC was replaced. Import once
    # after collecting paths instead of once per duplicate notification.
    plugin_dir = Path(resources) / "plugin"
    if plugin_dir.exists():
        load_existing_plugins(plugin_dir)
    else:
        import_global_import_lib()


def ensure_picklable(obj, default=None):
    from .editor_outputs import EditorResults, memory_outputs
    preview = memory_outputs(obj)
    try:
        pickle.dumps(obj)
        return EditorResults(obj, preview)
    except Exception:
        return EditorResults({}, preview)

def watch_editor_parent():
    """Stop an orphaned editor child when its owning process exits.

    The multiprocessing sentinel identifies the original parent, avoiding PID
    reuse and heartbeat guesses. This thread is not a native hard deadline:
    recovery must still verify physical exit before releasing any ownership.
    """
    import multiprocessing
    from multiprocessing.connection import wait
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    if not parent.is_alive():
        os._exit(1)

    def observe():
        wait([parent.sentinel])
        os._exit(1)

    threading.Thread(name='EditorParentMonitor', target=observe, daemon=True).start()


def run(task_q, result_q, progress_q, log_q, plugin_update_q, cancel_event_q, resources, *, reader=None, storage_options=None) -> None:
    watch_editor_parent()
    install_taskid_logrecord_factory()

    configure_worker_logger(log_q)
    redirect_fd_to_logger_once()

    executor = GraphExecutor(resources)
    logging.info("Worker PID=%s started", os.getpid())

    pid = os.getpid()
    pending_cancellations = set()

    while True:
        try:
            item = task_q.get(timeout=1.0, **({'reader': reader} if reader is not None else {}))
        except Empty:
            continue

        from contextlib import ExitStack
        with ExitStack() as ownership:
            guard_path = item[1].get('_editor_guard')
            if guard_path is not None:
                from .control_guard import ControlGuard
                guard = ControlGuard.join(guard_path)
                ownership.callback(guard.close)
            if reader is not None and not task_q.claim(item[0], reader):
                continue
            idx, payload = item
            task_id = payload["id"]

            def cancellation_requested():
                if reader is not None:
                    return task_q.cancellation_requested(idx, reader)
                # Legacy callers may still pass ordinary multiprocessing queues.
                # Preserve cancellation for other queued tasks instead of dropping it.
                for _ in range(1024):
                    try:
                        pending_cancellations.add(cancel_event_q.get_nowait())
                    except Empty:
                        break
                return task_id in pending_cancellations

            try:
                progress_q.put_nowait((idx, task_id, {"event": "started", "pid": pid}))
            except Exception:
                pass

            results = {}
            result_holder = {}
            done_evt = threading.Event()
            cancel_requested = False
            execution = payload.get('_editor_execution', {}).get('execution', {})
            managed = bool(payload.get('_editor_execution'))
            budgeted = bool(execution.get('devices') or storage_options and storage_options.get('devices'))
            device_run = None

            def _exec():
                nonlocal device_run
                token = set_task_id_fallback(task_id)
                try:
                    if cancellation_requested():
                        result_holder['cancelled'] = True
                        return
                    if budgeted:
                        from .devices import EditorDeviceRun
                        device_run = EditorDeviceRun(payload, storage_options)
                        device_run.acquire(cancellation_requested)
                    if managed:
                        if reader is None:
                            raise RuntimeError('Configured editor Run requires the supervised Worker queue')
                        if not task_q.begin_execution(idx, reader):
                            raise RuntimeError('Editor execution is no longer available')
                    cpu = None
                    if execution.get('cpu_threads') is not None:
                        from .cpu import CpuThreads
                        cpu = CpuThreads(execution['cpu_threads'])
                    poll_plugin_updates(plugin_update_q, resources)
                    if cancellation_requested():
                        result_holder['cancelled'] = True
                        return
                    from .editor_inputs import prepare_worker_inputs
                    graph_json = prepare_worker_inputs(payload, storage_options, resources,
                        retain=lambda lease: result_holder.update(asset_lease=lease))
                    if cancellation_requested():
                        result_holder['cancelled'] = True
                        return
                    tp_map, results, status, msg = executor.execute(graph_json, task_id,
                                                                   **({'cpu': cpu} if cpu is not None else {}))
                    result_holder["tp_map"] = tp_map
                    result_holder["results"] = results
                    result_holder["status"] = status
                    result_holder["msg"] = msg
                    if status == nndeploy.base.StatusCode.Ok:
                        if cancellation_requested():
                            result_holder['cancelled'] = True
                            return
                        try:
                            from .editor_outputs import capture_worker_files
                            capture_worker_files(payload, storage_options)
                        except Exception:
                            logging.exception('Editor output file publication failed')
                            result_holder['publication_failed'] = True
                        if cancellation_requested():
                            result_holder['cancelled'] = True
                except Exception as e:
                    if budgeted and getattr(e, 'code', None) == 'task_cancelled':
                        result_holder['cancelled'] = True
                    result_holder["error"] = e
                    result_holder["trace"] = traceback.format_exc()
                    result_holder["status"] = None
                    result_holder["msg"] = str(e)
                finally:
                    reset_task_id(token)
                    done_evt.set()

            t = threading.Thread(name=f"Exec-{task_id}", target=_exec, daemon=True)
            t.start()

            while not done_evt.wait(timeout=PROGRESS_INTERVAL_SEC):
                try:
                    if cancellation_requested():
                        if not cancel_requested:
                            logging.info("[Worker] task %s: cancel requested", task_id)
                        cancel_requested = True
                        executor.interrupt_running()
                except Exception as e:
                    logging.warning(f"check cancel signal failed: {e}")

                try:
                    status_dict = executor.runner.get_run_status()
                except Exception as e:
                    status_dict = {"error": str(e)}

                try:
                    progress_q.put_nowait((idx, task_id, {"event": "progress", "pid": pid, "status": status_dict}))
                except Exception:
                    pass

            t.join()

            try:
                status_dict = executor.runner.get_run_status()
            except Exception as e:
                status_dict = {"error": str(e)}

            try:
                progress_q.put_nowait((idx, task_id, {"event": "finished", "pid": pid, "status": status_dict}))
            except Exception:
                pass

            try:
                executor.runner.release()
                if result_holder.get('asset_lease') is not None:
                    result_holder['asset_lease'].close()
            except Exception:
                logging.warning("Graph release failed", exc_info=True)

            # memory reclamation
            malloc_trim()

            if cancel_requested or result_holder.get('cancelled'):
                results, time_profiler_map = {}, {}
                status = ExecutionStatus(False, 'Execution cancelled', 'cancelled')
            elif result_holder.get('publication_failed'):
                results, time_profiler_map = {}, {}
                status = ExecutionStatus(False, 'Output files could not be saved')
            elif "error" in result_holder:
                time_profiler_map = {}
                logging.error("Run failed: %s\n%s", result_holder["error"], result_holder.get("trace", ""))
                status = ExecutionStatus(False, str(result_holder["error"]))
            else:
                status_code = result_holder["status"]
                if status_code != nndeploy.base.StatusCode.Ok:
                    time_profiler_map = {}
                    msg = result_holder["msg"]
                    results = {}
                    status = ExecutionStatus(False, f"Run failed {msg}")
                else:
                    time_profiler_map = result_holder["tp_map"]
                    total = time_profiler_map.get("run_time", 0.0)
                    msg = result_holder["msg"]
                    results = ensure_picklable(result_holder["results"])
                    status = ExecutionStatus(True, f"Run success {total:.2f} ms, {msg}")


            result_holder.pop("results", None)
            gc.collect()
            pending_cancellations.discard(task_id)
            # The final snapshot must travel with the durable result: a short
            # graph can finish before the browser receives any progress frame.
            if isinstance(status_dict, dict) and 'error' not in status_dict:
                time_profiler_map['nodes'] = status_dict
            if reader is not None and not task_q.finish(idx, reader, status, results, time_profiler_map):
                return
            result_q.put((idx, status, results, time_profiler_map))
            if (budgeted or execution.get('execution_timeout_seconds') is not None
                    or execution.get('cpu_threads') is not None):
                if device_run is not None:
                    device_run.close()
                # A one-shot editor Graph may retain native buffers even after
                # deinit. Only process exit proves all its device use ended.
                # It also prevents process-global CPU settings and imported
                # plugins from leaking into the next workflow's execution.
                return


def _remember_close_failure(close):
    """Repeated cleanup must not turn an unconfirmed release into success."""
    from functools import wraps
    @wraps(close)
    async def checked(self, *args, **kwargs):
        failure = getattr(self, '_close_failure', None)
        if failure is not None:
            raise failure
        try:
            return await close(self, *args, **kwargs)
        except BaseException as exc:
            self._close_failure = exc
            raise
    return checked


class Worker:
    """A resident executor with a dedicated owning thread.

    Queues belong to named entries; a blocked worker never owns another entry's
    execution thread. Native work outlives cancellation of an HTTP waiter.
    """

    def __init__(self, source, config, storage, workdir):
        from concurrent.futures import ThreadPoolExecutor
        from .executor import Executor
        self.executor = Executor(source, config, storage, workdir)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'nndeploy-{config.name}')
        self.closed = False

    @property
    def config(self):
        return self.executor.config

    @property
    def resource_snapshot(self):
        return self.executor.source.get('resource_snapshot')

    @property
    def asset_paths(self):
        lease = self.executor.asset_lease
        return lease.paths if lease is not None else []

    @property
    def pid(self):
        return os.getpid()

    async def start(self, *, deadline=None):
        import asyncio
        try:
            await asyncio.get_running_loop().run_in_executor(self.pool, self.executor.open, deadline)
        except BaseException:
            await self.close()
            raise

    async def deploy(self, values, emit=None):
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(self.pool, self.executor.deploy, values, emit)

    @_remember_close_failure
    async def close(self):
        import asyncio
        if self.closed:
            return
        self.closed = True
        try:
            await asyncio.get_running_loop().run_in_executor(self.pool, self.executor.close)
        finally:
            self.pool.shutdown(wait=True)


class EngineWorker:
    """One resident async engine, one owning loop, bounded concurrent requests."""
    def __init__(self, source, config, storage, workdir):
        from concurrent.futures import Future, ThreadPoolExecutor
        self.source, self.config, self.storage, self.workdir = source, config, storage, workdir
        self.loop = self.executor = self.thread = None
        self.ready = Future()
        self.jobs = {}
        self.closed = False
        self.event_pool = ThreadPoolExecutor(max_workers=config.execution.max_concurrency_per_worker,
                                             thread_name_prefix=f'nndeploy-engine-events-{config.name}')

    @property
    def pid(self):
        return os.getpid()

    def _run(self):
        import asyncio
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.ready.set_result(None)
            self.loop.run_forever()
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            raise
        finally:
            if self.loop is not None:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.run_until_complete(self.loop.shutdown_default_executor())
                self.loop.close()
                asyncio.set_event_loop(None)

    async def _initialize(self, deadline=None):
        import asyncio
        import time
        from .executor import Executor
        from .errors import ServerError
        self.executor = Executor(self.source, self.config, self.storage, self.workdir)
        async def initialize():
            await self.executor.open_async()
            for values in self.executor.warmup_requests():
                await self._deploy(values, lambda value: None)
        deadline = min(deadline if deadline is not None else float('inf'),
                       time.monotonic() + self.config.execution.startup_timeout_seconds)
        try:
            await asyncio.wait_for(initialize(), max(0, deadline - time.monotonic()))
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise ServerError(504, 'startup_timeout', 'Worker initialization and warmup exceeded the startup deadline') from exc
            raise
        self.executor._check_startup_deadline(deadline)

    async def start(self, *, deadline=None):
        import asyncio
        self.thread = threading.Thread(target=self._run, name=f'nndeploy-engine-{self.config.name}', daemon=True)
        self.thread.start()
        try:
            await asyncio.wrap_future(self.ready)
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._initialize(deadline), self.loop))
        except BaseException:
            await self.close()
            raise

    def _interrupt(self, job, error):
        if job['failure'] is None:
            job['failure'] = error
        if self.config.execution.cooperative_cancel and not job['interrupted']:
            job['interrupted'] = True
            job['task'].cancel()

    async def _request_cancel(self, identity):
        from .errors import ServerError
        job = self.jobs.get(identity)
        if job is not None:
            self._interrupt(job, ServerError(409, 'task_cancelled', 'Engine request cancellation requested', task_id=identity))

    def cancel(self, identity):
        import asyncio
        if self.loop is not None and not self.closed:
            future = asyncio.run_coroutine_threadsafe(self._request_cancel(identity), self.loop)
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)

    async def _deploy(self, values, emit):
        import asyncio
        import json
        import time
        from .db import encode
        from .errors import ServerError
        from .logging_taskid import set_task_id, reset_task_id, set_request_id, reset_request_id
        identity = values['_task_id']
        if len(self.jobs) >= self.config.execution.max_concurrency_per_worker:
            raise ServerError(429, 'engine_full', 'Engine request capacity reached', retryable=True)
        job = dict(task=asyncio.current_task(), failure=None, interrupted=False)
        self.jobs[identity] = job
        context = set_task_id(identity)
        request_context = set_request_id(values.get('_request_id'))
        limits = self.config.streaming
        timeout = self.config.execution.execution_timeout_seconds
        if limits.enabled:
            timeout = min(timeout, limits.max_duration_seconds) if timeout else limits.max_duration_seconds
        deadline = time.monotonic() + timeout if timeout else None
        timer = self.loop.call_later(timeout, self._interrupt, job,
            ServerError(504, 'execution_timeout', 'Engine request exceeded its duration', task_id=identity)) if timeout else None
        count = 0

        async def publish(value):
            nonlocal count
            if job['failure'] is not None:
                # Without a declared cancellation guarantee, drain actual engine
                # computation while retaining its slot; suppress further output.
                return None
            try:
                count += 1
                if count > limits.max_events:
                    raise ServerError(422, 'stream_event_limit', 'Stream event count exceeds the configured limit')
                data = encode(await self.executor.encode_output_async(value, values))
                if len(data.encode('utf-8')) > limits.max_event_bytes:
                    raise ServerError(422, 'stream_event_too_large', 'Stream event exceeds the configured limit')
                result = json.loads(data)
                pending = self.loop.run_in_executor(self.event_pool, emit, result)
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    # Do not accumulate more writers by releasing this slot while
                    # a prior write remains blocked in a thread.
                    await pending
                    raise
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure = exc if isinstance(exc, ServerError) else ServerError(500, 'event_recording_failed', 'Could not record engine output')
                job['failure'] = job['failure'] or failure
                if self.config.execution.cooperative_cancel:
                    # Closing the iterator is already the cancellation path;
                    # another timer/request must not interrupt its cleanup.
                    job['interrupted'] = True
                    raise job['failure']
                return None

        try:
            result = await self._execute(values, publish)
            if deadline is not None and time.monotonic() >= deadline and job['failure'] is None:
                job['failure'] = ServerError(504, 'execution_timeout', 'Engine request exceeded its duration', task_id=identity)
            if job['failure'] is not None:
                raise job['failure']
            return result
        except asyncio.CancelledError as exc:
            raise job['failure'] or ServerError(409, 'task_cancelled', 'Engine request stopped', task_id=identity) from exc
        except Exception as exc:
            if job['failure'] is not None and exc is not job['failure']:
                raise job['failure'] from exc
            raise
        finally:
            if timer is not None:
                timer.cancel()
            self.jobs.pop(identity, None)
            reset_request_id(request_context)
            reset_task_id(context)

    async def _execute(self, values, publish):
        return await self.executor.deploy_async(values, publish)

    async def deploy(self, values, emit=None):
        import asyncio
        future = asyncio.run_coroutine_threadsafe(self._deploy(values, emit), self.loop)
        return await asyncio.shield(asyncio.wrap_future(future))

    async def _shutdown(self):
        import asyncio
        if self.jobs:
            await asyncio.gather(*(job['task'] for job in list(self.jobs.values())), return_exceptions=True)
        if self.executor is not None:
            await self.executor.close_async()

    @_remember_close_failure
    async def close(self):
        import asyncio
        if self.closed:
            return
        self.closed = True
        try:
            if self.loop is not None and self.thread.is_alive():
                await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._shutdown(), self.loop))
        finally:
            if self.loop is not None and self.thread.is_alive():
                self.loop.call_soon_threadsafe(self.loop.stop)
                await asyncio.to_thread(self.thread.join)
            await asyncio.to_thread(self.event_pool.shutdown, wait=True)


class ConcurrentWorker(EngineWorker):
    """One explicitly thread-safe handler, with bounded execution threads."""
    def __init__(self, source, config, storage, workdir):
        from concurrent.futures import ThreadPoolExecutor
        super().__init__(source, config, storage, workdir)
        self.compute_pool = ThreadPoolExecutor(max_workers=config.execution.max_concurrency_per_worker,
                                               thread_name_prefix=f'nndeploy-concurrent-{config.name}')

    async def _execute(self, values, publish):
        import asyncio
        import contextvars
        def emit(value):
            return asyncio.run_coroutine_threadsafe(publish(value), self.loop).result()
        context = contextvars.copy_context()
        pending = self.loop.run_in_executor(self.compute_pool, context.run, self.executor.deploy, values, emit)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # An asyncio cancellation cannot stop an executing native thread.
            await pending
            raise

    @_remember_close_failure
    async def close(self):
        import asyncio
        try:
            await super().close()
        finally:
            await asyncio.to_thread(self.compute_pool.shutdown, wait=True)


def _resident_process(connection, source_bytes, config, storage_options, workdir, control_guard_path=None, device_guard_path=None, startup_deadline=None):
    """Construct and destroy native objects in the child's owning thread."""
    from .config import ServerConfig
    from .db import DB
    from .executor import Executor
    from .storage import LocalStorage
    executor = None
    db = None
    guard = None
    device_guard = None
    diagnostic_handler = None
    try:
        from .logging_taskid import configure_resident_logger, close_resident_logger
        diagnostic_handler = configure_resident_logger()
        if device_guard_path is not None:
            from .control_guard import ControlGuard
            device_guard = ControlGuard.join(device_guard_path)
        if control_guard_path is not None:
            from .control_guard import ControlGuard
            guard = ControlGuard.join(control_guard_path)
        settings = ServerConfig.model_validate(config)
        cpu = None
        if settings.execution.cpu_threads is not None:
            from .cpu import CpuThreads
            cpu = CpuThreads(settings.execution.cpu_threads)
        db = DB(storage_options['database_url'])
        storage = LocalStorage(storage_options['root'], db, max_bytes=storage_options['max_bytes'], config=storage_options['config'])
        executor = Executor(pickle.loads(source_bytes), settings, storage, workdir, cpu=cpu)
        executor.open(startup_deadline)
        connection.send(('ready', dict(config=executor.config.model_dump(),
                                       resources=executor.source.get('resource_snapshot'),
                                       assets=executor.asset_lease.paths if executor.asset_lease is not None else [])))
        while True:
            command, payload = connection.recv()
            if command == 'close':
                executor.close()
                executor = None
                connection.send(('closed', None))
                return
            if command != 'deploy':
                raise ValueError('Unknown worker command')
            try:
                connection.send(('result', executor.deploy(payload, lambda value: connection.send(('event', value)))))
            except Exception as exc:
                from .errors import ServerError
                logging.exception('Isolated worker execution failed')
                detail = dict(status=exc.status_code, code=exc.code, message=str(exc)) if isinstance(exc, ServerError) else 'Execution failed in isolated worker'
                connection.send(('error', detail))
    except EOFError:
        pass
    except BaseException as exc:
        logging.exception('Isolated worker failed')
        try:
            from .errors import ServerError
            detail = dict(status=exc.status_code, code=exc.code, message=str(exc)) if isinstance(exc, ServerError) else 'Isolated worker failed to initialize'
            connection.send(('error', detail))
        except (OSError, EOFError):
            pass
    finally:
        if executor is not None:
            executor.close()
        if db is not None:
            db.close()
        if 'storage' in locals():
            storage.close()
        connection.close()
        if guard is not None:
            guard.close()
        if device_guard is not None:
            device_guard.close()
        if diagnostic_handler is not None:
            close_resident_logger(diagnostic_handler)


class ProcessWorker:
    """One bounded IPC command per isolated process, with confirmed termination."""
    def __init__(self, source, config, storage, workdir):
        from concurrent.futures import ThreadPoolExecutor
        if source.get('graph') is not None:
            raise ValueError('Process isolation requires a workflow or graph_factory')
        try:
            self.source_bytes = pickle.dumps(source)
        except (TypeError, pickle.PicklingError, AttributeError) as exc:
            raise ValueError('Process workers require importable functions or factories') from exc
        self.config = config
        self.storage_options = dict(database_url=storage.db.url.render_as_string(hide_password=False),
                                    root=str(storage.root), max_bytes=storage.max_bytes, config=storage.config.model_dump())
        self.workdir = workdir
        self.process = self.connection = None
        self.cancelled = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'nndeploy-ipc-{config.name}')
        self.event_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'nndeploy-events-{config.name}')
        self.pending_event = None
        self.closed = False
        self.resource_snapshot = None
        self.asset_paths = []

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    def _terminate(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _receive(self, timeout, cancellable=False, emit=None):
        import time
        from .errors import ServerError
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancellable and self.cancelled.is_set():
                self._terminate()
                raise ServerError(409, 'task_cancelled', 'Isolated worker stopped')
            if deadline is not None and time.monotonic() >= deadline:
                self._terminate()
                raise ServerError(504, 'execution_timeout', 'Isolated worker exceeded its deadline')
            try:
                if self.connection.poll(.05):
                    kind, value = self.connection.recv()
                    if kind == 'error':
                        if isinstance(value, dict):
                            raise ServerError(value['status'], value['code'], value['message'])
                        raise ServerError(500, 'execution_failed', value)
                    if kind == 'event':
                        if emit is None:
                            raise RuntimeError('Worker emitted an unexpected stream event')
                        from concurrent.futures import wait
                        self.pending_event = self.event_pool.submit(emit, value)
                        while not wait([self.pending_event], timeout=.02).done:
                            if cancellable and self.cancelled.is_set():
                                self._terminate()
                                raise ServerError(409, 'task_cancelled', 'Isolated worker stopped')
                            if deadline is not None and time.monotonic() >= deadline:
                                self._terminate()
                                raise ServerError(504, 'execution_timeout', 'Isolated worker exceeded its deadline')
                        self.pending_event.result()
                        self.pending_event = None
                        continue
                    return kind, value
            except (OSError, EOFError) as exc:
                self._terminate()
                raise ServerError(503, 'worker_crashed', 'Isolated worker exited', retryable=True) from exc
            if not self.process.is_alive():
                self._terminate()
                raise ServerError(503, 'worker_crashed', 'Isolated worker exited', retryable=True)

    def _start(self, deadline=None):
        import multiprocessing
        import time
        from .config import ServerConfig
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_resident_process,
            args=(child, self.source_bytes, self.config.model_dump(), self.storage_options, self.workdir,
                  getattr(self, 'control_guard_path', None), getattr(self, 'device_guard_path', None), deadline), daemon=True)
        try:
            self.process.start()
            child.close()
            from .errors import ServerError
            try:
                remaining = self.config.execution.startup_timeout_seconds if deadline is None else max(0, deadline - time.monotonic())
                kind, value = self._receive(remaining)
            except ServerError as exc:
                if exc.code == 'execution_timeout':
                    raise ServerError(504, 'startup_timeout', 'Worker initialization and warmup exceeded the startup deadline') from exc
                raise
            if kind != 'ready':
                raise RuntimeError('Worker did not become ready')
            self.config = ServerConfig.model_validate(value['config'])
            self.resource_snapshot = value['resources']
            self.asset_paths = value['assets']
            if self.resource_snapshot is not None:
                source = pickle.loads(self.source_bytes)
                source['resource_snapshot'] = self.resource_snapshot
                self.source_bytes = pickle.dumps(source)
        except BaseException:
            child.close()
            self._terminate()
            raise

    async def start(self, *, deadline=None):
        import asyncio
        try:
            await asyncio.get_running_loop().run_in_executor(self.pool, self._start, deadline)
        except BaseException:
            await self.close()
            raise

    def _deploy(self, values, emit=None):
        if self.pending_event is not None:
            if not self.pending_event.done():
                from .errors import ServerError
                raise ServerError(503, 'recording_unavailable', 'Previous event write is still being reconciled', retryable=True)
            # A timed-out event writer may have completed after task finalization.
            # Its DB conditional write rejects stale events; do not carry it into
            # another request or enqueue additional blocked writers.
            self.pending_event.exception()
            self.pending_event = None
        if self.process is None or not self.process.is_alive():
            self._terminate()
            self._start()
        try:
            self.connection.send(('deploy', values))
            timeout = self.config.execution.execution_timeout_seconds
            if self.config.streaming.enabled:
                timeout = min(timeout, self.config.streaming.max_duration_seconds) if timeout else self.config.streaming.max_duration_seconds
            kind, value = self._receive(timeout, cancellable=True, emit=emit)
            if kind != 'result':
                raise RuntimeError('Invalid worker response')
            return value
        except (BrokenPipeError, EOFError, OSError) as exc:
            from .errors import ServerError
            self._terminate()
            raise ServerError(503, 'worker_crashed', 'Isolated worker exited', retryable=True) from exc
        except BaseException:
            # A failed event commit must stop the emitting process before the
            # next request can reuse it or release its execution slot.
            self._terminate()
            raise

    async def deploy(self, values, emit=None):
        import asyncio
        self.cancelled.clear()
        return await asyncio.get_running_loop().run_in_executor(self.pool, self._deploy, values, emit)

    def cancel(self, identity=None):
        self.cancelled.set()

    def _close(self):
        try:
            if self.process is not None and self.process.is_alive():
                self.connection.send(('close', None))
                self._receive(5)
        finally:
            self._terminate()
            self.event_pool.shutdown(wait=True)

    @_remember_close_failure
    async def close(self):
        import asyncio
        if self.closed:
            return
        self.closed = True
        try:
            await asyncio.get_running_loop().run_in_executor(self.pool, self._close)
        finally:
            self.pool.shutdown(wait=True)

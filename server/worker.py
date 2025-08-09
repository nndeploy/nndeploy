# worker.py

import gc
import logging
import os
import contextvars
import threading
import traceback
from logging.handlers import QueueHandler
from .executor import GraphExecutor
from queue import Empty
from .task_queue import ExecutionStatus
import nndeploy

# import resource
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    _PROC = None

# def get_rss_mb():
#     if _PROC:
#         return _PROC.memory_info().rss / 1024 / 1024
#     return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

############# meminfo ################
# import threading
# def log_runtime_numbers(logger, progress_q):
#     fds = len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else -1
#     threads = len(threading.enumerate())
#     try:
#         qsize = progress_q.qsize()
#     except Exception:
#         qsize = -1
#     logger.info("[RUNTIME] threads=%d, fds=%d, progress_q=%d", threads, fds, qsize)

# import tracemalloc

# MEM_PROFILE_TOP = int(os.getenv("NNDEPLOY_MEM_PROFILE_TOP", "15"))
# MEM_FILTER_PREFIX = os.getenv("NNDEPLOY_MEM_FILTER", "")
# tracemalloc.start(25)

# class TaskMemProfiler:
#     def __init__(self, tag: str, logger: logging.Logger):
#         self.tag = tag
#         self.logger = logger
#         self.snap0 = tracemalloc.take_snapshot()
#         self.rss0 = get_rss_mb()

#     def report(self):
#         snap1 = tracemalloc.take_snapshot()
#         rss1 = get_rss_mb()
#         if MEM_FILTER_PREFIX:
#             filt = tracemalloc.Filter(True, MEM_FILTER_PREFIX + "*")
#             s0 = self.snap0.filter_traces((filt,))
#             s1 = snap1.filter_traces((filt,))
#         else:
#             s0, s1 = self.snap0, snap1

#         stats = s1.compare_to(s0, 'lineno')[:MEM_PROFILE_TOP]

#         self.logger.info("[MEM][%s] RSS diff: %.2f MB -> %.2f MB (Δ%.2f MB)",
#                          self.tag, self.rss0, rss1, rss1 - self.rss0)

#         for s in stats:
#             size_kb = s.size_diff / 1024
#             count = s.count_diff
#             tb = s.traceback[0]
#             self.logger.info("[MEM][%s] +%.1f KB (%+d objs) %s:%d",
#                              self.tag, size_kb, count, tb.filename, tb.lineno)
######################################

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

_CURRENT_TASK_ID = "0"

# current_task_id_var = contextvars.ContextVar("task_id", default="0")

def set_current_task_id(task_id: str):
    global _CURRENT_TASK_ID
    _CURRENT_TASK_ID = task_id
    # current_task_id_var.set(task_id)

def get_current_task_id() -> str:
    return _CURRENT_TASK_ID

class TaskLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        task_id = get_current_task_id()
        return f"[task_id={task_id}] {msg}", kwargs

# ---- global state ----
_STDIO_REDIRECTED = False

def redirect_fd_to_logger_once(logger):
    global _STDIO_REDIRECTED
    if _STDIO_REDIRECTED:
        return
    for fd, level, label in [(1, logging.INFO, "stdout"),
                             (2, logging.ERROR, "stderr")]:
        r, w = os.pipe()
        os.dup2(w, fd)
        os.close(w)

        def reader():
            with os.fdopen(r, "r", buffering=1) as pr:
                for line in pr:
                    line = line.rstrip()
                    if line:
                        logger.log(level, f"[C++ {label}]{line}")

        t = threading.Thread(target=reader, daemon=True,
                             name=f"FDReader-{label}")
        t.start()
    _STDIO_REDIRECTED = True

def configure_worker_logger(log_q) -> logging.LoggerAdapter:
    import sys

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(QueueHandler(log_q))

    logger = TaskLoggerAdapter(root, {})

    class stream_to_logger_:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
        def write(self, message):
            message = message.strip()
            if message:
                self.logger.log(self.level, message)
        def flush(self): pass

    sys.stdout = stream_to_logger_(logger, logging.INFO)
    sys.stderr = stream_to_logger_(logger, logging.ERROR)

    return logger

def poll_plugin_updates(plugin_update_q):
    from nndeploy.dag.node import add_global_import_lib, import_global_import_lib
    while not plugin_update_q.empty():
        plugin_path = plugin_update_q.get()
        if os.path.exists(plugin_path):
            add_global_import_lib(plugin_path)
            import_global_import_lib()
            logging.info(f"[Plugin] Imported plugin: {plugin_path}")
        else:
            logging.warning(f"[Plugin] Plugin path not found: {plugin_path}")

def run(task_q, result_q, progress_q, log_q, plugin_update_q, resources) -> None:
    logger = configure_worker_logger(log_q)
    redirect_fd_to_logger_once(logger)

    executor = GraphExecutor(resources)
    logging.info("Worker PID=%s started", os.getpid())

    while True:
        try:
            item = task_q.get(timeout=1.0)
        except Empty:
            continue

        poll_plugin_updates(plugin_update_q)

        idx, payload = item
        task_id = payload["id"]

        result_holder = {}
        done_evt = threading.Event()

        def _exec():
            set_current_task_id(task_id)
            try:
                executor.handle_urls(payload["graph_json"])
                tp_map, results, status, msg = executor.execute(payload["graph_json"], task_id)
                result_holder["tp_map"] = tp_map
                result_holder["results"] = results
                result_holder["status"] = status
                result_holder["msg"] = msg
            except Exception as e:
                result_holder["error"] = e
                result_holder["trace"] = traceback.format_exc()
                result_holder["status"] = status
                result_holder["msg"] = str(e)
            finally:
                done_evt.set()
        t = threading.Thread(name=f"Exec-{task_id}", target=_exec, daemon=True)
        t.start()

        while not done_evt.wait(timeout=PROGRESS_INTERVAL_SEC):
            try:
                status_dict = executor.runner.get_run_status()
            except Exception as e:
                status_dict = {"error": str(e)}
            try:
                progress_q.put_nowait((idx, task_id, status_dict))
            except Exception:
                pass

        t.join()

        try:
            status_dict = executor.runner.get_run_status()
        except Exception as e:
            status_dict = {"error": str(e)}
        try:
            progress_q.put_nowait((idx, task_id, status_dict))
        except Exception:
            pass

        try:
            executor.runner.release()
        except Exception:
            logger.warning("Graph release failed", exc_info=True)

        # memory reclamation
        malloc_trim()

        if "error" in result_holder:
            time_profiler_map = {}
            logger.error("Run failed: %s\n%s", result_holder["error"], result_holder.get("trace", ""))
            status = ExecutionStatus(False, str(result_holder["error"]))
        else:
            status = result_holder["status"]
            if status != nndeploy.base.StatusCode.Ok:
                time_profiler_map = {}
                msg = result_holder["msg"]
                status = ExecutionStatus(False, f"Run failed {msg}")
            else:
                time_profiler_map = result_holder["tp_map"]
                sum = time_profiler_map["run_time"]
                msg = result_holder["msg"]
                status = ExecutionStatus(True, f"Run success {sum:.2f} ms, {msg}")

        result_holder.pop("results", None)

        gc.collect()
        set_current_task_id("0")
        result_q.put((idx, status, time_profiler_map))

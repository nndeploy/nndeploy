import gc
import logging
import os
import threading
import traceback
from pathlib import Path
from logging.handlers import QueueHandler
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
    root.addHandler(QueueHandler(log_q))

    redirect_python_stdio()
    return root

def load_existing_plugins(plugin_dir: Path):
    for f in plugin_dir.iterdir():
        if f.suffix in {".py", ".so"} and f.is_file():
            add_global_import_lib(str(f.resolve()))
    import_global_import_lib()

def poll_plugin_updates(plugin_update_q, resources):
    plugin_dir = Path(resources) / "plugin"
    if plugin_dir.exists():
        load_existing_plugins(plugin_dir)
    while not plugin_update_q.empty():
        plugin_path = plugin_update_q.get()
        if os.path.exists(plugin_path):
            add_global_import_lib(plugin_path)
            import_global_import_lib()
            logging.info("[Plugin] Imported plugin: %s", plugin_path)
        else:
            logging.warning("[Plugin] Plugin path not found: %s", plugin_path)

def run(task_q, result_q, progress_q, log_q, plugin_update_q, resources) -> None:
    install_taskid_logrecord_factory()

    configure_worker_logger(log_q)
    redirect_fd_to_logger_once()

    executor = GraphExecutor(resources)
    logging.info("Worker PID=%s started", os.getpid())

    while True:
        try:
            item = task_q.get(timeout=1.0)
        except Empty:
            continue

        poll_plugin_updates(plugin_update_q, resources)

        idx, payload = item
        task_id = payload["id"]

        result_holder = {}
        done_evt = threading.Event()

        def _exec():
            token = set_task_id_fallback(task_id)
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
                result_holder["status"] = None
                result_holder["msg"] = str(e)
            finally:
                reset_task_id(token)
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
            logging.warning("Graph release failed", exc_info=True)

        # memory reclamation
        malloc_trim()

        if "error" in result_holder:
            time_profiler_map = {}
            logging.error("Run failed: %s\n%s", result_holder["error"], result_holder.get("trace", ""))
            status = ExecutionStatus(False, str(result_holder["error"]))
        else:
            status_code = result_holder["status"]
            if status_code != nndeploy.base.StatusCode.Ok:
                time_profiler_map = {}
                msg = result_holder["msg"]
                status = ExecutionStatus(False, f"Run failed {msg}")
            else:
                time_profiler_map = result_holder["tp_map"]
                total = time_profiler_map.get("run_time", 0.0)
                msg = result_holder["msg"]
                status = ExecutionStatus(True, f"Run success {total:.2f} ms, {msg}")

        result_holder.pop("results", None)

        gc.collect()
        result_q.put((idx, status, time_profiler_map))

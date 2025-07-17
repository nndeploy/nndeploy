# worker.py

import gc
import logging
import os
import contextvars
import threading
import traceback
from logging.handlers import QueueHandler
from executor import GraphExecutor
from queue import Empty
from task_queue import ExecutionStatus

PROGRESS_INTERVAL_SEC = 0.5

current_task_id_var = contextvars.ContextVar("task_id", default="0")

def set_current_task_id(task_id: str):
    current_task_id_var.set(task_id)

def get_current_task_id() -> str:
    return current_task_id_var.get()

class TaskLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        task_id = get_current_task_id()
        return f"[task_id={task_id}] {msg}", kwargs

def redirect_fd_to_logger(fd, level, label, logger):
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        ctx = contextvars.copy_context()

        def reader():
            with os.fdopen(read_fd, "r") as pipe_reader:
                for line in pipe_reader:
                    line = line.strip()
                    if line:
                        logger.log(level, f"[C++ {label}] {line}")
        threading.Thread(target=lambda: ctx.run(reader), daemon=True).start()

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

def run(task_q, result_q, progress_q, log_q) -> None:
    logger = configure_worker_logger(log_q)

    executor = GraphExecutor()

    while True:
        try:
            item = task_q.get(timeout=1.0)
        except Empty:
            continue
        logging.info("Worker PID=%s started", os.getpid())

        idx, payload = item
        task_id = payload["id"]

        result_holder = {}
        done_evt = threading.Event()

        def _exec():
            set_current_task_id(task_id)
            redirect_fd_to_logger(1, logging.INFO, "stdout", logger)
            redirect_fd_to_logger(2, logging.ERROR, "stderr", logger)
            try:
                tp_map, results = executor.execute(payload["graph_json"], task_id)
                result_holder["tp_map"] = tp_map
                result_holder["results"] = results
            except Exception as e:
                result_holder["error"] = e
                result_holder["trace"] = traceback.format_exc()
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

        if "error" in result_holder:
            logging.error("Run failed: %s\n%s", result_holder["error"], result_holder.get("trace", ""))
            status = ExecutionStatus(False, str(result_holder["error"]))
        else:
            time_profiler_map = result_holder["tp_map"]
            sum = time_profiler_map["sum_" + payload["graph_json"]["name_"]]
            status = ExecutionStatus(True, f"{sum:.2f}s")
            logging.info("Task %s done in %.2fs", task_id, sum)
        gc.collect()
        result_q.put((idx, status))

# worker.py

import gc
import logging
import time
import os
import traceback
from logging.handlers import QueueHandler
from executor import GraphExecutor
from task_queue import TaskQueue, ExecutionStatus

def configure_worker_logger(log_q) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(QueueHandler(log_q))

def run(task_q, log_q) -> None:
    configure_worker_logger(log_q)

    logging.info("Worker PID=%s started", os.getpid())
    executor = GraphExecutor()

    wrapped_q = TaskQueue(task_q)

    while True:
        try:
            item = wrapped_q.get(timeout=1.0)
            if item is None:
                continue

            idx, payload = item
            task_id = payload["id"]

            elapsed = executor.execute(payload["graph_json"], task_id)
            wrapped_q.task_done(idx, ExecutionStatus(True, f"{elapsed:.2f}s"))
            logging.info("Task %s done in %.2fs", task_id, elapsed)
        except Exception as e:
            logging.error("Run failed: %s\n%s", e, traceback.format_exc())
            wrapped_q.task_done(idx, ExecutionStatus(False, str(e)))
        finally:
            gc.collect()

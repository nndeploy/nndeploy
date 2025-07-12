# worker.py

import gc
import logging
import os
import traceback
from logging.handlers import QueueHandler
from executor import GraphExecutor
from queue import Empty
from task_queue import ExecutionStatus

def configure_worker_logger(log_q) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(QueueHandler(log_q))

def run(task_q, result_q, log_q) -> None:
    configure_worker_logger(log_q)

    executor = GraphExecutor()

    while True:
        try:
            item = task_q.get(timeout=1.0)
        except Empty:
            continue
        logging.info("Worker PID=%s started", os.getpid())

        idx, payload = item
        task_id = payload["id"]
        try:
            time_profiler_map, results = executor.execute(payload["graph_json"], task_id)
            sum = time_profiler_map["sum_" + payload["graph_json"]["name_"]]
            status = ExecutionStatus(True, f"{sum:.2f}s")
            logging.info("Task %s done in %.2fs", task_id, sum)
        except Exception as e:
            logging.error("Run failed: %s\n%s", e, traceback.format_exc())
            status = ExecutionStatus(False, str(e))
        finally:
            gc.collect()
        result_q.put((idx, status))

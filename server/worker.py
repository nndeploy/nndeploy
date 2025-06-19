# worker.py

import gc
import logging
import time
import traceback
from executor import GraphExecutor
from task_queue import TaskQueue, ExecutionStatus

def run(server, q: TaskQueue):
    ex = GraphExecutor(server)
    while True:
        item = q.get(timeout=1.0)
        if item is None:
            continue
        idx, payload = item
        task_id = payload["id"]

        try:
            elapsed = ex.execute(payload["graph_json"], task_id)
            q.task_done(idx, ExecutionStatus(True, f"{elapsed:.2f}s"))
            logging.info("Task %s done in %.2fs", task_id, elapsed)
        except Exception as e:
            logging.error("Run failed: %s\n%s", e, traceback.format_exc())
            q.task_done(idx, ExecutionStatus(False, str(e)))
        finally:
            gc.collect()


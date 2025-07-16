# worker.py

import gc
import logging
import os
import threading
import traceback
from logging.handlers import QueueHandler
from executor import GraphExecutor
from queue import Empty
from task_queue import ExecutionStatus

PROGRESS_INTERVAL_SEC = 0.5

def configure_worker_logger(log_q) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(QueueHandler(log_q))

def run(task_q, result_q, progress_q, log_q) -> None:
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

        result_holder = {}
        done_evt = threading.Event()

        def _exec():
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

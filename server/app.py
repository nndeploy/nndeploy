# app.py

import argparse
import asyncio
import threading
import logging
import os
import multiprocessing as mp
import uvicorn
import sys
import time
from pathlib import Path
from typing import Tuple
from task_queue import TaskQueue
from server import NnDeployServer
from worker import run as worker_run
from logging.handlers import QueueHandler, QueueListener

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--resources", default="./resources")
    ap.add_argument("--log", default="./logs/nndeploy_server.log")
    return ap.parse_args()

def configure_root_logger(log_q: mp.Queue, log_file: str) -> QueueListener:
    log_fmt = "%(asctime)s %(processName)s %(levelname)s %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    root.handlers.clear()
    root.addHandler(QueueHandler(log_q))

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(log_fmt))
    
    listener = QueueListener(log_q, *handlers, respect_handler_level=True)
    listener.start()
    return listener

def start_worker(
        task_q: "mp.queues.Queue",
        result_q: "mp.queues.Queue",
        log_q: "mp.queues.Queue") -> mp.Process:
    p = mp.Process(
        target=worker_run,
        name="WorkerProcess",
        args=(task_q, result_q, log_q),
        daemon=True,
    )
    p.start()
    logging.info("Worker spawned, pid=%s", p.pid)
    return p

def monitor_worker(
    worker: mp.Process,
    task_q: "mp.queues.Queue",
    result_q: "mp.queues.Queue",
    log_q: "mp.queues.Queue",
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        if not worker.is_alive():
            logging.error(
                "Worker died (exitcode=%s). Restarting in 2 seconds...", worker.exitcode
            )
            time.sleep(2)
            worker = start_worker(task_q, result_q, log_q)
        time.sleep(1)

def start_scheduler(queue: TaskQueue, job_q: mp.Queue):
    def _loop():
        while True:
            item = queue.get(timeout=None)  # blocks until a job is ready
            if item is None:
                continue  # shouldn't happen with timeout=None
            idx, payload = item
            job_q.put((idx, payload))

    th = threading.Thread(name="SchedulerThread", target=_loop, daemon=True)
    th.start()

def start_finisher(queue: TaskQueue, result_q: mp.Queue):
    def _loop():
        while True:
            idx, status = result_q.get()  # blocks
            queue.task_done(idx, status)

    th = threading.Thread(name="FinisherThread", target=_loop, daemon=True)
    th.start()

def main() -> None:
    mp.set_start_method("spawn", force=True)

    args = cli()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    job_mp_queue: mp.Queue = mp.Queue(maxsize=256)     # main ➜ worker
    result_q: mp.Queue = mp.Queue(maxsize=256)         # worker ➜ main
    log_q: mp.Queue = mp.Queue(-1)                     # all ➜ logger

    listener = configure_root_logger(log_q, args.log)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server = NnDeployServer(loop, args, job_mp_queue)

    start_scheduler(server.queue, job_mp_queue)
    start_finisher(server.queue, result_q)

    worker = start_worker(job_mp_queue, result_q, log_q)
    stop_event = threading.Event()
    monitor_t = threading.Thread(
        target=monitor_worker,
        args=(worker, job_mp_queue, result_q, log_q, stop_event),
        daemon=True,
    )
    monitor_t.start()

    try:
        uvicorn.run(server.app, host=args.host, port=args.port, loop="asyncio")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: shutting down...")
    finally:
        stop_event.set()
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=3)

        listener.stop()

if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()

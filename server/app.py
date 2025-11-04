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
from logging.handlers import QueueHandler, QueueListener
from nndeploy.dag.node import add_global_import_lib, import_global_import_lib

from .task_queue import TaskQueue
from .server import NnDeployServer
from .worker import run as worker_run
from .log_broadcast import LogBroadcaster
from .logging_taskid import install_taskid_logrecord_factory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resources", default="./resources")
    ap.add_argument("--log", default="./logs/nndeploy_server.log")
    ap.add_argument("--front-end-version", default="!", help="GitHub frontend, as owner/repo@0.0.1 or @latest,default latest")
    ap.add_argument("--debug", dest="debug", action="store_true",
                    help="enable debug mode")
    ap.add_argument("--no-debug", dest="debug", action="store_false",
                    help="disable debug mode")
    ap.add_argument("--plugin", type=str, nargs='*', default=[], required=False)
    ap.add_argument(
        "--json_file",
        type=str,
        nargs='*',
        default=[],
        help="Path(s) to one or more JSON files to copy into resources/workflow directory"
    )
    ap.set_defaults(debug=False)
    return ap.parse_args()

def configure_root_logger(log_q: mp.Queue, log_file: str, server) -> QueueListener:
    log_fmt = "%(asctime)s %(processName)s %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(QueueHandler(log_q))

    log_broadcaster = LogBroadcaster(
        get_loop=lambda: server.loop,
        get_ws_map=lambda: server.task_ws_map,
    )

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        ),
        log_broadcaster
    ]
    for h in handlers:
        h.setFormatter(logging.Formatter(log_fmt))
    
    listener = QueueListener(log_q, *handlers, respect_handler_level=True)
    listener.start()
    return listener

def start_worker(
        task_q: "mp.queues.Queue",
        result_q: "mp.queues.Queue",
        progress_q: "mp.queues.Queue",
        log_q: "mp.queues.Queue",
        plugin_update_q: "mp.queues.Queue",
        cancel_event_q: "mp.queues.Queue",
        resources) -> mp.Process:
    p = mp.Process(
        target=worker_run,
        name="WorkerProcess",
        args=(task_q, result_q, progress_q, log_q, plugin_update_q, cancel_event_q, resources),
        daemon=True,
    )
    p.start()
    logging.info("Worker spawned, pid=%s", p.pid)
    return p

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
) -> None:
    restart_count = 0
    while not stop_event.is_set():
        if not worker.is_alive():
            logging.error(
                "Worker died (exitcode=%s). Restarting in 2 seconds...", worker.exitcode
            )
            try:
                server.notify_system_event(
                    "worker_died",
                    {"exitcode": worker.exitcode, "restart_in_sec": 2}
                )
            except Exception:
                logging.exception("[monitor_worker] notify worker_died failed")
            time.sleep(2)
            logging.info("Worker restarted, new pid=%s (restart_count=%s)", worker.pid, restart_count)
            worker = start_worker(task_q, result_q, progress_q, log_q, plugin_update_q, cancel_event_q, resources)
            restart_count += 1
            try:
                server.notify_system_event(
                    "worker_restarted",
                    {"pid": worker.pid}
                )
            except Exception:
                logging.exception("[monitor_worker] notify worker_restarted failed")
        time.sleep(1)

def start_scheduler(queue: TaskQueue, job_q: mp.Queue):
    def _loop():
        while True:
            item = queue.get(timeout=None)  # blocks until a job is ready
            if item is None:
                continue  # shouldn't happen with timeout=None
            idx, payload = item
            job_q.put((idx, payload))
            queue.mark_dispatched(idx)

    th = threading.Thread(name="SchedulerThread", target=_loop, daemon=True)
    th.start()

def start_finisher(queue: TaskQueue, result_q: mp.Queue):
    def _loop():
        while True:
            idx, status, results, time_profile_map = result_q.get()  # blocks
            queue.task_done(idx, status, results, time_profile_map)

    th = threading.Thread(name="FinisherThread", target=_loop, daemon=True)
    th.start()

def start_progress_listener(server: NnDeployServer, progress_q: mp.Queue):
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
        while True:
            try:
                idx, task_id, result = progress_q.get()
            except Exception as err:
                logging.error("[ProgressThread] get failed: %s", err)
                continue
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

def load_existing_plugins(plugin_dir: Path):
    for f in plugin_dir.iterdir():
        if f.suffix in {".py", ".so"} and f.is_file():
            add_global_import_lib(str(f.resolve()))
    import_global_import_lib()

def main() -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    
    mp.set_start_method("spawn", force=True)

    args = cli()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

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

    # multi processing message queue
    job_mp_queue: mp.Queue = mp.Queue(maxsize=256)     # main ➜ worker
    result_q: mp.Queue = mp.Queue(maxsize=256)         # worker ➜ main
    progress_q: mp.Queue = mp.Queue(maxsize=1024)
    log_q: mp.Queue = mp.Queue(-1)                     # all ➜ logger
    plugin_update_q: mp.Queue = mp.Queue()
    cancel_event_queue: mp.Queue = mp.Queue()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # server
    server = NnDeployServer(args, job_mp_queue, plugin_update_q, cancel_event_queue)
    start_scheduler(server.queue, job_mp_queue)
    start_finisher(server.queue, result_q)

    # worker and monitor
    worker = start_worker(job_mp_queue, result_q, progress_q, log_q, plugin_update_q, cancel_event_queue, args.resources)
    stop_event = threading.Event()
    monitor_t = threading.Thread(
        target=monitor_worker,
        args=(worker, job_mp_queue, result_q, progress_q, log_q, plugin_update_q, cancel_event_queue, args.resources, server, stop_event),
        daemon=True,
    )
    monitor_t.start()

    log_listener = configure_root_logger(log_q, args.log, server)

    # progress listener
    start_progress_listener(server, progress_q)

    try:
        uvicorn.run(server.app, host=args.host, port=args.port, loop="asyncio")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: shutting down...")
    finally:
        stop_event.set()
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=3)

        log_listener.stop()

if __name__ == "__main__":
    main()

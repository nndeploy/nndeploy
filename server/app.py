# app.py

import argparse
import asyncio
import threading
import logging
import uvicorn
from server import NndeployServer
from worker import run as worker_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--workdir", default="./runs")
    return ap.parse_args()

def launch():
    args = cli()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server = NndeployServer(loop, args)

    # backend threading queue
    t = threading.Thread(target=worker_run, daemon=True, args=(server, server.queue))
    t.start()

    uvicorn.run(server.app, host=args.host, port=args.port, loop="asyncio")

if __name__ == "__main__":
    launch()

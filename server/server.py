from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import APIRouter, Request, status, UploadFile, File
from fastapi import Depends
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Set, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import os
import json
import asyncio
import uuid
import logging
import nndeploy.dag
from task_queue import TaskQueue
from schemas import (
    EnqueueRequest,
    EnqueueResponse,
    QueueStateResponse,
    NodeListResponse,
    HistoryItem,
    ProgressPayload,
    UploadResponse
)
import files
from files import router as files_router

class NnDeployServer:
    instance: "NnDeployServer" = None

    def __init__(self, loop, args):
        NnDeployServer.instance = self
        self.loop = loop
        self.args = args
        self.app = FastAPI(
            title="nndeploy backend",
            version="0.1.0"
        )
        self.queue = TaskQueue(self)
        self.sockets: set[WebSocket] = set()
        self._register_routes()

    def _get_workdir(self) -> Path:
        return Path(self.args.resources)

    def _register_routes(self):
        api = APIRouter(prefix="/api")

        # commit task
        @api.post(
            "/queue",
            response_model=EnqueueResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="enqueue nndeploy graph task",
        )
        async def enqueue(req: EnqueueRequest):
            task_id = req.id or str(uuid.uuid4())
            payload = {
                "id": task_id,
                "graph_json": req.graph_json,
                "priority": req.priority,
            }
            self.queue.put(payload, prio=req.priority)
            return EnqueueResponse(task_id=task_id)

        # loop
        @api.get(
            "/queue",
            response_model=QueueStateResponse,
            summary="check running / wait task",
        )
        async def queue_state():
            running, pending = self.queue.get_current_queue()
            return QueueStateResponse(running=running, pending=pending)
        
        @api.get(
            "/nodes",
            response_model=NodeListResponse,
            summary="return register nodes",
        )
        async def register_nodes():
            json_str = nndeploy.dag.get_all_node_json()
            data = json.loads(json_str) 
            return NodeListResponse(nodes=data)
        
        # history
        @api.get(
            "/history",
            response_model=Dict[str, HistoryItem],
            summary="check history",
        )
        async def history(max_items: Optional[int] = None):
            return self.queue.get_history(max_items)
        
        # websocket
        @api.websocket("/ws/progress")
        async def ws_progress(ws: WebSocket):
            await ws.accept()
            self.sockets.add(ws)
            try:
                while True:
                    await asyncio.sleep(60)
            except WebSocketDisconnect:
                self.sockets.discard(ws)

        # heartbeat
        @self.app.get("/", tags=["Web"])
        async def root():
            return HTMLResponse("<h2>nndeploy backend: API OK</h2>")

        self.app.include_router(
            files_router,
            dependencies=[Depends(self._get_workdir)]
        )
        self.app.dependency_overrides[files.get_workdir] = self._get_workdir

        self.app.include_router(api)
    
    # queue progress notification
    def queue_updated(self):
        payload = {"pending": len(self.queue.get_current_queue()[1])}
        self.send_sync("queue_update", payload)

    def send_sync(self, event:str, data:dict, ws: WebSocket | None = None):
        self.loop.call_soon_threadsafe(asyncio.create_task, self._broadcast(event, data, ws))
    
    async def _broadcast(self, event, data, ws):
        msg = ProgressPayload(type=event, data=data).model_dump()
        targets = [ws] if ws else list(self.sockets)
        for w in targets:
            try:
                await w.send_json(msg)
            except Exception:
                self.sockets.discard(w)

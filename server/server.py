from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Set, Dict, Any, Optional
import asyncio
import uuid
import logging
from queue import TaskQueue
from schemas import (
    EnqueueRequest,
    EnqueueResponse,
    QueueStateResponse,
    HistoryItem,
    ProgressPayload,
)

class NnDeployServer:
    instance: "NnDeployServer" = None

    def __init__(self, loop, args):
        NnDeployServer.instance = self
        self.loop = loop
        self.args = args
        self.app = FastAPI()
        self.queue = TaskQueue(self)
        self.sockets: set[WebSocket] = set()
        self._register_routes()
    
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

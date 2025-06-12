from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import APIRouter, Request, status, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Set, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
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
    UploadResponse
)

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

    def _register_routes(self):
        """upload help function"""
        def _save(file: UploadFile, subdir: str) -> UploadResponse:
            today = datetime.now().strftime("%Y-%m-%d")
            folder = Path(self.args.workdir, "uploads", today, subdir)
            folder.mkdir(parents=True, exist_ok=True)

            dst = folder / file.filename
            with dst.open("wb") as w:
                w.write(file.file.read())
            return UploadResponse(
                filename=file.filename,
                saved_path=str(dst.relative_to(self.args.workdir)),
                size=dst.stat().st_size,
                uploaded_at=datetime.utcnow(),
            )

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
        
        # upload image
        @api.post(
            "/upload/image",
            response_model=UploadResponse,
            summary="上传图片文件",
            tags=["Upload"],
        )
        async def upload_image(file: UploadFile = File(..., max_length=10 * 1024 * 1024)):
            return _save(file, "images")

        # upload video
        @api.post(
            "/upload/video",
            response_model=UploadResponse,
            summary="上传视频文件",
            tags=["Upload"],
        )
        async def upload_video(file: UploadFile = File(..., max_length=10 * 1024 * 1024)):
            return _save(file, "videos")

        # upload model
        @api.post(
            "/upload/model",
            response_model=UploadResponse,
            summary="上传模型/权重文件",
            tags=["Upload"],
        )
        async def upload_model(file: UploadFile = File(...)):
            return _save(file, "models")

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

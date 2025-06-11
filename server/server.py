from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, Reuqest
from fastapi.responses import JSONResponse, HTMLResponse
import asyncio
import uuid
import logging
from queue import TaskQueue

class NndeployServer:
    instance: "NndeployServer" = None

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
        @api.post("/queue")
        async def enqueue(req:Reuqest):
            body = await req.json()
            task_id = body.get("id") or str(uuid.uuid4())
            payload = {
                "id": task_id,
                "graph_json": body["graph_json"]
            }
            self.queue.put(payload, prio=body.get("priority", 0))
            return {"task_id": task_id}
        
        # loop
        @api.get("/queue")
        async def queue_state():
            running, pending = self.queue.get_current_queue()
            return JSONResponse({"running": running, "pending": pending})
        
        # history
        @api.get("/history")
        async def history(max_items: int | None = None):
            return JSONResponse(self.queue.get_history(max_items))
        
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
        @self.app.get("/")
        async def root():
            return HTMLResponse("<h2>nndeploy backend: API OK</h2>")
        
        self.app.include_router(api)
    
    # queue progress notification
    def queue_updated(self):
        payload = {"pending": len(self.queue.get_current_queue()[1])}
        self.send_sync("queue_update", payload)

    def send_sync(self, event:str, data:dict, sid: WebSocket | None = None):
        self.loop.call_soon_threadsafe(asyncio.create_task, self._broadcast(event, data, sid))
    
    async def _broadcast(self, event, data, ws: WebSocket | None):
        message = {"type": event, "data": data}
        targets = [ws] if ws else list(self.sockets)
        for w in targets:
            try:
                await w.send_json(message)
            except Exception:
                self.sockets.discard(w)

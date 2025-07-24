from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import APIRouter, Request, status, UploadFile, File, Query
from fastapi import Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Set, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from .files import get_workdir
# from .frontend import FrontendManager
from nndeploy.dag.node import add_global_import_lib, import_global_import_lib
import os
import json
import asyncio
import uuid
import logging
from .utils import extract_encode_output_paths
import nndeploy.dag
from nndeploy import get_type_enum_json
from .task_queue import TaskQueue
from .schemas import (
    EnqueueRequest,
    EnqueueResponse,
    QueueStateResponse,
    HistoryItem,
    UploadResponse,
    NodeListResponse,
    WorkFlowSaveResponse,
    WorkFlowListResponse,
    WorkFlowLoadResponse,
    WorkFlowDeleteResponse,
    ParamTypeResponse
)
from .files import router as files_router
from .files import get_workdir

class NnDeployServer:
    instance: "NnDeployServer" = None

    def __init__(self, args, job_mp_queue, plugin_update_q):
        NnDeployServer.instance = self
        self.loop: Optional[asyncio.AbstractEventLoop] = None # lazy loading
        self.args = args
        self.app = FastAPI(
            title="nndeploy backend",
            version="0.1.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # web_root = FrontendManager.init_frontend(args.front_end_version)
        # dist_inside = Path(web_root) / "dist"
        # if dist_inside.is_dir():
        #     web_root = str(dist_inside)

        # self.app.mount("/design", StaticFiles(directory=web_root, html=True), name="frontend")
        # static_dir = Path(web_root) / "static"
        # if static_dir.is_dir():
        #     self.app.mount("/static", StaticFiles(directory=static_dir), name="design_static")

        self.plugin_update_q = plugin_update_q
        self.queue = TaskQueue(self, job_mp_queue)
        self.sockets: set[WebSocket] = set()
        self.ws_task_map: dict[WebSocket, set[str]] = {}
        self.task_ws_map: dict[str, set[WebSocket]] = {}
        self._register_routes()

    def _get_workdir(self) -> Path:
        return Path(self.args.resources)

    def _register_routes(self):
        api = APIRouter(prefix="/api")

        # loop
        @api.get(
            "/queue",
            response_model=QueueStateResponse,
            summary="check running / wait task",
        )
        async def queue_state():
            running, pending = self.queue.get_current_queue()
            return QueueStateResponse(running=running, pending=pending)

        # commit task
        @api.post(
            "/queue",
            response_model=EnqueueResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="enqueue nndeploy graph task",
        )
        async def enqueue(req: EnqueueRequest):
            task_id = str(uuid.uuid4())
            payload = {
                "id": task_id,
                "graph_json": req.root,
                "priority": 100,
            }
            self.queue.put(payload, prio=100)
            flag = "success"
            message = "success"
            result = {"task_id":task_id}
            return EnqueueResponse(flag=flag, message=message, result=result)

        @api.post(
            "/workflow/save",
            tags=["Workflow"],
            response_model=WorkFlowSaveResponse,
            summary="save workflow to file",
        )
        async def save_json(req: EnqueueRequest, filename: Optional[str] = None):
            # Use the 'name_' field from the incoming JSON or generate a UUID as fallback
            file_name = filename or req.root.get("name_", str(uuid.uuid4()))
            save_dir = Path(self.args.resources) / "workflow"
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"{file_name}"

            try:
                with open(file_path, 'w') as f:
                    json.dump(req.root, f, indent=4)

                flag = "success"
                message = "success"
                result = {"message": f"JSON saved to {file_path}", "file_path": str(file_path)}
                return WorkFlowSaveResponse(flag=flag, message=message, result=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

        @api.post(
                "/workflow/upload",
                tags=["Workflow"],
                response_model=UploadResponse,
                status_code=status.HTTP_201_CREATED,
                summary="upload workflow",
        )
        async def upload_workflow(
            file: UploadFile = File(...)
        ):
            allowed_extensions = {".json", ".yaml", ".yml"}
            suffix = Path(file.filename).suffix.lower()
            if suffix not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only .json, .yml and .yaml files are allowed, got: {suffix}"
                )
            folder = Path(self.args.resources) / "workflow"
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            dst = folder / file.filename
            with dst.open("wb") as w:
                w.write(file.file.read())

            flag = "success"
            message = f"workflow {dst.name} has been uploaded successfully"
            result = {
                "filename":file.filename,
                "saved_path":str(dst.resolve()),
                "size":dst.stat().st_size,
                "uploaded_at":datetime.utcnow(),
                "extension": (dst.suffix or "unknown").lstrip(".")
            }
            return UploadResponse(flag=flag, message=message, result=result)

        @api.get(
            "/workflow/download",
            tags=["Workflow"],
            summary="download workflow",
            response_class=FileResponse,
        )
        async def download_workflow(file_path: str = Query(..., description="absolute_path or relative path")):
            """
            download existed workflow file
            """
            f = Path(self.args.resources) / "workflow" / file_path
            if not f.exists():
                raise HTTPException(status_code=404, detail="Not found")
            
            MIME_MAP: dict[str, str] = {
                ".json": "application/json",
                ".yaml": "application/x-yaml",
                ".yml":  "application/x-yaml"
            }

            media_type = MIME_MAP.get(f.suffix.lower(), "application/octet-stream")

            if media_type is None:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported download type"
                )
            return FileResponse(f, media_type=media_type, filename=f.name)

        @api.get(
            "/workflow",
            tags=["Workflow"],
            response_model=WorkFlowListResponse,
            summary="load workflow lists",
        )
        async def get_workflow_json():
            workflow_dir = Path(self.args.resources) / "workflow"

            if not workflow_dir.exists():
                raise HTTPException(status_code=404, detail="workflow dir is not exist")

            workflow = []
            filenames = []
            result = {}
            try:
                for json_file in workflow_dir.glob("*.json"):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        workflow.append(data)
                        filenames.append(json_file.name)
                result["fileNames"]=filenames
                result["workflows"]=workflow
                flag = "success"
                message = "success"
                return WorkFlowListResponse(flag=flag, message=message, result=result)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"reading error: {e}")

        @api.post(
            "/workflow/delete/{file_name}",
            tags=["Workflow"],
            response_model=WorkFlowDeleteResponse,
            summary="delete workflow json",
        )
        async def delete_workflow_json(file_name: str):
            workflow_dir = Path(self.args.resources) / "workflow"

            file_path = workflow_dir / f"{file_name}"

            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"file {file_name}.json is not existed")

            try:
                file_path.unlink()
                flag = "success"
                message = f"file {file_name}.json has been deleted"
                return WorkFlowDeleteResponse(flag=flag, message=message)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"delete error: {e}")

        @api.get(
            "/workflow/{file_name}",
            tags=["Workflow"],
            response_model=WorkFlowLoadResponse,
            summary="get workflow json",
        )
        async def get_workflow_json(file_name: str):
            workflow_dir = Path(self.args.resources) / "workflow"

            file_path = workflow_dir / f"{file_name}"

            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"file {file_name}.json is not existed")

            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                flag = "success"
                message = "success"
                return WorkFlowLoadResponse(flag=flag, message=message, result=result)

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"reading file error: {e}")
        
        @api.get(
            "/dag/info",
            tags=["Node"],
            response_model=NodeListResponse,
            summary="return register nodes",
        )
        async def register_nodes():
            json_str = nndeploy.dag.get_dag_json()
            nodes = json.loads(json_str)
            flag = "success"
            message = ""
            return NodeListResponse(flag=flag, message=message, result=nodes)

        @api.post(
                "/nodes/upload",
                tags=["Node"],
                response_model=UploadResponse,
                status_code=status.HTTP_201_CREATED,
                summary="upload node",
        )
        async def upload_plugin(
            file: UploadFile = File(...)
        ):
            allowed_extensions = {".py", ".so"}
            suffix = Path(file.filename).suffix.lower()
            if suffix not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only .py and .so files are allowed, got: {suffix}"
                )

            folder = Path(self.args.resources) / "plugin"
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            dst = folder / file.filename
            with dst.open("wb") as w:
                w.write(file.file.read())

            self.plugin_update_q.put(str(dst.resolve()))
            add_global_import_lib(str(dst.resolve()))
            import_global_import_lib()

            flag = "success"
            message = f"workflow {dst.name} has been uploaded successfully"
            result = {
                "filename":file.filename,
                "saved_path":str(dst.resolve()),
                "size":dst.stat().st_size,
                "uploaded_at":datetime.utcnow(),
                "extension": (dst.suffix or "unknown").lstrip(".")
            }
            return UploadResponse(flag=flag, message=message, result=result)

        @api.get(
            "/param/types",
            response_model=ParamTypeResponse,
            summary="return param enum types"
        )
        async def get_param_enum_type():
            type_json = get_type_enum_json()
            flag = "success"
            message = "success"
            return ParamTypeResponse(flag=flag, message=message, result=type_json)

        # history
        @api.get(
            "/history",
            response_model=Dict[str, HistoryItem],
            summary="check history",
        )
        async def history(max_items: Optional[int] = None):
            return self.queue.get_history(max_items)

        # index
        @self.app.get("/", tags=["Root"])
        async def root():
            return HTMLResponse("<h2>nndeploy backend: API OK</h2>")

        # preview
        @api.get("/preview", tags=["Files"],
                summary="preview images/videos")
        async def preview_file(file_path: str = Query(..., description="absolute_path or relative path"), time: Optional[str] = None):
            f = Path(self.args.resources) / file_path
            if not f.exists():
                raise HTTPException(status_code=404, detail="Not found")

            MIME_MAP: dict[str, str] = {
                # ---- image ----
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png":  "image/png",
                ".webp": "image/webp",
                ".gif":  "image/gif",
                ".svg":  "image/svg+xml",

                # ---- video ----
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",
            }

            mime = MIME_MAP.get(f.suffix.lower())
            if mime is None:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported preview type"
                )
            return FileResponse(f, media_type=mime, filename=None)

        # download
        @api.get("/download", tags=["Files"],
                summary="download images/videos/models")
        async def download_file(file_path: str = Query(..., description="absolute_path or relative path")):
            f = Path(self.args.resources) / file_path
            if not f.exists():
                raise HTTPException(status_code=404, detail="Not found")

            MIME_MAP: dict[str, str] = {
                # ---- image ----
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png":  "image/png",
                ".webp": "image/webp",
                ".gif":  "image/gif",
                ".svg":  "image/svg+xml",

                # ---- video ----
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",

                # ---- model ----
                ".onnx": "application/octet-stream"
            }

            mime = MIME_MAP.get(f.suffix.lower())
            if mime is None:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported download type"
                )
            return FileResponse(f, media_type=mime, filename=f.name)

        @self.app.on_event("startup")
        async def _on_startup():
            self.loop = asyncio.get_running_loop()

        @api.websocket("/ws/progress")
        async def ws_progress(ws: WebSocket):
            await ws.accept()
            self.sockets.add(ws)
            self.ws_task_map[ws] = set()

            try:
                while True:
                    msg = await ws.receive_json()
                    if msg.get("type") == "bind" and "task_id" in msg:
                        task_id = msg["task_id"]
                        self.ws_task_map[ws].add(task_id)
                        self.task_ws_map.setdefault(task_id, set()).add(ws)
            except WebSocketDisconnect:
                logging.info("[WebSocket] client disconnected")
            finally:
                self.sockets.discard(ws)
                task_ids = self.ws_task_map.pop(ws, set())
                for tid in task_ids:
                    self.task_ws_map.get(tid, set()).discard(ws)

        self.app.include_router(
            files_router,
            dependencies=[Depends(self._get_workdir)]
        )
        self.app.dependency_overrides[get_workdir] = self._get_workdir

        self.app.include_router(api,dependencies=[Depends(lambda: get_workdir(self))])

    # task progress notify
    def notify_task_progress(self, task_id: str, status_dict: dict):
        flag = "success"
        message = "task running"
        result = {
            "task_id": task_id,
            "type": "progress",
            "detail": status_dict
        }
        payload = {"flag": flag, "message": message, "result": result}
        ws_set = self.task_ws_map.get(task_id, set())
        if not ws_set:
            return

        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(payload, ws),
                    self.loop
                )
            else:
                logging.warning("[notify_task_progress] Event loop not ready or not running")

    # task done notify
    def notify_task_done(self, task_id: str):
        task_info = self.queue.get_task_by_id(task_id)
        if task_info is None:
            raise HTTPException(status_code=404, detail="task not found")
        graph_json = task_info.get("task").get("graph_json")
        path, text = extract_encode_output_paths(graph_json)

        flag = "success"
        message = "notify task done"
        result = {"task_id": task_id, "type": "preview", "path": path, "text": text}
        payload = {"flag": flag, "message": message, "result": result}
        ws_set = self.task_ws_map.get(task_id, set())
        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(payload, ws),
                    self.loop
                )
            else:
                logging.warning("[notify_task_done] Event loop not ready or not running")

    async def _broadcast(self, payload: dict, ws: WebSocket | None = None):
        targets = [ws] if ws else list(self.sockets)
        for w in targets:
            try:
                logging.info(f"[_broadcast] sending to {w.client}")
                await w.send_json(payload)
            except Exception as e:
                logging.error(f"[_broadcast] failed to send: {e}")
                self.sockets.discard(w)

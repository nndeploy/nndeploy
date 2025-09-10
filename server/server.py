from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import APIRouter, Request, status, UploadFile, File, Query
from fastapi import Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from typing import Set, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import os
import json
import asyncio
import uuid
import logging
import sqlite3
import shutil
import requests
import contextvars
import nndeploy.dag

from nndeploy.dag.node import add_global_import_lib, import_global_import_lib
from nndeploy import get_type_enum_json
from .utils import extract_encode_output_paths, _handle_urls
from .frontend import FrontendManager
from .template import WorkflowTemplateManager
from .task_queue import TaskQueue
from .task_queue import ExecutionStatus
from .schemas import (
    EnqueueRequest,
    EnqueueResponse,
    QueueStateResult,
    QueueStateResponse,
    HistoryItem,
    HistoryResponse,
    UploadResponse,
    NodeListResponse,
    WorkFlowSaveResponse,
    WorkFlowListResponse,
    WorkFlowLoadResponse,
    WorkFlowDeleteResponse,
    ParamTypeResponse,
    TemplateJsonListResponse,
    TemplateLoadResponse
)
from .files import router as files_router
from .files import get_workdir
from .logging_taskid import set_task_id, reset_task_id, run_func_in_copied_context, scoped_stdio_to_logging
from .db import DB

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise

class NnDeployServer:
    instance: "NnDeployServer" = None

    def __init__(self, args, job_mp_queue, plugin_update_q, cancel_event_queue):
        NnDeployServer.instance = self
        self.loop: Optional[asyncio.AbstractEventLoop] = None # lazy loading
        self.args = args
        self.app = FastAPI(
            title="nndeploy backend",
            version="1.0.0"
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.workflow_dir = Path(self.args.resources) / "workflow"
        self.db_path = Path(self.args.resources) / "db" / "nndeploy.db"
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = DB(self.db_path).init_schema()

        self.cancel_event_queue = cancel_event_queue
        self.plugin_update_q = plugin_update_q
        self.queue = TaskQueue(self, job_mp_queue)
        self.sockets: set[WebSocket] = set()
        self.ws_task_map: dict[WebSocket, set[str]] = {}
        self.task_ws_map: dict[str, set[WebSocket]] = {}
        self._register_routes(args)

        template_parent = WorkflowTemplateManager.init_templates()
        self.template_path = Path(template_parent) / "nndeploy-workflow"

    def _write_json_replace(self, dst: Path, data: dict | bytes) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if isinstance(data, dict):
            tmp.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        else:
            tmp.write_bytes(data)
        tmp.replace(dst)

    def _norm_id(self, val) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, str):
            s = val.strip()
            return s or None
        s = str(val).strip()

    def _generate_unique_path(self, desired: Path) -> Path:
        if not desired.exists():
            return desired
        stem, suf = desired.stem, desired.suffix
        i = 1
        while True:
            cand = desired.with_name(f"{stem}({i}){suf}")
            if not cand.exists():
                return cand
            i += 1

    def _get_workdir(self) -> Path:
        return Path(self.args.resources)

    def read_json_recursive(self, root: Path) -> list[dict]:
        result = []
        for json_file in root.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    result.append(content)
            except Exception as e:
                logging.warning(f"Failed to read {json_file}: {e}")
        return result

    def _find_cover_and_requirements(self, json_path: Path) -> tuple[Optional[str], Optional[str]]:
        d = json_path.parent
        base = json_path.stem

        # cover
        cover = None
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            p = d / f"cover.{base}{ext}"
            if p.exists():
                cover = str(p.resolve())
                break

        # requirements
        req = None
        cand = d / f"readme.{base}.md"
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                req = f.read()
        else:
            cand2 = d / "readme.template.md"
            if cand2.exists():
                with open(cand2, "r", encoding="utf-8") as f:
                    req = f.read()

        return cover, req

    def _cancel_task(self, task_id: str):
        self.cancel_event_queue.put(task_id)
        return True

    def _register_routes(self, args):
        api = APIRouter(prefix="/api")

        @api.get(
            "/queue/info",
            tags=["Task"],
            response_model=QueueStateResponse,
            summary="check running / wait task",
        )
        async def queue_state():
            state = self.queue.get_current_queue()
            return QueueStateResponse(
                flag="success",
                message="queue state fetched",
                result=QueueStateResult(
                    running=state["RUNNING"],
                    pending=state["PENDING"],
                    dispatched=state["DISPATCHED"],
                ),
            )

        # commit task
        @api.post(
            "/queue",
            tags=["Task"],
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
            "/queue/cancel/{task_id}",
            tags=["Task"],
            summary="cancel task execution",
        )
        async def cancel_task(task_id: str):
            try:
                if self._cancel_task(task_id):
                    return {"flag": "success", "message": f"Task {task_id} has been cancelled"}
            except HTTPException as e:
                raise e

        @api.post(
            "/queue/flush",
            tags=["Task"],
            summary="clear pending tasks and drain job queue (running tasks not interrupted)"
        )
        async def flush_queue():
            try:
                res = self.queue.flush()
                msg = f"pending cleared: {res['cleared_pending']}, job_q drained: {res['drained_job_q']}"
                logging.info("[queue/flush] %s", msg)
                return {"status": "success", "message": msg, "result": res}
            except Exception as e:
                logging.exception("[queue/flush] error")
                raise HTTPException(status_code=500, detail=f"flush queue error: {e}")

        @api.post(
            "/workflow/save",
            tags=["Workflow"],
            response_model=WorkFlowSaveResponse,
            summary="save workflow (update if id provided, else create)",
        )
        async def save_json(req: EnqueueRequest):
            try:
                root = req.root if isinstance(req.root, dict) else {}
                data = root.get("businessContent")
                if data is None:
                    raise HTTPException(status_code=400, detail="businessContent is required")

                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError as e:
                        raise HTTPException(status_code=400, detail=f"businessContent must be valid JSON: {e}")
                elif not isinstance(data, dict):
                    raise HTTPException(status_code=400, detail="businessContent must be a JSON object or JSON string")

                id = self._norm_id(root.get("id"))

                if id:
                    dst = self.db.get_workflow_path(id)
                    if not dst:
                        raise HTTPException(status_code=404, detail=f"workflow id {id} not found")
                    self._write_json_replace(dst, data)
                    cover, req = self._find_cover_and_requirements(dst)
                    self.db.update_workflow_metadata(id, dst, cover, req)

                    return WorkFlowSaveResponse(
                        flag="success",
                        message="updated",
                        result={"id": id}
                    )

                else:
                    name_from_json = data.get("name_")
                    base_name = name_from_json if name_from_json else f"{uuid.uuid4()}.json"
                    if not base_name.endswith((".json", ".yml", ".yaml")):
                        base_name = f"{base_name}.json"
                    desired = self.workflow_dir / base_name
                    dst = self._generate_unique_path(desired)

                    self._write_json_replace(dst, data)
                    cover, req = self._find_cover_and_requirements(dst)
                    wid = self.db.insert_workflow(dst, cover, req)

                    return WorkFlowSaveResponse(
                        flag="success",
                        message="created",
                        result={"id": wid}
                    )

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"save error: {e}")

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
            desired = self.workflow_dir / file.filename
            dst = self._generate_unique_path(desired)
            try:
                content = await file.read()
                self._write_json_replace(dst, content)
                cover, req = self._find_cover_and_requirements(dst)
                wid = self.db.insert_workflow(dst, cover, req)

                return UploadResponse(
                    flag="success",
                    message=f"workflow {dst.name} has been uploaded",
                    result={
                        "id": wid,
                        "filename": dst.name,
                        "saved_path": str(dst.resolve()),
                        "size": dst.stat().st_size,
                        "uploaded_at": datetime.utcnow(),
                        "extension": (dst.suffix or "unknown").lstrip("."),
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"upload error: {e}")

        @api.get(
            "/workflow/download/{id}",
            tags=["Workflow"],
            summary="download workflow by id",
            response_class=FileResponse,
        )
        async def download_workflow(id: str):
            """
            download workflow file by id (UUID)
            """
            f = self.db.get_workflow_path(id)
            if not f:
                raise HTTPException(status_code=404, detail=f"workflow id {id} not found in DB")
            f = Path(f)
            if not f.exists():
                raise HTTPException(status_code=404, detail=f"workflow file not found: {f}")

            MIME_MAP: dict[str, str] = {
                ".json": "application/json",
                ".yaml": "application/x-yaml",
                ".yml": "application/x-yaml"
            }

            media_type = MIME_MAP.get(f.suffix.lower(), "application/octet-stream")
            if not media_type:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported download type: {f.suffix}"
                )

            return FileResponse(f, media_type=media_type, filename=f.name)

        @api.get(
            "/workflows",
            tags=["Workflow"],
            response_model=WorkFlowListResponse,
            summary="list workflows summary",
        )
        async def list_workflows_summary():
            if not self.workflow_dir.exists():
                raise HTTPException(status_code=404, detail="workflow dir is not exist")

            results = []

            try:
                for jf in self.workflow_dir.rglob("*.json"):
                    try:
                        wid = self.db.upsert_workflow_by_path(jf)

                        id_cover = self.db.get_workflow_id_and_cover_by_path(jf)
                        if not id_cover:
                            logging.warning(f"[workflow->db] missing id for {jf}")
                            continue
                        wid, _cover = id_cover

                        with jf.open("r", encoding="utf-8") as f:
                            content = json.load(f)

                        results.append({
                            "id": wid,
                            "name_": content.get("name_"),
                            "developer_": content.get("developer_"),
                            "desc_": content.get("desc_"),
                        })
                    except Exception as fe:
                        logging.warning(f"[workflow->scan] failed for {jf}: {fe}")
            except Exception as e:
                logging.warning(f"[workflow->db] batch record failed: {e}")

            return WorkFlowListResponse(
                flag="success",
                message=f"{len(results)} workflows found",
                result=results
            )

        @api.post(
            "/workflow/delete/{id}",
            tags=["Workflow"],
            response_model=WorkFlowDeleteResponse,
            summary="delete workflow by id",
        )
        async def delete_workflow_json(id: str):
            file_path = self.db.get_workflow_path(id)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"workflow id {id} not found in DB")

            file_path = Path(file_path)
            if not file_path.exists():
                self.db.delete_workflow(id)
                raise HTTPException(status_code=404, detail=f"workflow file not found: {file_path}")

            try:
                file_path.unlink(missing_ok=False)
                self.db.delete_workflow(id)
                return WorkFlowDeleteResponse(flag="success", message=f"workflow id {id} has been deleted")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"delete error: {e}")

        @api.get(
            "/workflow/{id}",
            tags=["Workflow"],
            response_model=WorkFlowLoadResponse,
            summary="get workflow json by id",
        )
        async def get_workflow_by_id(id: str):
            file_path = self.db.get_workflow_path(id)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"workflow id {id} not found")
            file_path = Path(file_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"workflow file not found: {file_path}")
            if file_path.suffix.lower() != ".json":
                raise HTTPException(status_code=400, detail=f"unsupported workflow type: {file_path.suffix}")

            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return WorkFlowLoadResponse(flag="success", message="success", result=data)

        @api.get(
            "/template",
            tags=["Template"],
            response_model=TemplateJsonListResponse,
            summary="Recursively return all JSON contents under template directory"
        )
        async def get_all_template_jsons():
            if not self.template_path.exists():
                return TemplateJsonListResponse(
                    flag="error",
                    message="template directory not found",
                    result=[]
                )

            results = []
            new_count = 0

            try:
                for jf in self.template_path.rglob("*.json"):
                    try:
                        cover, req = self._find_cover_and_requirements(jf)
                        self.db.insert_or_ignore_template(jf, cover, req)

                        meta = self.db.get_template_meta_by_path(jf)
                        if meta is None:
                            logging.warning(f"[template->db] missing id for {jf}")
                            continue
                        tid, cover, requirements = meta

                        with jf.open("r", encoding="utf-8") as f:
                            content = json.load(f)

                        item = {
                            "id": tid,
                            "name_": content.get("name_"),
                            "developer_": content.get("developer_"),
                            "source_": content.get("source_"),
                            "desc_": content.get("desc_"),
                            "cover_": cover,
                            "requirements_": requirements
                        }
                        results.append(item)
                        new_count += 1
                    except Exception as fe:
                        logging.warning(f"[template->scan] failed to handle {jf}: {fe}")

            except Exception as e:
                logging.warning(f"[template->db] batch record failed: {e}")

            return TemplateJsonListResponse(
                flag="success",
                message=f"{len(results)} json files loaded",
                result=results
            )

        @api.get(
            "/template/{id}",
            tags=["Template"],
            response_model=WorkFlowLoadResponse,
            summary="get workflow json",
        )
        async def get_template_json(id: str):
            try:
                file_path = self.db.get_template_path(id)
                if file_path is None:
                    raise HTTPException(status_code=404, detail=f"template id {id} not found")
                file_path = Path(file_path)

                if not file_path.exists():
                    raise HTTPException(status_code=404, detail=f"template file not found: {file_path}")

                if file_path.suffix.lower() != ".json":
                    raise HTTPException(status_code=400, detail=f"unsupported template type: {file_path.suffix}")

                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                flag = "success"
                message = "template load response"
                return TemplateLoadResponse(flag=flag, message=message, result=data)

            except HTTPException:
                raise
            except Exception as e:
                logging.exception("[get_template_json] unexpected error")
                raise HTTPException(status_code=500, detail=f"read template error: {e}")

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
            "/queue/history",
            tags=["Task"],
            response_model=HistoryResponse,
            summary="check history",
        )
        async def history(max_items: Optional[int] = Query(None, ge=1, le=1000)):
            try:
                hist_map: Dict[str, Dict[str, Any]] = self.queue.get_history(max_items=max_items)

                items: List[Dict[str, Any]] = []
                for task_id, rec in hist_map.items():
                    item = {
                        "task_id": task_id,
                        "task": rec.get("task"),
                        "status": rec.get("status"),
                        "state": rec.get("state"),
                        "ts_submit": rec.get("ts_submit"),
                        "ts_dispatch": rec.get("ts_dispatch"),
                        "ts_start": rec.get("ts_start"),
                        "ts_finish": rec.get("ts_finish"),
                        "worker_pid": rec.get("worker_pid"),
                        "time_profile": rec.get("time_profile", {}),
                    }
                    items.append(item)

                def _key(x: Dict[str, Any]):
                    return x.get("ts_finish") or x.get("ts_start") or x.get("ts_submit") or 0.0

                items.sort(key=_key, reverse=True)

                return HistoryResponse(
                    flag="success",
                    message="history fetched",
                    result={"items": items, "total": len(items)},
                )
            except Exception as e:
                logging.exception("Get history failed")
                return HistoryResponse(
                    flag="fail",
                    message=str(e),
                    result={"items": [], "total": 0},
                )


        # index
        # @self.app.get("/", tags=["Root"])
        # async def root():
        #     return HTMLResponse("<h2>nndeploy backend: API OK</h2>")

        # preview
        @api.get("/preview", tags=["Files"],
                summary="preview images/videos/txt")
        async def preview_file(file_path: str = Query(..., description="absolute_path or relative path"), time: Optional[str] = None):
            file_path = Path(file_path)
            # unsafe process for relative path
            if file_path.is_absolute():
                f = file_path
            else:
                resource_root = Path(self.args.resources).resolve()
                first_part = file_path.parts[0] if file_path.parts else ""
                if first_part == resource_root.name:
                    f = file_path
                else:
                    f = resource_root / file_path

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

                # ---- text ----
                ".txt": "text/plain",
            }

            mime = MIME_MAP.get(f.suffix.lower())
            if mime is None:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported preview type"
                )
            if mime == "text/plain":
                return PlainTextResponse(f.read_text(encoding="utf-8"))
            else:
                return FileResponse(f, media_type=mime, filename=None)

        # download
        @api.get("/download", tags=["Files"],
                summary="download images/videos/models/txt")
        async def download_file(file_path: str = Query(..., description="absolute_path or relative path")):
            f = Path(file_path)
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
                ".onnx": "application/octet-stream",

                # ---- text ----
                ".txt": "text/plain"
            }

            mime = MIME_MAP.get(f.suffix.lower())
            if mime is None:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported download type"
                )
            return FileResponse(f, media_type=mime, filename=f.name)

        @api.post(
            "/models/download",
            status_code=status.HTTP_202_ACCEPTED,
            summary="start model download by json"
        )
        async def download_models(req: EnqueueRequest):
            task_id = str(uuid.uuid4())
            graph_json = req.root if isinstance(req.root, dict) else {}
            asyncio.create_task(self._download_models_task(task_id, graph_json))
            return JSONResponse({
                "flag": "success",
                "message": "download started",
                "result": {"task_id": task_id}
            }, status_code=status.HTTP_202_ACCEPTED)

        @self.app.on_event("startup")
        async def _on_startup():
            self.loop = asyncio.get_running_loop()

        @self.app.on_event("shutdown")
        async def _on_shutdown():
            try:
                self.db.close()
                logging.info("[DB] closed")
            except Exception:
                pass

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

        if not args.debug:
            web_root = FrontendManager.init_frontend(args.front_end_version)
            dist_inside = Path(web_root) / "dist"
            if dist_inside.is_dir():
                web_root = str(dist_inside)
            self.app.mount("/", SPAStaticFiles(directory=web_root, html=True), name="frontend")

    # model download notify
    def notify_download_done(self, task_id: str, success: bool, result: dict | None, error: str | None):
        payload = {
            "flag": "success" if success else "failed",
            "message": "download done" if success else f"download failed: {error}",
            "result": {
                "task_id": task_id,
                "type": "model_download_done",
                "detail": result or {}
            }
        }
        ws_set = self.task_ws_map.get(task_id, set())
        if not ws_set:
            return
        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self._broadcast(payload, ws), self.loop)
            else:
                logging.warning("[notify_download_done] Event loop not ready or not running")

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
    def notify_task_done(self, task_id: str, status: ExecutionStatus, results: Dict, time_profile_map: Dict):
        task_info = self.queue.get_task_by_id(task_id)
        if task_info is None:
            raise HTTPException(status_code=404, detail="task not found")
        graph_json = task_info.get("task").get("graph_json")
        path, text = extract_encode_output_paths(graph_json)

        # send result
        flag = status.str
        message = status.messages
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

        # send graph output
        show_items = []
        try:
            for node_name, out_map in (results or {}).items():
                if not isinstance(out_map, dict):
                    continue
                if "String" not in out_map:
                    continue
                img_val = out_map.get("String")
                img_list = img_val if isinstance(img_val, list) else [img_val]
                text_list = [str(x) for x in img_list]
                show_items.append({"name": node_name, "text": text_list})
        except Exception as e:
            logging.exception("[notify_task_done] build show_items failed: %s", e)

        if show_items:
            payload = {
                "flag": status.str,
                "message": status.messages,
                "result": {"task_id": task_id, "type": "show", "string": show_items},
            }
            for ws in ws_set.copy():
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(self._broadcast(payload, ws), self.loop)
                else:
                    logging.warning("[notify_task_done] Event loop not ready or not running")

        # send graph run info
        flag = status.str
        message = "graph run info"
        result = {"task_id": task_id, "type": "task_run_info", "time_profile": time_profile_map}
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

    async def _download_models_task(self, task_id: str, graph_json: dict):
        token = set_task_id(task_id)
        try:
            with scoped_stdio_to_logging("model-download"):
                logging.info("model download task started")

                result = await asyncio.to_thread(
                    run_func_in_copied_context,
                    _handle_urls, graph_json, self.args.resources
                )
                logging.info("model download task finished: %s", result)
            self.notify_download_done(task_id, success=True, result=result, error=None)
        except Exception as e:
            logging.exception("model download task failed")
            self.notify_download_done(task_id, success=False, result=None, error=str(e))
        finally:
            reset_task_id(token)

# files.py
from __future__ import annotations
from pydantic import BaseModel

import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from typing import Optional
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import JSONResponse

from .schemas import (UploadResponse, DeleteResponse, FileListResponse)

# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
router = APIRouter(
    prefix="/api/files",
    tags=["Files"],
)

# ──────────────────────────────────────────────
# dependency_overrides or include_router.dependencies
# ──────────────────────────────────────────────
def get_workdir(server) -> Path:
    return server._get_workdir()

# ──────────────────────────────────────────────
# save function
# ──────────────────────────────────────────────
# def _save(file: UploadFile, workdir: Path, subdir: str) -> UploadResponse:
#     folder = workdir / subdir
#     folder.mkdir(parents=True, exist_ok=True)

#     dst = folder / file.filename
#     with dst.open("wb") as w:
#         w.write(file.file.read())

#     flag = "success"
#     message = f"file {dst.name} has been uploaded successfully"
#     result = {
#         "filename":file.filename,
#         "saved_path":str(dst.resolve()),
#         "size":dst.stat().st_size,
#         "uploaded_at":datetime.utcnow(),
#         "extension": (dst.relative_to(workdir).suffix or "unknown").lstrip(".")
#     }
#     return UploadResponse(flag=flag, message=message, result=result)

def path_to_id(path: Path) -> str:
    return str(uuid5(NAMESPACE_URL, str(path.resolve())))

def parent_path_to_id(path: Path, root: Path | None = None) -> str:
    parent = path.parent
    if root and parent.resolve() == root.resolve():
        return ""
    return path_to_id(parent)

def _save(file: UploadFile, target_dir: str) -> UploadResponse:
    """
    upload file to target path
    """
    folder = Path(target_dir)
    folder.mkdir(parents=True, exist_ok=True)

    dst = folder / file.filename
    with dst.open("wb") as w:
        w.write(file.file.read())

    flag = "success"
    message = f"file {dst.name} has been uploaded successfully"
    result = {
        "filename": file.filename,
        "saved_path": str(dst.resolve()),
        "size": dst.stat().st_size,
        "uploaded_at": datetime.utcnow(),
        "extension": (dst.suffix or "unknown").lstrip(".")
    }
    return UploadResponse(flag=flag, message=message, result=result)

# def _delete(filename: str, workdir: Path, subdir: str) -> DeleteResponse:
#     folder = workdir / subdir

#     if not folder.exists():
#         return DeleteResponse(
#             flag="failed",
#             message=f"path {folder} does not exist",
#         )

#     dst = folder / filename

#     if not dst.exists():
#         return DeleteResponse(
#             flag="failed",
#             message=f"file {dst.name} does not exist in {folder}",
#         )

#     if not dst.exists():
#         raise FileNotFoundError(f"{dst} does not exist.")
#     original_size = dst.stat().st_size
#     dst.unlink()
#     flag = "success"
#     message = f"file {filename}.json has been deleted"
#     return DeleteResponse(flag=flag, message=message)

def _delete(target_path: str) -> DeleteResponse:
    """
    delete file according to target path
    """
    path = Path(target_path)

    if not path.exists():
        return DeleteResponse(
            flag="failed",
            message=f"path {path} does not exist",
        )

    try:
        if path.is_file():
            size = path.stat().st_size
            path.unlink()
            message = f"file {path.name} has been deleted (size={size} bytes)"
        elif path.is_dir():
            shutil.rmtree(path)
            message = f"directory {path.name} has been deleted"
        else:
            return DeleteResponse(
                flag="failed",
                message=f"path {path} is neither a file nor a directory",
            )
    except Exception as e:
        return DeleteResponse(
            flag="failed",
            message=f"failed to delete {path}: {str(e)}",
        )

    return DeleteResponse(flag="success", message=message)

# def _file_info(path: Path, base: Optional[Path] = None) -> Dict[str, str]:
#     stat = path.stat()
#     return {
#         "filename":path.name,
#         "saved_path":str(path),
#         "size": stat.st_size,
#         "uploaded_at":datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
#         "extension": (path.suffix or "unknown").lstrip(".")
#     }

def _file_info(path: Path, base: Optional[Path] = None, parent_id: str = "") -> Dict[str, str]:
    stat = path.stat()
    is_dir = path.is_dir()

    size = 4096 if is_dir else stat.st_size

    node_id = path_to_id(path)

    file_info = {
        "filename": path.name,
        "saved_path": str(path),
        "size": size,
        "uploaded_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "extension": "" if is_dir else (path.suffix.lstrip(".") or "unknown"),
    }

    return {
        "id": node_id,
        "name": path.name,
        "parentId": parent_id,
        "type": "branch" if is_dir else "leaf",
        "path": str(path),
        "file_info": file_info
    }

# ──────────────────────────────────────────────
# node: file list
# ──────────────────────────────────────────────
# @router.get("", response_model=FileListResponse, summary="list upload dir files")
# async def list_files(workdir: Path = Depends(get_workdir)) -> Dict[str, List[Dict[str, str]]]:
#     base = workdir
#     tree: Dict[str, List[Dict[str, str]]] = {}
#     if base.exists():
#         for folder in base.iterdir():
#             if folder.is_dir():
#                 if (folder.name not in ["images", "videos", "models"]):
#                     continue
#                 tree[folder.name] = [_file_info(p, base=folder) for p in folder.rglob("*") if p.is_file()]
#     flag = "success"
#     message = "List all resources"
#     return FileListResponse(flag=flag, message=message, result=tree)

@router.get("", response_model=FileListResponse, summary="list upload dir files")
async def list_files(workdir: Path = Depends(get_workdir)) -> Dict[str, List[Dict[str, str]]]:
    allowed_dirs = ["images", "videos", "models"]
    file_info_list: List[Dict[str, str]] = []

    def traverse_directory(path: Path, parent_id: str = "") -> None:
        if path.exists() and path.is_dir():
            for item in path.iterdir():
                if item.name.startswith("."):
                    continue
                info = _file_info(item, base=workdir, parent_id=parent_id)
                file_info_list.append(info)
                if item.is_dir():
                    traverse_directory(item, info["id"])

    if workdir.exists():
        for folder in workdir.iterdir():
            if folder.is_dir() and folder.name in allowed_dirs:
                if folder.name.startswith("."):
                    continue
                top_info = _file_info(folder, base=workdir, parent_id="")
                file_info_list.append(top_info)
                traverse_directory(folder, top_info["id"])

    return FileListResponse(
        flag="success",
        message="List all resources",
        result=file_info_list
    )

# ──────────────────────────────────────────────
# node: delete image
# ──────────────────────────────────────────────
# @router.post(
#     "/delete/images",
#     response_model=DeleteResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="delete images",
# )
# async def delete_image(
#     file_path: str
# ):
#     return _delete(file_path)

# ──────────────────────────────────────────────
# node: upload images
# ──────────────────────────────────────────────
# @router.post(
#     "/images",
#     response_model=UploadResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="upload images",
# )
# async def upload_image(
#     file_path: str,
#     file: UploadFile = File(...)
# ):
#     return _save(file, file_path)

# ──────────────────────────────────────────────
# node: delete video
# ──────────────────────────────────────────────
# @router.post(
#     "/delete/videos",
#     response_model=DeleteResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="delete video",
# )
# async def delete_video(file_path: str):
#     return _delete(file_path)

# ──────────────────────────────────────────────
# node: upload videos
# ──────────────────────────────────────────────
# @router.post(
#     "/videos",
#     response_model=UploadResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="upload video files",
# )
# async def upload_video(
#     file_path: str,
#     file: UploadFile = File(...)
# ):
#     return _save(file, file_path)

# ──────────────────────────────────────────────
# node: delete model
# ──────────────────────────────────────────────
# @router.post(
#     "/delete/models",
#     response_model=DeleteResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="delete model",
# )
# async def delete_model(file_path: str
# ):
#     return _delete(file_path)

# ──────────────────────────────────────────────
# node: upload models
# ──────────────────────────────────────────────
# @router.post(
#     "/models",
#     response_model=UploadResponse,
#     status_code=status.HTTP_201_CREATED,
#     summary="upload models",
# )
# async def upload_model(
#     file_path: str,
#     file: UploadFile = File(...)
# ):
#     return _save(file, file_path)

# ──────────────────────────────────────────────
# node: delete file
# ──────────────────────────────────────────────
@router.post(
    "/delete",
    response_model=DeleteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="delete images/videos/models",
)
async def delete_file(file_path: str
):
    return _delete(file_path)

# ──────────────────────────────────────────────
# node: upload file
# ──────────────────────────────────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="upload images/videos/models",
)
async def upload_file(
    file_path: str,
    file: UploadFile = File(...)
):
    return _save(file, file_path)

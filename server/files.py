# files.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from typing import Optional

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
def _save(file: UploadFile, workdir: Path, subdir: str) -> UploadResponse:
    folder = workdir / subdir
    folder.mkdir(parents=True, exist_ok=True)

    dst = folder / file.filename
    with dst.open("wb") as w:
        w.write(file.file.read())

    flag = "success"
    message = f"file {dst.name} has been uploaded successfully"
    result = {
        "filename":file.filename,
        "saved_path":str(dst.resolve()),
        "size":dst.stat().st_size,
        "uploaded_at":datetime.utcnow(),
        "extension": (dst.relative_to(workdir).suffix or "unknown").lstrip(".")
    }
    return UploadResponse(flag=flag, message=message, result=result)

def _delete(filename: str, workdir: Path, subdir: str) -> DeleteResponse:
    folder = workdir / subdir

    if not folder.exists():
        return DeleteResponse(
            flag="failed",
            message=f"path {folder} does not exist",
        )

    dst = folder / filename

    if not dst.exists():
        return DeleteResponse(
            flag="failed",
            message=f"file {dst.name} does not exist in {folder}",
        )

    if not dst.exists():
        raise FileNotFoundError(f"{dst} does not exist.")
    original_size = dst.stat().st_size
    dst.unlink()
    flag = "success"
    message = f"file {filename}.json has been deleted"
    return DeleteResponse(flag=flag, message=message)

def _file_info(path: Path, base: Optional[Path] = None) -> Dict[str, str]:
    stat = path.stat()
    return {
        "filename":path.name,
        "saved_path":str(path.relative_to(base)) if base else p.name,
        "size": stat.st_size,
        "uploaded_at":datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "extension": (path.suffix or "unknown").lstrip(".")
    }

# ──────────────────────────────────────────────
# node: file list
# ──────────────────────────────────────────────
@router.get("", response_model=FileListResponse, summary="list upload dir files")
async def list_files(workdir: Path = Depends(get_workdir)) -> Dict[str, List[Dict[str, str]]]:
    base = workdir
    tree: Dict[str, List[Dict[str, str]]] = {}
    if base.exists():
        for folder in base.iterdir():
            if folder.is_dir():
                if (folder.name not in ["images", "videos", "models"]):
                    continue
                tree[folder.name] = [_file_info(p, base=folder) for p in folder.rglob("*") if p.is_file()]
    flag = "success"
    message = "List all resources"
    return FileListResponse(flag=flag, message=message, result=tree)

# ──────────────────────────────────────────────
# node: delete image
# ──────────────────────────────────────────────
@router.post(
    "/delete/images/{file_name}",
    response_model=DeleteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="delete images",
)
async def delete_image(
    file_name: str,
    workdir: Path = Depends(get_workdir)
):
    return _delete(file_name, workdir, "images")

# ──────────────────────────────────────────────
# node: upload images
# ──────────────────────────────────────────────
@router.post(
    "/images",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="upload images",
)
async def upload_image(
    file: UploadFile = File(...),
    workdir: Path = Depends(get_workdir),
):
    return _save(file, workdir, "images")

# ──────────────────────────────────────────────
# node: delete video
# ──────────────────────────────────────────────
@router.post(
    "/delete/videos/{file_name}",
    response_model=DeleteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="delete video",
)
async def delete_video(file_name: str,
                       workdir: Path = Depends(get_workdir)
):
    return _delete(file_name, workdir, "videos")

# ──────────────────────────────────────────────
# node: upload videos
# ──────────────────────────────────────────────
@router.post(
    "/videos",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="upload video files",
)
async def upload_video(
    file: UploadFile = File(...),
    workdir: Path = Depends(get_workdir),
):
    return _save(file, workdir, "videos")

# ──────────────────────────────────────────────
# node: delete model
# ──────────────────────────────────────────────
@router.post(
    "/delete/models/{file_name}",
    response_model=DeleteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="delete model",
)
async def delete_model(file_name: str,
                       workdir: Path = Depends(get_workdir)
):
    return _delete(file_name, workdir, "models")

# ──────────────────────────────────────────────
# node: upload models
# ──────────────────────────────────────────────
@router.post(
    "/models",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="upload models",
)
async def upload_model(
    file: UploadFile = File(...),
    workdir: Path = Depends(get_workdir),
):
    return _save(file, workdir, "models")

# files.py
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import JSONResponse

from schemas import UploadResponse

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
def get_workdir() -> Path:
    return Path.cwd()

# ──────────────────────────────────────────────
# save function
# ──────────────────────────────────────────────
def _save(file: UploadFile, workdir: Path, subdir: str) -> UploadResponse:
    folder = workdir / subdir
    folder.mkdir(parents=True, exist_ok=True)

    dst = folder / file.filename
    with dst.open("wb") as w:
        w.write(file.file.read())

    return UploadResponse(
        filename=file.filename,
        saved_path=str(dst.relative_to(workdir)),
        size=dst.stat().st_size,
        uploaded_at=datetime.utcnow(),
    )

def _file_info(path: Path) -> Dict[str, str]:
    stat = path.stat()
    return {
        "filename": path.name,
        "file_path": str(path.resolve()),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "extension": (path.suffix or "unknown").lstrip("."),
    }

# ──────────────────────────────────────────────
# node: file list
# ──────────────────────────────────────────────
@router.get("/", summary="list upload dir files")
async def list_files(workdir: Path = Depends(get_workdir)) -> Dict[str, List[Dict[str, str]]]:
    base = workdir
    tree: Dict[str, List[Dict[str, str]]] = {}
    if base.exists():
        for folder in base.iterdir():
            if folder.is_dir():
                tree[folder.name] = [_file_info(p) for p in folder.iterdir() if p.is_file()]
    return tree

# ──────────────────────────────────────────────
# node: upload images
# ──────────────────────────────────────────────
@router.post(
    "/image",
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
# node: upload videos
# ──────────────────────────────────────────────
@router.post(
    "/video",
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
# node: upload models
# ──────────────────────────────────────────────
@router.post(
    "/model",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="upload models",
)
async def upload_model(
    file: UploadFile = File(...),
    workdir: Path = Depends(get_workdir),
):
    return _save(file, workdir, "models")

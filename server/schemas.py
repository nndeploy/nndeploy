# schemas.py

from __future__ import annotations
from pydantic import BaseModel, Field, RootModel
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

JsonDict = Dict[str, Any]

# -------------- api / queue ------------------
class EnqueueRequest(RootModel):
    root: Dict[str, Any]

class EnqueueResponse(BaseModel):
    task_id: str

class QueueItem(BaseModel):
    id: str
    priority: int

class QueueStateResponse(BaseModel):
    running: Dict[int, QueueItem]
    pending: List[Tuple[int, QueueItem]]

class HistoryItem(BaseModel):
    task: Dict[str, Any]
    status: Dict[str, Any]

class ProgressPayload(BaseModel):
    type: str
    data: Dict[str, Any]

class NodeListResponse(BaseModel):
    flag: str
    message: str
    result: List[Any]

class WorkFlowListResponse(BaseModel):
    flag: str
    message: str
    result: List[Any]

class WorkFlowSaveResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, str]

class WorkFlowLoadResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class WorkFlowDeleteResponse(BaseModel):
    flag: str
    message: str

class UploadResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class DeleteResponse(BaseModel):
    flag: str
    message: str

class FileListResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class PreviewPayload(BaseModel):
    type: Literal["preview"]
    data: Dict[str, Any]

class ParamTypeResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

UploadResponse.model_rebuild()

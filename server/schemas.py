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
    flag: str
    message: str
    result: Dict[str, Any]

class QueueItem(BaseModel):
    id: str
    priority: int

class QueueStateResult(BaseModel):
    running: List[Dict[str, Any]]
    pending: List[Dict[str, Any]]
    dispatched: List[Dict[str, Any]] = []

class QueueStateResponse(BaseModel):
    flag: str
    message: str
    result: QueueStateResult

class HistoryItem(BaseModel):
    items: List[Dict[str, Any]]
    total: int

class HistoryResponse(BaseModel):
    flag: str
    message: str
    result: HistoryItem

class ProgressPayload(BaseModel):
    type: str
    data: Dict[str, Any]

class NodeListResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class WorkFlowListResponse(BaseModel):
    flag: str
    message: str
    result: list[dict]

class WorkFlowSaveResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, str]

class WorkFlowLoadResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class TemplateDirListResponse(BaseModel):
    flag: str
    message: str
    result: list[str]

class TemplateJsonListResponse(BaseModel):
    flag: str
    message: str
    result: list[dict]

class TemplateLoadResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class TemplateDownloadRequest(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class TemplateDownloadResponse(BaseModel):
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
    result: list[Dict]

class FileInfoResponse(BaseModel):
    flag: str
    message: str
    result: dict

class PreviewPayload(BaseModel):
    type: Literal["preview"]
    data: Dict[str, Any]

class ParamTypeResponse(BaseModel):
    flag: str
    message: str
    result: Dict[str, Any]

class WsPreviewPayload(BaseModel):
    type: str
    result: str

UploadResponse.model_rebuild()

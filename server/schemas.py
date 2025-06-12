# schemas.py

from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

JsonDict = Dict[str, Any]

# -------------- api / queue ------------------
class EnqueueRequest(BaseModel):
    graph_json: JsonDict = Field(..., description="nndeploy graph in JSON")
    id: Optional[str] = Field(None, description="task id")
    priority: int = Field(0, description="priority")

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
    outputs: Dict[str, Any]
    status: Dict[str, Any]

class ProgressPayload(BaseModel):
    type: str
    data: Dict[str, Any]

class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    size: int
    uploaded_at: datetime

UploadResponse.model_rebuild()

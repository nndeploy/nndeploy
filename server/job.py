# job.py

from enum import Enum
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import uuid

class JobStatus(Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"

@dataclass
class Job:
    graph: dict[str, Any]
    priority: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: JobStatus = JobStatus.queued
    submit_time: datetime = field(default_factory=datetime.utcnow)
    progress: float = 0.0
    error: str | None = None

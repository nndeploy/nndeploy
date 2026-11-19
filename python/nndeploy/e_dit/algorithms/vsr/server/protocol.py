# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang VSR/OpenAI video interfaces; see ../LICENSE.SGLang.
import time
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ..geometry import parse_resolution

class VideoResponse(BaseModel):
    id: str
    object: str = "video"
    model: str = "sora-2"
    status: str = "queued"
    progress: int = 0
    created_at: int = Field(default_factory=lambda: int(time.time()))
    size: str = ""
    seconds: str = "4"
    quality: str = "standard"
    url: Optional[str] = None
    remixed_from_video_id: Optional[str] = None
    completed_at: Optional[int] = None
    expires_at: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    file_path: Optional[str] = None
    peak_memory_mb: Optional[float] = None
    inference_time_s: Optional[float] = None

class VideoRepairMinioConfig(BaseModel):
    endpoint: str
    bucket_name: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: str = "us-east-1"
    provider: Optional[str] = None
    prefix: Optional[str] = None

class VideoListResponse(BaseModel):
    data: List[VideoResponse]
    object: str = "list"
_VIDEO_REPAIR_FIELD_ALIASES = {
    "taskId": "task_id",
    "callbackUrl": "callback_url",
    "videoUrl": "video_url",
    "maskUrl": "mask_url",
    "referenceImagePath": "reference_image_path",
    "referenceImageUrl": "reference_image_url",
    "refFrameIdx": "ref_frame_idx",
    "bridgeOverlap": "bridge_overlap",
    "decodeMode": "decode_mode",
    "bboxExpandScale": "bbox_expand_scale",
    "useClip": "use_clip",
    "clipPreprocess": "clip_preprocess",
    "teacacheThresh": "teacache_thresh",
    "teacacheStartSkipping": "teacache_start_skipping",
    "teacacheEndSkipping": "teacache_end_skipping",
    "minioConfig": "minio_config",
    "outputObjectKey": "output_object_key",
}

_VIDEO_REPAIR_MINIO_FIELD_ALIASES = {
    "bucket": "bucket_name",
    "bucketName": "bucket_name",
    "rootUser": "access_key",
    "accessKey": "access_key",
    "rootPass": "secret_key",
    "secretKey": "secret_key",
    "useSSL": "secure",
}

def _normalize_aliases(
    payload: Dict[str, Any], aliases: Dict[str, str]
) -> Dict[str, Any]:
    normalized = dict(payload)
    for alias, canonical in aliases.items():
        if alias not in payload:
            continue
        normalized[canonical] = payload[alias]
        if alias != canonical:
            normalized.pop(alias, None)
    return normalized

def _normalize_video_repair_payload(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    payload = _normalize_aliases(body, _VIDEO_REPAIR_FIELD_ALIASES)
    minio_config = payload.get("minio_config")
    if minio_config is not None:
        if not isinstance(minio_config, dict):
            raise ValueError("minioConfig must be a JSON object")
        payload["minio_config"] = _normalize_aliases(
            minio_config, _VIDEO_REPAIR_MINIO_FIELD_ALIASES
        )
    return payload

def _job_error_message(job: Dict[str, Any]) -> str:
    error = job.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    if isinstance(error, str):
        return error
    return "failed"

def _job_reason(job: Dict[str, Any]) -> str | None:
    if job.get("status") not in ("failed", "cancelled", "expired"):
        return None
    reason = job.get("reason")
    if reason:
        return str(reason)
    return _job_error_message(job)
class VideoRestorationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,128}$")
    video_input_path: str | None = None
    video_url: str | None = None
    callback_url: str | None = None
    minio_config: VideoRepairMinioConfig | None = None
    output_object_key: str | None = None
    output_bucket: str | None = None
    timeout: int = -1
    target_resolution: str | None = None
    long_edge: int | None = Field(default=None, gt=0)
    tile_t: int | None = Field(default=None, gt=0)
    tile_h: int | None = Field(default=None, gt=0)
    tile_w: int | None = Field(default=None, gt=0)
    temporal_overlap: int | None = Field(default=None, ge=0)
    spatial_overlap: int | None = Field(default=None, ge=0)
    color_ref: Literal["global", "chunk", "none"] | None = None
    color_ref_samples: int | None = Field(default=None, ge=0)
    crf: int | None = Field(default=None, ge=0, le=51)
    preserve_audio: bool = True
    read_queue: int | None = Field(default=None, ge=1, le=16)
    write_queue: int | None = Field(default=None, ge=1, le=16)
    gpu_postprocess: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def aliases(cls, value):
        return _normalize_video_repair_payload(value)

    @model_validator(mode="after")
    def validate_input(self):
        if bool(self.video_input_path) == bool(self.video_url):
            raise ValueError("Provide exactly one of video_input_path or videoUrl")
        if self.target_resolution:
            h, w = parse_resolution(self.target_resolution)
            if h <= 0 or w <= 0 or h % 2 or w % 2:
                raise ValueError("target_resolution must be positive even HxW")
        if self.callback_url and not self.callback_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("callbackUrl must use http or https")
        if self.output_object_key is not None and self.minio_config is None:
            raise ValueError("output_object_key requires minioConfig")
        return self

def _validate_tiling(req, cfg):
    def effective(name, default):
        value = getattr(req, name)
        return getattr(cfg, default) if value is None else value

    t, h, w = [effective(name, name) for name in ("tile_t", "tile_h", "tile_w")]
    temporal = effective("temporal_overlap", "temporal_overlap")
    spatial = effective("spatial_overlap", "spatial_overlap")
    if (t - 1) % 4 or h % 32 or w % 32:
        raise ValueError("tile_t must be 4n+1; tile_h/tile_w must be multiples of 32")
    if temporal >= t or spatial >= min(h, w):
        raise ValueError("overlap must be smaller than the corresponding tile")

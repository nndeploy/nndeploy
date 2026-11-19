# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang multimodal_gen/runtime/entrypoints/openai.
"""VideoEdit HTTP contract; deliberately independent of model/runtime imports."""
import time
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field, StrictFloat, StrictInt, model_validator

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


_REMOVED_VIDEOEDIT_REQUEST_FIELDS = frozenset(
    {
        "chunks",
        "drop_reference_frame",
        "dropReferenceFrame",
        "keep_intermediate_windows",
        "keepIntermediateWindows",
        "overlap_commit_mode",
        "overlapCommitMode",
        "tail_padding_mode",
        "tailPaddingMode",
        "mask_downsample_mode",
        "maskDownsampleMode",
        "use_repaired_context",
        "useRepairedContext",
        "init_latent_mode",
        "initLatentMode",
        "strength",
        "vary_seed_by_window",
        "varySeedByWindow",
        "generator_device",
        "generatorDevice",
    }
)


def default_video_repair_output_object_key(
    request_id: str, now: datetime | None = None, extension: str = ".mp4"
) -> str:
    now = now or datetime.now()
    extension = extension if extension.startswith(".") else f".{extension}"
    return f"{now:%Y/%m/%d}/{now:%H%M%S}_{request_id}{extension}"


class VideoRepairRequest(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def reject_removed_videoedit_fields(cls, value):
        if isinstance(value, dict):
            removed = sorted(_REMOVED_VIDEOEDIT_REQUEST_FIELDS.intersection(value))
            if removed:
                raise ValueError(
                    "removed VideoEdit request fields are not supported: "
                    + ", ".join(removed)
                )
        return value

    task_id: Optional[str] = None
    timeout: int = -1
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = None

    video_input_path: Optional[str] = None
    mask_input_path: Optional[str] = None
    video_url: Optional[str] = None
    mask_url: Optional[str] = None
    reference_image_path: Optional[str] = None
    reference_image_url: Optional[str] = None

    callback_url: Optional[str] = None
    minio_config: Optional[VideoRepairMinioConfig] = None
    output_storage: str = "local"
    output_path: Optional[str] = None
    output_bucket: Optional[str] = None
    output_object_key: Optional[str] = None

    num_frames: Optional[int] = -1
    ref_frame_idx: StrictInt = 0
    bridge_overlap: StrictInt = 5
    infer_len: int = 49
    overlap: int = 5
    num_inference_steps: int = 40
    guidance_scale: float = 5.0
    seed: int = 42
    dtype: str = "bf16"
    dynamic_cfg: bool = True
    dynamic_cfg_max_step: int = 15
    dynamic_cfg_min: float = 1.0

    bbox_padding: int = 0
    bbox_expand_scale: float = 0.3
    dilate_px: int = 8
    mask_scale: float = 1.0
    feather_px: int = 8
    adain_boundary_dilate: int = 0
    crop_edge_feather: int = Field(default=0, ge=0)
    chunk_bbox_mode: Literal["tight", "fixed_size", "global"] = "tight"
    stabilize_mask_union: bool = False
    stabilize_mask_shape: bool = False
    stabilize_smooth_window: int = Field(default=5, ge=1)
    preserve_audio: bool = True
    enable_paste_back: bool = True
    save_crop_only: bool = False
    use_clip: bool = True
    clip_preprocess: Literal["diffuser", "diffsynth"] = "diffuser"
    decode_mode: Literal["eager", "stream"] = "stream"
    enable_teacache: bool = True
    teacache_thresh: float = 0.3
    teacache_start_skipping: Union[StrictInt, StrictFloat] = 5
    teacache_end_skipping: Union[StrictInt, StrictFloat] = 1.0
    enable_frame_interpolation: bool = False
    frame_interpolation_exp: int = 1
    frame_interpolation_scale: float = 1.0
    frame_interpolation_model_path: Optional[str] = None
    enable_upscaling: bool = False
    upscaling_model_path: Optional[str] = None
    upscaling_scale: int = 4
    output_quality: Optional[str] = "default"
    output_compression: Optional[int] = None
    perf_dump_path: Optional[str] = None


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
    "chunkBboxMode": "chunk_bbox_mode",
    "cropEdgeFeather": "crop_edge_feather",
    "stabilizeMaskUnion": "stabilize_mask_union",
    "stabilizeMaskShape": "stabilize_mask_shape",
    "stabilizeSmoothWindow": "stabilize_smooth_window",
    "preserveAudio": "preserve_audio",
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


def submit_response(code: int, message: str) -> Dict[str, Any]:
    response = {"code": int(code), "message": str(message)}
    if int(code) == 1:
        response["reason"] = str(message)
    return response


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


def normalize_request(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if 'inputs' in body:
        inputs = body['inputs']
        parameters = body.get('parameters', body.get('parameter', {}))
        if not isinstance(inputs, dict) or not isinstance(parameters, dict):
            raise ValueError('inputs and parameters must be JSON objects')
        inputs = inputs.get('request', inputs)
        if not isinstance(inputs, dict):
            raise ValueError('inputs.request must be a JSON object')
        body = {**inputs, **parameters}
    payload = _normalize_aliases(body, _VIDEO_REPAIR_FIELD_ALIASES)
    minio_config = payload.get("minio_config")
    if minio_config is not None:
        if not isinstance(minio_config, dict):
            raise ValueError("minioConfig must be a JSON object")
        payload["minio_config"] = _normalize_aliases(
            minio_config, _VIDEO_REPAIR_MINIO_FIELD_ALIASES
        )
    return payload


def validate_request(req: VideoRepairRequest, *, storage_enabled: bool = False) -> None:
    if req.stabilize_mask_union and req.stabilize_mask_shape:
        raise ValueError("Choose at most one mask stabilization mode")
    if not req.task_id:
        raise ValueError("taskId is required")
    if req.timeout == 0 or req.timeout < -1:
        raise ValueError("timeout must be positive or -1")
    if not (req.video_input_path or req.video_url):
        raise ValueError("videoUrl or video_input_path is required")
    if not (req.mask_input_path or req.mask_url):
        raise ValueError("maskUrl or mask_input_path is required")
    if not (req.reference_image_path or req.reference_image_url):
        raise ValueError(
            "referenceImageUrl or reference_image_path is required"
        )
    if req.num_frames is not None and (
        req.num_frames == 0 or req.num_frames < -1
    ):
        raise ValueError("num_frames must be a positive integer or -1")
    if req.ref_frame_idx < 0:
        raise ValueError("ref_frame_idx must be non-negative")
    if (
        req.num_frames is not None
        and req.num_frames > 0
        and req.ref_frame_idx >= req.num_frames
    ):
        raise ValueError("ref_frame_idx must be smaller than num_frames")
    if req.infer_len < 1 or (req.infer_len - 1) % 4 != 0:
        raise ValueError("infer_len must be >= 1 and satisfy (infer_len - 1) % 4 == 0")
    if not (0 <= req.overlap < req.infer_len):
        raise ValueError("overlap must be in [0, infer_len)")
    if req.bridge_overlap < 1 or (req.bridge_overlap - 1) % 4 != 0:
        raise ValueError(
            "bridge_overlap must be >= 1 and satisfy (bridge_overlap - 1) % 4 == 0"
        )
    if req.minio_config is None and (
        req.output_storage == "s3" or req.output_object_key is not None
    ):
        if not storage_enabled:
            raise ValueError("minioConfig is required for S3 output")



def parse_request(payload, *, storage_enabled=False):
    request = VideoRepairRequest.model_validate(normalize_request(payload))
    validate_request(request, storage_enabled=storage_enabled)
    return request

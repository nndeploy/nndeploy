# SPDX-License-Identifier: Apache-2.0
# Sampling translation adapted from SGLang multimodal_gen VideoEdit API.
"""Resident native graphs behind the existing Server Worker."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import hashlib
import time
from .schemas import VideoRepairRequest, VideoRepairMinioConfig, default_video_repair_output_object_key
from .policy import resolve_variant, apply_variant_policy


def build_videoedit_teacache_params(**options):
    from ..videoedit_sampling import build_videoedit_teacache_params as build
    return build(**options)

def sampling_kwargs(
    req: VideoRepairRequest,
    *,
    request_id: str,
    video_input_path: str,
    mask_input_path: str,
    reference_image_path: str,
    output_dir: str,
    output_file_name: str,
    resolved_num_frames: int,
    progress_path: str,
) -> Dict[str, Any]:
    """Build the one-to-one OpenAI repair to VideoEdit sampling contract."""
    return {
        "request_id": request_id,
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "video_input_path": video_input_path,
        "mask_input_path": mask_input_path,
        "reference_image_path": reference_image_path,
        "output_path": output_dir,
        "output_file_name": output_file_name,
        "num_frames": resolved_num_frames,
        "ref_frame_idx": req.ref_frame_idx,
        "bridge_overlap": req.bridge_overlap,
        "infer_len": req.infer_len,
        "overlap": req.overlap,
        "num_inference_steps": req.num_inference_steps,
        "guidance_scale": req.guidance_scale,
        "seed": req.seed,
        "dtype": req.dtype,
        "dynamic_cfg": req.dynamic_cfg,
        "dynamic_cfg_max_step": req.dynamic_cfg_max_step,
        "dynamic_cfg_min": req.dynamic_cfg_min,
        "bbox_padding": req.bbox_padding,
        "bbox_expand_scale": req.bbox_expand_scale,
        "dilate_px": req.dilate_px,
        "mask_scale": req.mask_scale,
        "feather_px": req.feather_px,
        "adain_boundary_dilate": req.adain_boundary_dilate,
        "crop_edge_feather": req.crop_edge_feather,
        "chunk_bbox_mode": req.chunk_bbox_mode,
        "stabilize_mask_union": req.stabilize_mask_union,
        "stabilize_mask_shape": req.stabilize_mask_shape,
        "stabilize_smooth_window": req.stabilize_smooth_window,
        "preserve_audio": req.preserve_audio,
        "enable_paste_back": req.enable_paste_back,
        "save_crop_only": req.save_crop_only,
        "use_clip": req.use_clip,
        "clip_preprocess": req.clip_preprocess,
        "decode_mode": req.decode_mode,
        "enable_teacache": req.enable_teacache,
        "teacache_params": build_videoedit_teacache_params(
            teacache_thresh=req.teacache_thresh,
            start_skipping=req.teacache_start_skipping,
            end_skipping=req.teacache_end_skipping,
        ),
        "perf_dump_path": req.perf_dump_path,
        "progress_path": progress_path,
    }



def object_client(config):
    import boto3
    from botocore.config import Config
    endpoint = config.endpoint.strip().rstrip('/')
    if not endpoint:
        raise ValueError('S3 endpoint must not be empty')
    if not endpoint.startswith(('http://', 'https://')):
        endpoint = ('https://' if config.secure else 'http://') + endpoint
    return boto3.client('s3', endpoint_url=endpoint, region_name=config.region or 'us-east-1',
        aws_access_key_id=config.access_key, aws_secret_access_key=config.secret_key,
        config=Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 3})), endpoint


def object_key(request, path):
    from urllib.parse import urlparse
    config = request.minio_config
    key = request.output_object_key
    extension = Path(path).suffix
    if key:
        key = str(Path(key).with_suffix(extension))
    else:
        key = default_video_repair_output_object_key(request.task_id, extension=extension)
    key = key.strip().lstrip('/')
    if not key or '..' in key.split('/'):
        raise ValueError('Invalid output object key')
    hostname = urlparse(('https://' if '://' not in config.endpoint else '') + config.endpoint).hostname or ''
    provider = (config.provider or '').strip().lower()
    cos = provider == 'cos' if provider else (hostname.endswith('.myqcloud.com') and
                                             (hostname.startswith('cos.') or '.cos.' in hostname))
    prefix = (config.prefix or '').strip().strip('/') if cos else ''
    if prefix and key != prefix and not key.startswith(prefix + '/'):
        key = prefix + '/' + key
    if '..' in key.split('/'):
        raise ValueError('Invalid output object prefix')
    return key


@dataclass
class HandlerFactory:
    models: dict
    secrets_directory: str
    key_env: str | None = None
    max_input_bytes: int = 5 * 1024**3
    graph_factory: Any = None
    audit_directory: str | None = None

    def __call__(self):
        return VideoEditHandler(self)


class VideoEditHandler:
    reports_progress = True

    def __init__(self, config):
        self.config = config
        self.graphs = {}

    def init(self):
        from nndeploy.base import StatusCode
        from nndeploy.e_dit import DiTParams, build_graph
        import logging
        failures = []
        self.service_status = {}
        for variant, params in self.config.models.items():
            graph = None
            try:
                graph = self.config.graph_factory(variant, params) if self.config.graph_factory else build_graph(DiTParams(**params))
                status = graph.init()
                if status != StatusCode.Ok:
                    raise RuntimeError(f'VideoEdit {variant} model initialization failed')
                self.graphs[variant] = graph
                self.service_status[variant] = True
            except Exception as error:
                self.service_status[variant] = False
                failures.append(error)
                logging.exception('VideoEdit %s backend initialization failed', variant)
                if graph is not None:
                    graph.deinit()
        if not self.graphs:
            raise RuntimeError('No VideoEdit backend initialized successfully') from failures[0]

    def deinit(self):
        failures = []
        for graph in self.graphs.values():
            try:
                graph.deinit()
            except BaseException as error:
                failures.append(error)
        self.graphs.clear()
        if failures:
            raise failures[0]

    def deploy(self, inputs):
        from nndeploy.server.request_context import current_request, FileArtifact
        from nndeploy.server.storage.sources import materialize
        from nndeploy.server.secret_store import SecretStore
        from ..video.preprocess import resolve_videoedit_num_frames
        from ..video.request_audit import VideoEditRequestAudit
        context = current_request()
        request = VideoRepairRequest.model_validate(inputs['request'])
        variant = context.metadata.get('variant') or resolve_variant(request.model)
        request = apply_variant_policy(request, variant)
        if variant not in self.graphs:
            raise ValueError(f'VideoEdit {variant} backend is unavailable')
        audit = VideoEditRequestAudit(self.config.audit_directory, task_id=request.task_id)
        audit.update(status='running', effective_request=request.model_dump())
        started = time.monotonic()
        client, endpoint = None, None
        preparing = True
        try:
            if inputs.get('storage_secret'):
                raw = SecretStore(self.config.secrets_directory, key_env=self.config.key_env).open(inputs['storage_secret'])
                request.minio_config = VideoRepairMinioConfig.model_validate(raw)
                client, endpoint = object_client(request.minio_config)
            paths = {}
            for name, url_name, extension in (
                ('video_input_path', 'video_url', '.mp4'), ('mask_input_path', 'mask_url', '.mp4'),
                ('reference_image_path', 'reference_image_url', '.png')):
                from urllib.parse import urlparse
                source = getattr(request, url_name) or getattr(request, name)
                suffix = Path(urlparse(source).path).suffix or extension
                paths[name] = materialize(source, context.directory / (name + suffix), check=context.check,
                    max_bytes=self.config.max_input_bytes, object_client=client,
                    default_bucket=request.minio_config.bucket_name if client else None, endpoint=endpoint)
            context.check()
            if request.enable_frame_interpolation and (request.frame_interpolation_exp < 1 or request.frame_interpolation_scale <= 0):
                raise ValueError('Frame interpolation requires exp >= 1 and scale > 0')
            if request.enable_upscaling and request.upscaling_scale <= 0:
                raise ValueError('Upscaling requires scale > 0')
            frames = resolve_videoedit_num_frames(request.num_frames or -1, paths['video_input_path'], paths['mask_input_path'])
            extension = Path(paths['video_input_path']).suffix.lower() or '.mp4'
            if extension not in ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'):
                extension = '.mp4'
            stem = request.task_id if '/' not in request.task_id and '\\' not in request.task_id else hashlib.sha256(request.task_id.encode()).hexdigest()
            output = Path(request.output_path) if request.output_path else context.directory
            if output.suffix.lower() in ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi'):
                output = output.with_suffix(extension)
            else:
                output = output / (stem + extension)
            output = output.absolute()
            sampling = sampling_kwargs(request, request_id=context.task_id, **paths,
                output_dir=str(output.parent), output_file_name=output.name, resolved_num_frames=frames,
                progress_path=str(context.directory/'progress.json'))
            audit.update(resolved={**paths, 'output_path': str(output), 'num_frames': frames})
            preparing = False
            context.publish({'progress': 1, 'stage': 'editing'})
            from nndeploy.dag.execution import ExecutionContext, execution_scope
            from ..video.progress import read_videoedit_progress
            def publish(value):
                # Generic node fractions restart for every window. Prefer the
                # source's total-window/total-step percentage for the HTTP API.
                detail = value if isinstance(value.get('progress'), (int, float)) else (
                    read_videoedit_progress(sampling['progress_path']) or {'progress': 1, 'stage': 'editing'})
                context.publish({**value, **detail})
            with execution_scope(ExecutionContext(cancelled=lambda: (context.check() or False), publish=publish)):
                paths = self.graphs[variant].infer_files(output, **sampling)
            from .postprocess import process_files
            from nndeploy.e_dit.config.server_args import config_scope
            from nndeploy.e_dit import DiTParams
            with config_scope(DiTParams(**self.config.models[variant]).build_config()):
                paths = process_files(paths, request, context)
            context.check()
            artifacts = []
            key = url = None
            for index, path in enumerate(paths):
                media_type = {'.mp4': 'video/mp4', '.m4v': 'video/x-m4v', '.webm': 'video/webm', '.avi': 'video/x-msvideo',
                              '.mov': 'video/quicktime', '.mkv': 'video/x-matroska'}.get(Path(path).suffix, 'application/octet-stream')
                item = {'role': 'primary_video' if index == 0 else 'crop_video', 'file': FileArtifact(path, media_type, include_local_path=True)}
                if client:
                    target = object_key(request, path)
                    if index:
                        target = str(Path(target).with_name(Path(target).stem + '_crop_only' + Path(target).suffix))
                    bucket = request.output_bucket or request.minio_config.bucket_name
                    context.publish({'progress': 99, 'stage': 'uploading'})
                    client.upload_file(str(path), bucket, target, Callback=lambda count: context.check(), ExtraArgs={'ContentType': media_type})
                    item.update(object_key=target, url=f'{endpoint}/{bucket}/{target}')
                    if index == 0:
                        key, url = target, item['url']
                artifacts.append(item)
            result = dict(artifacts=artifacts, output_object_key=key, url=url,
                file_path=str(output) if request.output_path else None,
                inference_time_s=time.monotonic()-started, variant=variant)
            audit.update(status='inference_completed', result={k:v for k,v in result.items() if k != 'artifacts'})
            return result
        except BaseException as error:
            audit.update(status='failed', error=type(error).__name__)
            if preparing and isinstance(error, (ValueError, FileNotFoundError)):
                from nndeploy.server.errors import ServerError
                raise ServerError(422, 'invalid_videoedit_input', str(error)) from error
            raise
        finally:
            if client is not None:
                client.close()

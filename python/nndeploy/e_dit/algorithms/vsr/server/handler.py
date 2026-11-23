"""Resident VSR execution including input preparation and output delivery."""
import asyncio
import time

from nndeploy.server.errors import ServerError
from nndeploy.server.request_context import current_request, FileArtifact
from nndeploy.server.secret_store import SecretStore
from nndeploy.server.storage.sources import materialize
from .protocol import VideoRestorationRequest, _validate_tiling
from .storage import RequestCloudStorage, cloud_storage
from ..geometry import parse_resolution, resize_to_long_edge
from ..video_io import probe_video


class VSRHandler:
    reports_progress = True

    def __init__(self, params, secrets_directory, key_env=None):
        self.params, self.secrets_directory, self.key_env = params, secrets_directory, key_env
        self.graph = None

    def init(self):
        from nndeploy.e_dit import build_graph
        from nndeploy.base import StatusCode
        self.graph = build_graph(self.params)
        if self.graph.init() != StatusCode.Ok:
            raise RuntimeError('VSR model initialization failed')

    def deploy(self, inputs):
        from nndeploy.e_dit import DiTParams
        context = current_request()
        storage = None
        try:
            body = dict(inputs['request'])
            if inputs.get('storage_secret'):
                body['minio_config'] = SecretStore(self.secrets_directory, key_env=self.key_env).open(inputs['storage_secret'])
            request = VideoRestorationRequest.model_validate(body)
            config = DiTParams(**self.params).build_config()
            _validate_tiling(request, config)
            if request.minio_config:
                storage = RequestCloudStorage(request.minio_config)
            context.publish({'progress': 0, 'stage': 'preparing'})
            video = materialize(request.video_input_path or request.video_url, context.directory / 'input.mp4',
                check=context.check, object_client=storage.client if storage else None,
                default_bucket=storage.bucket_name if storage else None,
                endpoint=storage.endpoint_url if storage else None)
            fps, frames, h, w = probe_video(video)
            target = request.target_resolution
            if target is None and request.long_edge is None:
                target = config.target_resolution
            h, w = parse_resolution(target) if target else resize_to_long_edge(h, w, request.long_edge or config.long_edge)
            media = {'size': f'{w}x{h}', 'seconds': str(frames / fps)}
            context.publish({'progress': 1, 'stage': 'restoring', **media})
            sampling = request.model_dump(exclude_none=True, exclude={
                'task_id', 'video_url', 'video_input_path', 'callback_url', 'minio_config',
                'output_object_key', 'output_bucket', 'timeout'})
            started = time.monotonic()
            path = self.graph.infer_file(video, str(context.directory / 'output.mp4'), **sampling)
            duration = time.monotonic() - started
            context.check()
            uploader = storage or cloud_storage
            url = None
            if storage or uploader.is_enabled():
                context.publish({'progress': 99, 'stage': 'uploading', **media})
                key = request.output_object_key or f'{context.task_id}.mp4'
                url = asyncio.run(uploader.upload_file(path, key, bucket_name=request.output_bucket))
                if not url:
                    raise RuntimeError('Video upload failed')
            context.check()
            return {**media, 'inference_time_s': duration, 'url': url,
                    'artifacts': [{'role': 'primary_video', 'file': FileArtifact(path, 'video/mp4', include_local_path=True)}]}
        except ServerError:
            raise
        except Exception as error:
            # Report a failed Task, never a successful result containing an error.
            raise ServerError(500, 'vsr_execution_failed', str(error)) from error
        finally:
            if storage is not None:
                storage.client.close()

    def deinit(self):
        if self.graph is not None:
            self.graph.deinit()
            self.graph = None

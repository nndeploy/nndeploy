"""VSR protocol adapter over the common durable Server task lifecycle."""
from pathlib import Path
import time
import uuid

from fastapi import HTTPException

from nndeploy.server.errors import ServerError
from nndeploy.server.secret_store import SecretStore
from nndeploy.server.service_tasks import ServiceTasks
from .callbacks import task_response
from .protocol import VideoRestorationRequest, VideoResponse, _validate_tiling


class VSRService:
    def __init__(self, server, config, *, name='vsr', key_env=None, memory_estimator=None):
        self.server, self.config, self.name = server, config, name
        self.tasks = ServiceTasks(server, name, memory_estimator=memory_estimator)
        self.group = 'vsr:' + name
        self.secrets = SecretStore(server.data_dir / 'request-secrets', key_env=key_env)

    async def submit(self, body, caller):
        caller.require('deploy', self.name)
        try:
            req = VideoRestorationRequest.model_validate(body)
            _validate_tiling(req, self.config)
            if await self.tasks.busy(self.group):
                return {'code': 2, 'message': 'A task is running.'}
            identity = req.task_id or str(uuid.uuid4())
            if await self.tasks.read(identity, caller, 'deploy'):
                return {'code': 1, 'message': 'taskId already exists', 'id': identity}
            if req.video_input_path and not Path(req.video_input_path).is_file():
                return {'code': 1, 'message': 'Input video does not exist'}
            metadata = dict(external_id=identity, model='vsr',
                deadline=time.time() + req.timeout if req.timeout > 0 else None,
                callback_formatter='nndeploy.e_dit.algorithms.vsr.server.callbacks:task_callback',
                callback_max_attempts=3, callback_retry_delays=[1, 2], callback_timeout=10,
                callback_redirects=True)
            inputs = {'request': req.model_dump(exclude={'minio_config'}),
                      'storage_secret': self.secrets.seal(req.minio_config.model_dump()) if req.minio_config else None}
            await self.tasks.submit(inputs, caller, metadata=metadata, callback_url=req.callback_url)
            return {'code': 0, 'message': 'Task submitted', 'id': identity}
        except ServerError as error:
            if error.code in ('service_busy', 'queue_full', 'server_full'):
                return {'code': 2, 'message': 'A task is running.'}
            if error.code == 'duplicate_task':
                return {'code': 1, 'message': 'taskId already exists', 'id': identity}
            if error.status_code in (401, 403):
                raise
            return {'code': 1, 'message': str(error)}
        except (ValueError, TypeError) as error:
            return {'code': 1, 'message': str(error)}

    def response(self, record):
        return VideoResponse(**task_response(record))

    def progress(self, record):
        value = task_response(record)
        return {key: value[key] for key in ('id', 'status', 'progress', 'file_path', 'url', 'error', 'reason',
                                           'callback_status', 'callback_error', 'callback_attempts')}

    def check_content(self, record):
        url = (record.get('outputs') or {}).get('url')
        if url:
            raise HTTPException(400, f'Video has been uploaded to cloud storage. Please use the cloud URL: {url}')
        if record['state'] != 'succeeded':
            raise HTTPException(404, 'Generation is still in-progress')

    async def health(self):
        return {'status': 'ok' if self.tasks.status()['ready'] else 'unavailable'}

"""Original VideoEdit HTTP paths over nndeploy's durable task execution."""
import asyncio
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request, Query

from nndeploy.server.dependencies import limited_identity
from nndeploy.server.errors import ServerError
from .schemas import VideoRepairRequest, VideoResponse, normalize_request, parse_request, submit_response
from .policy import resolve_variant, apply_variant_policy
from .callbacks import task_response


class VideoEditAPI:
    def __init__(self, server, name, variants, *, contract='gateway', storage=None, key_env=None, memory_estimator=None):
        if contract not in ('backend', 'gateway'):
            raise ValueError('VideoEdit contract must be backend or gateway')
        self.server, self.name, self.variants, self.contract = server, name, set(variants), contract
        from nndeploy.server.service_tasks import ServiceTasks
        self.tasks = ServiceTasks(server, name, memory_estimator=memory_estimator)
        self.scope = self.tasks.scope
        self.group = 'videoedit:' + name
        self.storage, self.key_env = storage, key_env
        self.secrets_directory = str(server.data_dir/'request-secrets')

    def metadata(self, request, variant):
        return dict(external_id=request.task_id, variant=variant, model=request.model, contract=self.contract,
            deadline=time.time()+request.timeout if request.timeout > 0 else None,
            callback_formatter='nndeploy.e_dit.algorithms.wan.server.callbacks:task_callback',
            callback_max_attempts=3, callback_retry_delays=[1, 2], callback_timeout=10, callback_redirects=True,
            progress_callbacks=True, progress_interval=5, scope=self.scope)

    async def read(self, identity, caller, permission='tasks:read'):
        record = await self.tasks.read(identity, caller, permission)
        if record is None:
            raise HTTPException(404, 'Video not found')
        return record

    async def failed_submission(self, payload, reason, caller):
        try:
            value = normalize_request(payload)
            external = value.get('task_id')
            if not external:
                return
            from nndeploy.server.callbacks import validate_url
            callback = validate_url(value.get('callback_url'))
            service = dict(scope=self.scope, external_id=external,
                callback_formatter='nndeploy.e_dit.algorithms.wan.server.callbacks:task_callback',
                callback_max_attempts=3, callback_retry_delays=[1, 2], callback_timeout=10, callback_redirects=True)
            await self.tasks.record_failure(caller, service, reason, callback)
        except ServerError as error:
            if error.code != 'duplicate_task':
                raise
        except (ValueError, TypeError):
            return

    async def submit(self, body, caller):
        caller.require('deploy', self.name)
        if await self.tasks.busy(self.group):
            return submit_response(2, 'A task is running.')
        try:
            payload = normalize_request(body)
            if self.contract == 'gateway':
                payload['task_id'] = str(payload.get('task_id') or 'videoedit-' + uuid.uuid4().hex).strip()
            request = parse_request(payload, storage_enabled=self.storage is not None)
            variant = (next(iter(self.variants)) if self.contract == 'backend' and len(self.variants) == 1
                       else resolve_variant(request.model))
            if not self.backend_status().get(variant):
                raise HTTPException(503, f'VideoEdit {variant} backend is unavailable')
            if self.contract == 'gateway':
                request = apply_variant_policy(request, variant)
            service = self.metadata(request, variant)
            storage = request.minio_config or self.storage
            inputs = {'request': request.model_dump(exclude={'minio_config'})}
            if storage is not None:
                from nndeploy.server.secret_store import SecretStore
                inputs['storage_secret'] = SecretStore(self.secrets_directory, key_env=self.key_env).seal(
                    storage.model_dump() if hasattr(storage, 'model_dump') else storage)
            else:
                inputs['storage_secret'] = None
            identity = await self.tasks.submit(inputs, caller, callback_url=request.callback_url, metadata=service)
            if self.contract == 'backend':
                # The backend acknowledges after preparation; the gateway acknowledges admission.
                while True:
                    record = await self.tasks.read(identity, caller, 'deploy')
                    progress = record['_progress']
                    if record['state'] == 'succeeded' or progress:
                        return submit_response(0, 'ok')
                    if record['state'] in ('failed', 'cancelled', 'expired'):
                        return submit_response(1, task_response(record)['reason'])
                    await asyncio.sleep(.05)
            return dict(code=0, message='accepted', task_id=request.task_id, status='dispatching', variant=variant)
        except ServerError as error:
            if error.code in ('service_busy', 'queue_full', 'server_full'):
                return submit_response(2, 'A task is running.')
            if error.code == 'duplicate_task':
                raise HTTPException(409, str(error)) from error
            if self.contract == 'gateway':
                raise
            reason = str(error)
        except (ValueError, TypeError) as error:
            reason = 'Invalid request body: ' + str(error)
            if self.contract == 'gateway':
                raise HTTPException(400, reason) from error
        await self.failed_submission(body, reason, caller)
        return submit_response(1, reason)

    def backend_status(self):
        variants = self.tasks.status()['variants']
        return {variant: variants.get(variant, False) for variant in ('normal', 'dmd')}

    async def counts(self):
        counts = dict.fromkeys(('queued', 'dispatching', 'running', 'cancelling', 'completed', 'failed', 'cancelled', 'expired'), 0)
        for state, cancelled, count in await self.tasks.counts():
            status = task_response({'state': state, 'cancel_requested': cancelled, 'task_id': '', 'created_at': 0, 'service': {'contract': self.contract}})['status']
            counts[status] += count
        return counts

    def response(self, record):
        result = task_response(record)
        return VideoResponse(**result) if self.contract == 'backend' else result

    def progress(self, record):
        result = task_response(record)
        keys = ('id', 'status', 'progress', 'file_path', 'url', 'error', 'reason',
                'callback_status', 'callback_error', 'callback_attempts')
        if self.contract == 'gateway':
            keys += ('variant', 'queue_position')
        return {key: result[key] for key in keys}

    def check_content(self, record):
        if self.contract == 'backend' and (record.get('outputs') or {}).get('url'):
            raise HTTPException(400, 'Video is stored remotely. Please use the video URL.')

    async def health(self):
        backends = self.backend_status()
        return {'status': 'ok' if all(backends.values()) else 'degraded_normal_only' if backends['normal'] else 'unavailable',
                'backends': backends, 'queue': await self.counts()}

    def router(self):
        router = APIRouter(tags=['VideoEdit'])

        schema = VideoRepairRequest.model_json_schema()
        definitions = schema.pop('$defs', {})
        def inline(value):
            if isinstance(value, dict):
                if '$ref' in value:
                    return inline(definitions[value['$ref'].split('/')[-1]])
                return {key: inline(item) for key, item in value.items()}
            if isinstance(value, list):
                return [inline(item) for item in value]
            return value
        body_schema = {'anyOf': [inline(schema), {'type': 'object', 'required': ['inputs'],
            'properties': {'inputs': {'type': 'object'}, 'parameters': {'type': 'object'}}}]}

        @router.post('/v1/videos/repairs', openapi_extra={'requestBody': {
            'required': True, 'content': {'application/json': {'schema': body_schema}}}})
        async def submit(request: Request, caller=Depends(limited_identity)):
            try:
                body = await request.json()
            except ValueError:
                if self.contract == 'gateway':
                    raise HTTPException(400, 'Invalid JSON body') from None
                return submit_response(1, 'Invalid request body: Invalid JSON body')
            return await self.submit(body, caller)

        @router.get('/admin/queue')
        async def queue(status: str | None = None, limit: int = Query(100, ge=1, le=1000), caller=Depends(limited_identity)):
            if (self.server.authentication.enabled or self.server.authentication.admin_digest) and not caller.administrator:
                raise HTTPException(403, 'Management permission required')
            records = await self.tasks.list(caller, limit=None if status else limit)
            values = [task_response(record) for record in records]
            if status:
                values = [value for value in values if value['status'] == status][:limit]
            return {'counts': await self.counts(), 'tasks': values}
        return router

"""HTTP access to the same Server used by Python callers."""

import asyncio
import secrets
import uuid
from typing import Any

from fastapi import Depends, Header, HTTPException, UploadFile, File, Request, Query, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from ..operations import read_task, cancel_task, submit_task
from ..errors import ServerError
from ..protocol import DeployRequest, TaskResponse, BatchRequest, TaskInput
from ..dependencies import Identity, limited_identity
from ..streaming import accepts_events


class BodyLimit:
    def __init__(self, app, max_bytes=17 * 1024 * 1024, file_bytes=17 * 1024 * 1024):
        self.app, self.max_bytes, self.file_bytes = app, max_bytes, file_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        size = 0
        limit = self.file_bytes if scope['path'] in ('/v1/files', '/api/files/upload') else self.max_bytes
        async def bounded():
            nonlocal size
            message = await receive()
            size += len(message.get('body', b''))
            if size > limit:
                raise HTTPException(413, 'Request body exceeds the size limit')
            return message
        await self.app(scope, bounded, send)


async def save_upload(server, request, file, owner):
    """Keep the multipart stream alive until storage writing actually ends."""
    writing = server.own(asyncio.to_thread(server.storage.save, file.file,
        file.content_type or 'application/octet-stream', owner=owner))
    try:
        record = await asyncio.shield(writing)
        request.state.audit_resource = record['file_id']
        return record
    except asyncio.CancelledError:
        try:
            record = await writing
            await asyncio.shield(server.own(asyncio.to_thread(server.storage.remove, record['file_id'])))
        finally:
            raise


class FileContentResponse(Response):
    """Keep the file referenced until HTTP reading (including ranges) ends."""
    def __init__(self, server, identity, *, inline=False):
        super().__init__()
        self.server, self.identity = server, identity
        self.inline = inline

    async def __call__(self, scope, receive, send):
        server, identity, holder = self.server, self.identity, uuid.uuid4().hex
        async def prepare():
            try:
                await asyncio.to_thread(server.db.hold_files, holder, {'file_id': identity})
            except ServerError as exc:
                if exc.code == 'invalid_file':
                    raise ServerError(404, 'file_not_found', 'File is no longer available') from exc
                raise
            return await asyncio.to_thread(server.storage.resolve, identity)
        preparing = server.own(prepare())
        async def release():
            try:
                await preparing
            except BaseException:
                pass
            await asyncio.to_thread(server.db.release_files, holder)
        try:
            path, record = await asyncio.shield(preparing)
            filename = identity + '.zip' if record['media_type'] == 'application/zip' else None if self.inline else identity
            response = FileResponse(path, media_type=record['media_type'], filename=filename)
            # A delegated path transfer can outlive the ASGI response. Read the
            # content here so the database hold covers the actual transfer.
            scope = dict(scope, extensions={key: value for key, value in scope.get('extensions', {}).items()
                                           if key != 'http.response.pathsend'})
            await response(scope, receive, send)
        finally:
            await asyncio.shield(server.own(release()))


async def deploy_request(server, name, body, caller, *, request, idempotency_key=None, accept='*/*'):
    """Common HTTP admission and response handling for standard and algorithm routes."""
    caller.require('deploy', name)
    wait_timeout = server.wait_timeout(name, body.wait_timeout_seconds) if body.mode == 'sync' else None
    if body.mode == 'sync':
        from ..request import reserve_sync_connection, wait_sync_response
        reserve_sync_connection(request, caller)
        async def result():
            identity, future = await server.submit(name, body.inputs, body.parameters,
                                                   idempotency_key=idempotency_key, owner=caller.subject, callback_url=body.callback_url)
            request.state.audit_resource = identity
            return await server.wait_result(identity, future, wait_timeout)
        return await wait_sync_response(request, result())
    if body.mode == 'async':
        value = await submit_task(server, name, body.inputs, body.parameters, caller,
                                  idempotency_key=idempotency_key, callback_url=body.callback_url)
        request.state.audit_resource = value['task_id']
        return JSONResponse(value, status_code=202)
    if body.mode == 'stream':
        caller.require('tasks:read', name)
        if not server._entry(name)['config'].streaming.enabled:
            raise ServerError(422, 'stream_unsupported', 'This entry does not declare streaming support')
        if not accepts_events(accept):
            raise ServerError(406, 'invalid_accept', 'Streaming requires Accept: text/event-stream')
        from ..streaming import reserve_connection, TaskEventResponse
        reserve_connection(server)
    try:
        identity, future = await server.submit(name, body.inputs, body.parameters, idempotency_key=idempotency_key, owner=caller.subject, callback_url=body.callback_url, observe=False)
        request.state.audit_resource = identity
    except BaseException:
        if body.mode == 'stream':
            server.stream_connections -= 1
        raise
    # Event readers use durable task records; no result observer was created.
    if body.mode == 'stream':
        return TaskEventResponse(server, identity, reserved=True)


def register_routes(server):
    app = server.app
    from .stream import register_routes as register_stream_routes
    register_stream_routes(server)
    app.add_middleware(BodyLimit, file_bytes=server.storage.max_bytes + 1024**2)

    @app.exception_handler(ServerError)
    async def error(request, exc):
        headers = {'Location': exc.status_url} if exc.status_url is not None else {}
        if exc.retry_after is not None:
            headers['Retry-After'] = str(exc.retry_after)
        return JSONResponse(exc.response(), status_code=exc.status_code,
                            headers=headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Raw validation inputs/contexts can contain NaN, exception objects or
        # entire request bodies. Return the JSON-safe field diagnostics only.
        return JSONResponse({'detail': [{key: item[key] for key in ('type', 'loc', 'msg')}
                                       for item in exc.errors()]}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        identity = getattr(request.state, 'request_id', None)
        return JSONResponse({'error': {'code': 'internal_error', 'message': 'Internal Server error', 'retryable': False}},
                            status_code=500, headers={'X-Request-ID': identity} if identity else {})

    async def administrator(request: Request, authorization: str = Header(default='')):
        if server.editor_access is not None:
            server.editor_access.require(request, management=True)
            await server.audit.identify(request.scope, request.state.audit_identity)
            return
        if not server.admin_token:
            raise ServerError(503, 'management_disabled', 'Configure an admin token to enable management')
        if not secrets.compare_digest(authorization.encode(), ('Bearer ' + server.admin_token).encode()):
            raise ServerError(401, 'unauthorized', 'Management token required')
        await server.audit.identify(request.scope, Identity('@administrator', frozenset(), frozenset(), True))

    if hasattr(server, 'workflow_dir'):
        @app.post('/api/workflow/server/parameters', dependencies=[Depends(administrator)])
        async def workflow_parameters(body: dict[str, Any]):
            from ..config import suggest_config, load_json
            from ..parameters import parameter_paths
            data = body.get('businessContent')
            if not isinstance(data, dict):
                raise ServerError(422, 'invalid_request', 'businessContent must be a workflow object')
            saved = {}
            path = server.db.get_workflow_path(body.get('id')) if body.get('id') else None
            if path and path.with_suffix('.server.json').is_file():
                saved = load_json(path.with_suffix('.server.json')).get('parameters', {})
            fields = suggest_config(data)['parameters']
            return {name: {'default': field['default'], 'exposed':
                    saved[name].get('exposed', list(parameter_paths(field['default']))) if name in saved
                    else [p for p in parameter_paths(field['default']) if p in (['path_'], ['str_'])
                          and name in suggest_config(data)['inputs']]}
                    for name, field in fields.items()}

        @app.post('/api/workflow/server/start', dependencies=[Depends(administrator)])
        async def start_workflow(request: Request, body: dict[str, Any],
                                 idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
                                 wait_timeout_seconds: float = Query(default=5, ge=0, le=60)):
            target = 'workflow-' + str(body.get('id') or 'new')
            return await control(request, target, 'workflow', body, idempotency_key,
                                 lambda: server.start_workflow(body), wait_timeout_seconds)

    @app.get('/health/live')
    async def live():
        return {'status': 'alive'}

    @app.get('/metrics', include_in_schema=False)
    async def metrics(authorization: str = Header(default='')):
        # Cookie sessions are intentionally not accepted on the scrape endpoint.
        if not server.admin_token:
            raise ServerError(503, 'management_disabled', 'Configure an admin token to enable metrics')
        if not authorization:
            raise ServerError(401, 'unauthorized', 'Management token required')
        if not server.authentication.authenticate(authorization).administrator:
            raise ServerError(403, 'forbidden', 'Management token required')
        from prometheus_client import CONTENT_TYPE_LATEST
        snapshot = server.metrics.snapshot(server)
        content = await asyncio.to_thread(server.metrics.render, snapshot)
        return Response(content, headers={'Content-Type': CONTENT_TYPE_LATEST, 'Cache-Control': 'no-store'})

    @app.get('/health/ready')
    async def ready():
        healthy = server.started and not server.shutting_down and (
            not server.entries or any(server.accepting(name, entry) for name, entry in server.entries.items()))
        editor = getattr(server, '_editor_runtime', None)
        if editor is not None:
            healthy = healthy and editor.ready
        if healthy:
            try:
                healthy = await asyncio.wait_for(asyncio.to_thread(server.db.ping), timeout=2)
                if server.redis_config is not None:
                    async def shared_ready():
                        for name, entry in list(server.entries.items()):
                            queue = entry.get('queue')
                            if queue is None or not server.accepting(name, entry):
                                continue
                            active = await server.shared_runtime.status(entry)
                            if active['active_queue'] == queue.key and active['accepting'] and await queue.client.ping():
                                return True
                        return False
                    healthy = healthy and await asyncio.wait_for(shared_ready(), timeout=2)
            except Exception:
                healthy = False
        return JSONResponse({'ready': healthy}, status_code=200 if healthy else 503)

    @app.get('/v1/models')
    async def models(caller: Identity = Depends(limited_identity)):
        return {'models': [server.describe(name) for name in server.entries if caller.allows('deploy', name)]}

    @app.get('/v1/server/{name}/schema')
    async def schema(name: str, caller: Identity = Depends(limited_identity)):
        caller.require('deploy', name)
        return server.describe(name)

    @app.post('/v1/server/{name}/deploy')
    async def deploy(name: str, body: dict[str, Any], request: Request, idempotency_key: str | None = Header(default=None),
                     caller: Identity = Depends(limited_identity), accept: str = Header(default='*/*')):
        from ..protocol import node_request
        from pydantic import ValidationError
        caller.require('deploy', name)
        try:
            entry = server._entry(name)
            body = DeployRequest.model_validate(node_request(entry['config'], body, untyped=entry['untyped']))
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        return await deploy_request(server, name, body, caller, request=request, idempotency_key=idempotency_key, accept=accept)

    @app.post('/v1/tasks/{task_id}/inputs')
    async def task_input(task_id: str, body: TaskInput, caller: Identity = Depends(limited_identity)):
        record = await asyncio.to_thread(server.db.get_task, task_id)
        if record is None or not caller.owns(record):
            raise ServerError(404, 'task_not_found', 'Task does not exist')
        caller.require('deploy', record['name'])
        return await server.send_input(task_id, body.data, sequence=body.sequence, end=body.end, owner=record['owner'])

    @app.post('/v1/server/{name}/deploy/batch')
    async def batch(name: str, body: dict[str, Any], request: Request, caller: Identity = Depends(limited_identity),
                    idempotency_key: str | None = Header(default=None)):
        caller.require('deploy', name)
        from ..protocol import node_request
        from pydantic import ValidationError
        entry = server._entry(name)
        if isinstance(body.get('items'), list):
            body = dict(body, items=[node_request(entry['config'], item, untyped=entry['untyped'])
                                    if isinstance(item, dict) else item for item in body['items']])
        try:
            body = BatchRequest.model_validate(body)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        if idempotency_key is not None:
            raise ServerError(422, 'batch_idempotency', 'Provide idempotency_key in each batch item')
        wait_timeout = server.wait_timeout(name, body.wait_timeout_seconds) if body.mode == 'sync' else None
        if body.mode == 'sync':
            from ..request import reserve_sync_connection, wait_sync_response
            reserve_sync_connection(request, caller)
        operation = server.deploy_batch(name, body.items, mode=body.mode, owner=caller.subject, wait_timeout_seconds=wait_timeout)
        result = await wait_sync_response(request, operation) if body.mode == 'sync' else await operation
        status = 202 if body.mode == 'async' and any(item['accepted'] for item in result['items']) else 200
        return JSONResponse(result, status_code=status)

    @app.get('/v1/tasks/{identity}', response_model=TaskResponse)
    async def task(identity: str, caller: Identity = Depends(limited_identity)):
        return await read_task(server, identity, caller)

    @app.get('/v1/tasks/{identity}/events')
    async def events(identity: str, last_event_id: str = Header(default='0'),
                     caller: Identity = Depends(limited_identity), accept: str = Header(default='*/*')):
        caller.require('tasks:read')
        value = await asyncio.to_thread(server.db.get_task, identity)
        if value is None or not caller.owns(value):
            raise ServerError(404, 'task_not_found', 'Unknown task')
        caller.require('tasks:read', value['name'])
        if not accepts_events(accept):
            raise ServerError(406, 'invalid_accept', 'Events require Accept: text/event-stream')
        if not last_event_id.isascii() or not last_event_id.isdecimal() or len(last_event_id) > 10:
            raise ServerError(422, 'invalid_event_id', 'Last-Event-ID must be a nonnegative event sequence')
        after = int(last_event_id)
        if after > value['event_seq']:
            raise ServerError(409, 'invalid_event_id', 'Last-Event-ID exceeds the task event sequence')
        from ..streaming import TaskEventResponse
        return TaskEventResponse(server, identity, after)

    @app.post('/v1/tasks/{identity}/cancel', response_model=TaskResponse)
    async def cancel(identity: str, caller: Identity = Depends(limited_identity)):
        return await cancel_task(server, identity, caller)

    @app.post('/v1/files', status_code=201)
    async def upload(request: Request, file: UploadFile = File(...), caller: Identity = Depends(limited_identity)):
        caller.require('files:write')
        return await save_upload(server, request, file, caller.subject)

    async def authorize_file(identity, caller, permission):
        caller.require(permission)
        record = await asyncio.to_thread(server.db.get_file, identity)
        if record is None or not caller.owns(record):
            raise ServerError(404, 'file_not_found', 'Unknown file')

    @app.get('/v1/files/{identity}')
    async def metadata(identity: str, caller: Identity = Depends(limited_identity)):
        await authorize_file(identity, caller, 'files:read')
        return await asyncio.to_thread(server.storage.metadata, identity)

    @app.get('/v1/files/{identity}/content')
    async def content(identity: str, caller: Identity = Depends(limited_identity)):
        await authorize_file(identity, caller, 'files:read')
        return FileContentResponse(server, identity)

    @app.delete('/v1/files/{identity}')
    async def remove_file(identity: str, caller: Identity = Depends(limited_identity)):
        if not server.authentication.enabled and not caller.administrator:
            raise ServerError(401, 'unauthorized', 'Management token required')
        await authorize_file(identity, caller, 'files:write')
        await asyncio.shield(server.own(asyncio.to_thread(server.storage.remove, identity)))
        return {'file_id': identity, 'deleted': True}

    @app.post('/v1/files/{identity}/download-url')
    async def download_url(identity: str, caller: Identity = Depends(limited_identity)):
        if not server.authentication.enabled and not caller.administrator:
            raise ServerError(401, 'unauthorized', 'Management token required')
        await authorize_file(identity, caller, 'files:read')
        return {'url': await asyncio.to_thread(server.storage.signed_get, identity, 300), 'expires_in': 300}

    async def control(request, name, action, body, key, operation, wait):
        # The existing management token/session represents one management
        # identity; caller API keys do not gain access to control receipts.
        return await server.controls.request(request.state.request_id, 'management', name, action, body, key, operation, wait)

    @app.get('/api/server/audit/records', dependencies=[Depends(administrator)])
    async def audit_records(limit: int = Query(default=100, ge=1, le=1000),
                            before: int | None = Query(default=None, ge=1, le=2**63 - 1),
                            request_id: str | None = Query(default=None, pattern=r'^req_[0-9a-f]{32}$')):
        return JSONResponse(await server.audit.read(limit, before, request_id), headers={'Cache-Control': 'no-store'})

    @app.get('/api/server', dependencies=[Depends(administrator)])
    async def entries(request_id: str | None = Query(default=None, pattern=r'^req_[0-9a-f]{32}$'), include_resources: bool = False):
        if request_id is not None:
            record = await server.controls.lookup(request_id, 'management')
            return server.controls.response(record)
        result = {'entries': [server.describe(name) for name in server.entries]}
        if include_resources:
            result['devices'] = await asyncio.to_thread(server.devices.snapshot) if server.devices is not None else {}
        return result

    @app.post('/api/server/cleanup-tasks', dependencies=[Depends(administrator)])
    async def cleanup_tasks():
        return {'deleted_tasks': await server.cleanup_tasks()}

    @app.post('/api/server/cleanup-files', dependencies=[Depends(administrator)])
    async def cleanup_files():
        return await server.cleanup_files()

    @app.get('/api/server/mcp/config', dependencies=[Depends(administrator)])
    async def mcp_config():
        return {'enabled': server.mcp is not None, 'path': server.mcp.config.path if server.mcp else None}

    @app.get('/api/server/{name}', dependencies=[Depends(administrator)])
    async def entry(name: str):
        return server.describe(name)

    @app.post('/api/server/{name}/stop', dependencies=[Depends(administrator)])
    async def stop(name: str, request: Request,
                   idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
                   wait_timeout_seconds: float = Query(default=5, ge=0, le=60)):
        return await control(request, name, 'stop', {}, idempotency_key, lambda: server.stop_entry(name), wait_timeout_seconds)

    @app.post('/api/server/{name}/start', dependencies=[Depends(administrator)])
    async def start(name: str, request: Request,
                    idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
                    wait_timeout_seconds: float = Query(default=5, ge=0, le=60)):
        return await control(request, name, 'start', {}, idempotency_key, lambda: server.start_entry(name), wait_timeout_seconds)

    @app.post('/api/server/{name}/reload', dependencies=[Depends(administrator)])
    async def reload(name: str, request: Request, body: dict[str, Any] | None = Body(default=None),
                     idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
                     wait_timeout_seconds: float = Query(default=5, ge=0, le=60)):
        value = body or {}
        if set(value) - {'workflow', 'config'} or any(item is not None and not isinstance(item, dict) for item in value.values()):
            raise ServerError(422, 'invalid_request', 'Reload accepts workflow and config JSON objects')
        return await control(request, name, 'reload', value, idempotency_key,
                             lambda: server.reload_entry(name, **value), wait_timeout_seconds)

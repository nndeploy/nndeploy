"""One video resource router shared by algorithm-specific submission adapters."""
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..dependencies import limited_identity
from .extensions import prepare_router, mount_router


class VideoResources:
    def __init__(self, server):
        self.server = server
        self.adapters = {}
        self.mounted = prepare_router(server, 'videos', self.router(), prefix='', authorize_entry=False)

    def register(self, adapter):
        if self.server.started or self.server.starting_task is not None:
            raise ValueError('Register video resources before starting the Server')
        if adapter.name not in self.server.entries:
            raise ValueError('Register the Server entry before its video adapter')
        if adapter.name in self.adapters:
            raise ValueError(f'Duplicate video entry: {adapter.name}')
        if not self.adapters:
            mount_router(self.server, self.mounted)
            self.server.app.state.video_resources = self
        self.adapters[adapter.name] = adapter

    def names(self, caller, permission, entry=None):
        if entry is not None:
            if entry not in self.adapters:
                raise HTTPException(404, 'Video entry not found')
            caller.require(permission, entry)
            return [entry]
        return [name for name in self.adapters if caller.allows(permission, name)]

    @property
    def scope(self):
        return self.server.redis_config.namespace if self.server.redis_config else self.server.config_scope

    async def read(self, identity, caller, permission, entry, request):
        names = self.names(caller, permission, entry)
        record = await asyncio.to_thread(self.server.db.resource_task, names, identity, self.scope,
                                        owner=caller.subject, administrator=caller.administrator)
        if record is None:
            raise HTTPException(404, 'Video not found')
        request.state.server_entry = (record['name'], caller)
        return record, self.adapters[record['name']]

    def response(self, record, adapter, caller):
        value = adapter.response(record)
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if len(self.adapters) > 1 or caller.administrator:
            value = {**value, 'entry': record['name'], 'task_id': record['task_id']}
        return value

    def router(self):
        router = APIRouter(tags=['Videos'])

        @router.get('/v1/videos')
        async def listing(request: Request, after: str | None = None, limit: int | None = Query(None, ge=1, le=100),
                          order: str = 'desc', entry: str | None = None, caller=Depends(limited_identity)):
            names = self.names(caller, 'tasks:read', entry)
            if len(names) == 1:
                request.state.server_entry = (names[0], caller)
            records = await asyncio.to_thread(self.server.db.service_tasks, names, self.scope,
                owner=caller.subject, administrator=caller.administrator, after=after,
                limit=limit, order=order.lower())
            return {'data': [self.response(record, self.adapters[record['name']], caller) for record in records], 'object': 'list'}

        @router.get('/v1/videos/{identity}')
        async def retrieve(identity: str, request: Request, entry: str | None = None, caller=Depends(limited_identity)):
            record, adapter = await self.read(identity, caller, 'tasks:read', entry, request)
            return self.response(record, adapter, caller)

        @router.get('/v1/videos/{identity}/progress')
        async def progress(identity: str, request: Request, entry: str | None = None, caller=Depends(limited_identity)):
            record, adapter = await self.read(identity, caller, 'tasks:read', entry, request)
            return adapter.progress(record)

        @router.delete('/v1/videos/{identity}')
        async def cancel(identity: str, request: Request, entry: str | None = None, caller=Depends(limited_identity)):
            record, adapter = await self.read(identity, caller, 'tasks:cancel', entry, request)
            await self.server.cancel_task(record['task_id'])
            record, adapter = await self.read(record['task_id'], caller, 'tasks:cancel', record['name'], request)
            return self.response(record, adapter, caller)

        @router.get('/v1/videos/{identity}/content')
        async def content(identity: str, request: Request, entry: str | None = None, variant: str | None = None,
                          caller=Depends(limited_identity)):
            from .deploy import FileContentResponse
            record, adapter = await self.read(identity, caller, 'files:read', entry, request)
            adapter.check_content(record)
            artifacts = (record.get('outputs') or {}).get('artifacts') or []
            if record['state'] != 'succeeded' or not artifacts:
                raise HTTPException(404, 'Video content not found')
            return FileContentResponse(self.server, artifacts[0]['file']['file_id'], inline=True)

        @router.get('/health')
        async def health(caller=Depends(limited_identity)):
            names = self.names(caller, 'tasks:read')
            statuses = {name: await self.adapters[name].health() for name in names}
            if len(self.adapters) == 1 and statuses:
                return next(iter(statuses.values()))
            return {'status': 'ok' if statuses and all(value['status'] == 'ok' for value in statuses.values())
                    else 'degraded', 'services': statuses}

        return router


def prepare_video_resources(server):
    """Preflight common routes before adding a model; commit after Server.add."""
    return getattr(server.app.state, 'video_resources', None) or VideoResources(server)

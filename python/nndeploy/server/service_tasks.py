"""Business task access using the Server's ownership, persistence and admission."""
import asyncio
import time
import uuid


class ServiceTasks:
    def __init__(self, server, name, *, memory_estimator=None):
        if memory_estimator is not None and not callable(memory_estimator):
            raise TypeError('memory_estimator must be callable')
        self.server, self.name = server, name
        self.memory_estimator = memory_estimator

    @property
    def scope(self):
        return self.server.redis_config.namespace if self.server.redis_config else self.server.config_scope

    async def submit(self, inputs, caller, *, metadata, callback_url=None):
        caller.require('deploy', self.name)
        if self.memory_estimator is not None:
            # The registration supplies a trusted, calibrated estimator. Never
            # take an admission budget directly from an HTTP request field.
            from copy import deepcopy
            estimate = await asyncio.to_thread(self.memory_estimator, deepcopy(inputs['request']))
            metadata = {**metadata, 'temporary_memory_bytes': estimate}
        identity, _ = await self.server.submit(self.name, inputs, owner=caller.subject,
            service=metadata, callback_url=callback_url, observe=False)
        return identity

    async def read(self, identity, caller, permission='tasks:read'):
        caller.require(permission, self.name)
        return await asyncio.to_thread(self.server.db.resource_task, [self.name], identity, self.scope,
                                      owner=caller.subject, administrator=caller.administrator)

    async def list(self, caller, **options):
        caller.require('tasks:read', self.name)
        return await asyncio.to_thread(self.server.db.service_tasks, self.name, self.scope,
            owner=caller.subject, administrator=caller.administrator, **options)

    async def busy(self, group):
        return await asyncio.to_thread(self.server.db.service_busy, self.scope, group)

    async def counts(self):
        return await asyncio.to_thread(self.server.db.service_counts, self.name, self.scope)

    def status(self):
        return self.server.service_status(self.name)

    async def record_failure(self, caller, metadata, reason, callback_url=None):
        """Persist a legacy protocol's rejected submission and its notification."""
        from .callbacks import validate_url
        caller.require('deploy', self.name)
        callback_url = validate_url(callback_url)
        identity = 'task_' + uuid.uuid4().hex
        def persist():
            self.server.db.create_task(identity, self.name, time.time(), {}, {}, owner=caller.subject,
                callback_url=callback_url, callback_scope=self.server.config_scope,
                service={**metadata, 'scope': self.scope})
            self.server.db.finish_task(identity, 'failed', time.time(), error={'message': reason})
        await asyncio.shield(self.server.own(asyncio.to_thread(persist)))
        return identity

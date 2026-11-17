"""Caller-aware task operations shared by HTTP and MCP transports."""
import asyncio
from .errors import ServerError


async def read_task(server, identity, caller, permission='tasks:read'):
    caller.require(permission)
    record = await asyncio.to_thread(server.db.get_task, identity)
    if record is None or not caller.owns(record):
        raise ServerError(404, 'task_not_found', 'Unknown task')
    caller.require(permission, record['name'])
    return record


async def cancel_task(server, identity, caller):
    await read_task(server, identity, caller, 'tasks:cancel')
    return await server.cancel_task(identity)


async def submit_task(server, name, inputs, parameters, caller, *, idempotency_key=None, callback_url=None):
    caller.require('deploy', name)
    identity, future = await server.submit(name, inputs, parameters, owner=caller.subject,
                                          idempotency_key=idempotency_key, callback_url=callback_url)
    future.cancel()  # Detach this observer; the durable accepted task continues.
    return {'task_id': identity, 'status_url': f'/v1/tasks/{identity}',
            'dispatch_pending': getattr(future, 'dispatch_pending', False)}

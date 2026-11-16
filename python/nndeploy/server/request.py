"""Transport correlation and metadata-only structured Server events."""
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from starlette.requests import ClientDisconnect

from .logging_taskid import get_request_id, set_request_id, reset_request_id

LOGGER = logging.getLogger('nndeploy.server.events')
SYNC_SEND_SECONDS = 5


def reserve_sync_connection(request, caller):
    """One slot per HTTP response, including algorithm-specific response conversion."""
    from .errors import ServerError
    state, server = request.scope['state'], request.app.state.server
    if state.get('sync_connection_closed'):
        raise ServerError(409, 'request_closed', 'The HTTP request has already ended')
    if 'sync_connection_subject' in state:
        return
    count = server.sync_connections_by_subject.get(caller.subject, 0)
    maximum = server.authentication.max_sync_connections_per_subject
    if server.sync_connections >= server.max_sync_connections:
        raise ServerError(429, 'sync_connections_full', 'Server synchronous HTTP connection limit reached', retryable=True)
    if maximum is not None and count >= maximum:
        raise ServerError(429, 'sync_connections_full', 'Caller synchronous HTTP connection limit reached', retryable=True)
    # Called on the HTTP loop without awaiting: check and reserve are atomic.
    server.sync_connections += 1
    server.sync_connections_by_subject[caller.subject] = count + 1
    state['sync_connection_subject'] = caller.subject


async def wait_sync_response(request, operation):
    """Monitor disconnect after the route has consumed/translated its input.

    Cancelling this HTTP waiter must never cancel Server-owned task admission
    or execution. Server.submit/deploy_batch own those operations independently.
    """
    async def disconnected():
        while True:
            if (await request.receive())['type'] == 'http.disconnect':
                return
    work = asyncio.create_task(operation)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((work, watcher), return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            watcher.result()
            raise ClientDisconnect
        return work.result()
    finally:
        for task in (work, watcher):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, watcher, return_exceptions=True)


def event(kind, *, request_id=None, task_id=None, entry=None, worker_pid=None,
          state=None, method=None, route=None, status=None, duration_seconds=None):
    # Deliberately no arbitrary extras, request bodies, headers, SQL or exception text.
    value = dict(event=kind, timestamp=time.time(), pid=os.getpid())
    fields = dict(request_id=request_id, task_id=task_id, entry=entry, worker_pid=worker_pid,
                  state=state, method=method, route=route, status=status, duration_seconds=duration_seconds)
    value.update({key: item for key, item in fields.items() if item is not None})
    LOGGER.info(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')),
                extra={'server_event': value})


def http_labels(scope, server):
    route = getattr(scope.get('route'), 'path', '__unmatched__')
    method = scope['method'] if scope['method'] in {'GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'} else 'OTHER'
    name = scope.get('path_params', {}).get('name')
    if name is None and scope.get('state', {}).get('server_entry'):
        name = scope['state']['server_entry'][0]
    return method, route, name if name in server.entries else '__unknown__'


class RequestMiddleware:
    def __init__(self, app, server):
        self.app, self.server = app, server

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'websocket':
            return await self.websocket(scope, receive, send)
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        identity = 'req_' + uuid.uuid4().hex
        scope.setdefault('state', {})['request_id'] = identity
        token = set_request_id(identity)
        started, status, complete = time.perf_counter(), None, False
        tracking = None

        async def correlated(message):
            nonlocal status, complete
            if message['type'] == 'http.response.start':
                message = dict(message, headers=[(key, value) for key, value in message.get('headers', [])
                                                if key.lower() != b'x-request-id'] + [(b'x-request-id', identity.encode())])
                status = message['status']
            if 'sync_connection_subject' in scope['state']:
                try:
                    await asyncio.wait_for(send(message), timeout=SYNC_SEND_SECONDS)
                except (asyncio.TimeoutError, OSError) as exc:
                    raise ClientDisconnect from exc
            else:
                await send(message)
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                complete = True
                if tracking is not None:
                    tracking['active'] = False

        try:
            audit = getattr(self.server, 'audit', None)
            if audit is not None and audit.covers(scope):
                from .errors import ServerError
                from starlette.responses import JSONResponse
                try:
                    await audit.begin(scope)
                except ServerError as exc:
                    await JSONResponse(exc.response(), status_code=exc.status_code)(scope, receive, correlated)
                    return
            if audit is None or not scope['state'].get('audit_started'):
                await self.app(scope, receive, correlated)
            else:
                with audit.track() as tracking:
                    await self.app(scope, receive, correlated)
        except asyncio.CancelledError:
            status = status or 499
            raise
        except ClientDisconnect:
            # A slow/broken transport ends the reader, not its accepted task.
            if 'sync_connection_subject' not in scope['state']:
                raise
            status = status or 499
        except Exception:
            status = status or 500
            raise
        finally:
            state = scope['state']
            state['sync_connection_closed'] = True
            if 'sync_connection_subject' in state:
                subject = state.pop('sync_connection_subject')
                self.server.sync_connections -= 1
                remaining = self.server.sync_connections_by_subject[subject] - 1
                if remaining:
                    self.server.sync_connections_by_subject[subject] = remaining
                else:
                    del self.server.sync_connections_by_subject[subject]
            method, route, entry = http_labels(scope, self.server)
            try:
                event('http_request', request_id=identity, method=method, route=route, entry=entry,
                      status=status, state='complete' if complete else 'interrupted',
                      duration_seconds=time.perf_counter() - started)
                if audit is not None:
                    await audit.finish(scope, status, complete, time.perf_counter() - started)
            finally:
                reset_request_id(token)

    async def websocket(self, scope, receive, send):
        from starlette.websockets import WebSocketDisconnect
        from .errors import ServerError

        identity = 'req_' + uuid.uuid4().hex
        state = scope.setdefault('state', {})
        state['request_id'] = identity
        token = set_request_id(identity)
        audit = getattr(self.server, 'audit', None)
        started, status, complete, ended = time.perf_counter(), None, False, False
        tracking = None

        async def observed_receive():
            nonlocal status, complete, ended
            message = await receive()
            if message['type'] == 'websocket.disconnect' and not ended:
                status = message.get('code', 1005)
                complete, ended = status in (1000, 1001), True
                if tracking is not None:
                    tracking['active'] = False
            return message

        async def observed_send(message):
            nonlocal status, complete, ended
            kind = message['type']
            if kind == 'websocket.close' and ended:
                return  # Cleanup must not replace an observed peer disconnect.
            if kind == 'websocket.accept':
                caller = state.get('audit_identity')
                if audit is not None and caller is not None:
                    try:
                        await audit.identify(scope, caller)
                    except ServerError:
                        await observed_send({'type': 'websocket.close', 'code': 1013})
                        raise WebSocketDisconnect(code=1013) from None
                message = dict(message, headers=[(key, value) for key, value in message.get('headers', [])
                    if key.lower() != b'x-request-id'] + [(b'x-request-id', identity.encode())])
            await send(message)
            if kind == 'websocket.accept':
                status = 101
            elif kind == 'websocket.close':
                status, complete, ended = message.get('code', 1000), True, True
            elif kind == 'websocket.http.response.start':
                status = message['status']
            elif kind == 'websocket.http.response.body' and not message.get('more_body', False):
                complete, ended = True, True
            if ended and tracking is not None:
                tracking['active'] = False

        try:
            if audit is not None and audit.covers(scope):
                try:
                    await audit.begin(scope)
                except ServerError:
                    await observed_send({'type': 'websocket.close', 'code': 1013})
                    return
            if audit is None or not state.get('audit_started'):
                await self.app(scope, observed_receive, observed_send)
            else:
                with audit.track() as tracking:
                    await self.app(scope, observed_receive, observed_send)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not ended:
                status, complete = 1011, False
            raise
        finally:
            try:
                duration = time.perf_counter() - started
                route = getattr(scope.get('route'), 'path', state.get('audit_route', '__unmatched__'))
                event('websocket_connection', request_id=identity, method='WEBSOCKET', route=route,
                      status=status, state='complete' if complete else 'interrupted', duration_seconds=duration)
                if audit is not None:
                    # Client disconnect may cancel the ASGI scope. Retain the
                    # final write and join it before Server disposes its DB.
                    await asyncio.shield(self.server.own(audit.finish(scope, status, complete, duration)))
            finally:
                reset_request_id(token)


@contextmanager
def worker_log_context(record, worker):
    fields = dict(task_id=record['task_id'], request_id=record.get('request_id'), entry=record['name'], worker_pid=worker.pid)
    started, state = time.perf_counter(), 'succeeded'
    event('worker_request_started', **fields)
    try:
        yield
    except BaseException:
        state = 'failed'
        raise
    finally:
        # A process worker can be recreated during deploy after a prior failure.
        # On failure its process may already be gone; retain the last known PID.
        fields['worker_pid'] = worker.pid or fields['worker_pid']
        event('worker_request_finished', **fields, state=state, duration_seconds=time.perf_counter() - started)


from .log_policy import DiagnosticFormatter


class StructuredFormatter(DiagnosticFormatter):
    """Compatibility name for Server-owned structured/diagnostic sinks."""


def logging_config(*, editor=False):
    """Used by Server.run; embedded applications may provide their own handlers."""
    import copy
    from uvicorn.config import LOGGING_CONFIG
    config = copy.deepcopy(LOGGING_CONFIG)
    config['formatters']['server_events'] = {'()': DiagnosticFormatter, 'format': '%(message)s'}
    config['formatters']['default'] = {'()': DiagnosticFormatter, 'format': '%(levelname)s: %(message)s'}
    config['handlers']['default']['class'] = 'nndeploy.server.log_policy.QueuedDiagnosticStreamHandler'
    config['handlers']['server_events'] = {'class': 'nndeploy.server.log_policy.QueuedDiagnosticStreamHandler', 'formatter': 'server_events', 'stream': 'ext://sys.stderr'}
    config['loggers']['nndeploy.server.events'] = {'handlers': ['server_events'], 'level': 'INFO', 'propagate': False}
    if editor:
        config['handlers'].pop('server_events')
        config['loggers']['nndeploy.server.events'] = {'handlers': [], 'level': 'INFO', 'propagate': True}
    else:
        config['root'] = {'handlers': ['default'], 'level': 'INFO'}
    return config

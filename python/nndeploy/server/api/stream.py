"""Duplex transport for existing Tasks, with durable input acknowledgements."""
import asyncio
import json
import time
import uuid

from anyio import BrokenResourceError, ClosedResourceError
from fastapi import Depends, WebSocket, WebSocketDisconnect

from ..auth import origin_key
from ..db import TERMINAL
from ..dependencies import Identity, current_identity
from ..errors import ServerError
from ..protocol import TaskInput
from ..streaming import reserve_connection

CONNECTION_SECONDS = 60
RECEIVE_SECONDS = 30
SEND_SECONDS = 5


def register_routes(server):
    @server.app.websocket('/v1/tasks/{task_id}/stream')
    async def stream(ws: WebSocket, task_id: str, caller: Identity = Depends(current_identity)):
        ws.scope.setdefault('state', {})['audit_identity'] = caller
        reserved = False
        accepted = False
        sending = asyncio.Lock()
        readers = []

        async def database(operation, *args):
            return await asyncio.shield(server.own(asyncio.to_thread(operation, *args)))

        async def send(value):
            async def write():
                async with sending:
                    await ws.send_json(value)
            await asyncio.wait_for(write(), SEND_SECONDS)

        async def error(exc):
            await send(dict(event='error', terminal=False, **exc.response()))

        try:
            origin = ws.headers.get('origin')
            expected = ('https://' if ws.url.scheme == 'wss' else 'http://') + ws.url.netloc
            try:
                allowed_origin = not origin or origin_key(origin) in {
                    origin_key(expected), *(origin_key(value) for value in server.cors_origins)}
            except ValueError:
                allowed_origin = False
            if not allowed_origin:
                raise ServerError(403, 'forbidden_origin', 'WebSocket origin is not allowed')
            reserve_connection(server)
            reserved = True
            await server.request_limiter.take(caller.subject)
            record = await database(server.db.get_task, task_id)
            if record is None or not caller.owns(record):
                raise ServerError(404, 'task_not_found', 'Task does not exist')
            ws.scope['state']['audit_entry'] = record['name']
            caller.require('deploy', record['name'])
            caller.require('tasks:read', record['name'])
            state = json.loads(record['input_state']) if record['input_state'] else None
            if state is None:
                raise ServerError(422, 'duplex_unsupported', 'Task does not declare duplex input')
            try:
                after = int(ws.query_params.get('after', '0'))
            except ValueError:
                raise ServerError(422, 'invalid_sequence', 'after must be a nonnegative event sequence') from None
            if after < 0 or after > record['event_seq']:
                raise ServerError(409, 'invalid_sequence', 'Event sequence is outside the retained task history')
            await ws.accept()
            accepted = True
            await send(dict(event='ready', data=dict(task_id=task_id, input_sequence=state['received'], input_closed=state['closed'])))

            async def receive():
                while not server.shutting_down:
                    message = await asyncio.wait_for(ws.receive(), RECEIVE_SECONDS)
                    if message['type'] == 'websocket.disconnect':
                        return
                    await server.request_limiter.take(caller.subject)
                    text = message.get('text')
                    if text is None:
                        await ws.close(code=1003)
                        return
                    if len(text.encode('utf-8')) > state['max_bytes'] + 4096:
                        await ws.close(code=1009)
                        return
                    audit_scope = None
                    started = time.monotonic()
                    status, complete = 500, False
                    try:
                        # Frames mutate the same Task as HTTP input. Record
                        # metadata through the existing optional audit store.
                        if server.audit.config is not None:
                            audit_scope = dict(ws.scope, method='WS', state=dict(
                                request_id='req_' + uuid.uuid4().hex, audit_resource=task_id))
                            await server.audit.begin(audit_scope)
                            await server.audit.identify(audit_scope, caller)
                        try:
                            value = TaskInput.model_validate_json(text)
                        except ValueError:
                            raise ServerError(422, 'invalid_input', 'Expected a JSON input with sequence, data and optional end') from None
                        accepted = await server.send_input(task_id, value.data, sequence=value.sequence,
                                                           end=value.end, owner=record['owner'])
                        await send(dict(event='input_ack', data=accepted))
                        status, complete = 200, True
                    except ServerError as exc:
                        await error(exc)
                        status, complete = exc.status_code, True
                    finally:
                        if audit_scope is not None:
                            await asyncio.shield(server.own(server.audit.finish(
                                audit_scope, status, complete, time.monotonic() - started)))

            async def events():
                cursor = after
                heartbeat = time.monotonic()
                while not server.shutting_down:
                    for event in await database(server.db.read_task_events, task_id, cursor):
                        await send(dict(event=event['kind'], sequence=event['sequence'], data=event['data']))
                        cursor = event['sequence']
                        if event['kind'] == 'done':
                            return
                    latest = await database(server.db.get_task, task_id)
                    if latest is None:
                        raise ServerError(404, 'task_not_found', 'Task history is no longer available')
                    if latest['state'] in TERMINAL and cursor >= latest['event_seq']:
                        return
                    if time.monotonic() - heartbeat >= 10:
                        await send(dict(event='heartbeat', data=dict(task_id=task_id)))
                        heartbeat = time.monotonic()
                    if server.event_notifications is not None:
                        await server.event_notifications.wait(task_id, cursor)
                    else:
                        await asyncio.sleep(.05)

            readers = [asyncio.create_task(receive()), asyncio.create_task(events())]
            done, _ = await asyncio.wait(readers, timeout=CONNECTION_SECONDS, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except ServerError as exc:
            if accepted:
                await error(exc)
        except (asyncio.TimeoutError, WebSocketDisconnect, OSError, BrokenResourceError, ClosedResourceError):
            pass
        except Exception:
            if accepted:
                try:
                    await error(ServerError(503, 'stream_unavailable', 'Task stream is temporarily unavailable', retryable=True))
                except (asyncio.TimeoutError, WebSocketDisconnect, OSError, RuntimeError, BrokenResourceError, ClosedResourceError):
                    pass
        finally:
            async def cleanup():
                try:
                    for task in readers:
                        task.cancel()
                    await asyncio.gather(*readers, return_exceptions=True)
                    try:
                        await asyncio.wait_for(ws.close(code=1000 if accepted else 1008), SEND_SECONDS)
                    except (asyncio.TimeoutError, WebSocketDisconnect, OSError, RuntimeError, BrokenResourceError, ClosedResourceError):
                        pass
                finally:
                    if reserved:
                        server.stream_connections -= 1
            # ASGI cancellation may arrive while cleanup itself is awaiting a
            # send. Keep it Server-owned so connection capacity is released.
            await asyncio.shield(server.own(cleanup()))

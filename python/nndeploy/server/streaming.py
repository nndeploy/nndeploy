"""Bounded SSE transport over the existing durable task records."""
import asyncio
import json
import time

from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect

from .db import TERMINAL
from .errors import ServerError


def encode_event(kind, value, sequence=None):
    prefix = '' if sequence is None else f'id: {sequence}\n'
    if sequence is not None:
        value = dict(value, sequence=sequence)
    return (prefix + f'event: {kind}\ndata: ' + json.dumps(value, ensure_ascii=False, allow_nan=False,
                                                        separators=(',', ':')) + '\n\n').encode('utf-8')


def reserve_connection(server):
    if server.stream_connections >= server.max_inflight_tasks:
        raise ServerError(429, 'stream_connections_full', 'Server stream connection limit reached', retryable=True)
    server.stream_connections += 1


def accepts_events(value):
    """Respect explicit exclusions even when a broader media range is present."""
    matches = []
    for part in value.lower().split(','):
        media, *parameters = [item.strip() for item in part.split(';')]
        specificity = {'*/*': 0, 'text/*': 1, 'text/event-stream': 2}.get(media)
        if specificity is None:
            continue
        quality = 1.0
        for parameter in parameters:
            name, _, item = parameter.partition('=')
            if name.strip() == 'q':
                try:
                    quality = float(item.strip())
                except ValueError:
                    return False
        if not 0 <= quality <= 1:
            return False
        matches.append((specificity, quality))
    return bool(matches) and max(matches)[1] > 0


class TaskEventResponse(StreamingResponse):
    """A disconnect ends only this reader; task execution remains Server-owned."""
    def __init__(self, server, task_id, after=0, *, reserved=False, connection_seconds=60, send_seconds=5):
        if not reserved:
            reserve_connection(server)
        self.server = server
        self.connection_seconds = connection_seconds
        self.send_seconds = send_seconds
        super().__init__(self.events(task_id, after), media_type='text/event-stream', headers={
            'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', 'X-Task-ID': task_id})

    async def events(self, task_id, after):
        heartbeat = time.monotonic()
        yield b': connected\n\n'
        while not self.server.shutting_down:
            try:
                batch = await asyncio.to_thread(self.server.db.read_task_events, task_id, after)
                if batch:
                    for event in batch:
                        yield encode_event(event['kind'], event['data'], event['sequence'])
                        after = event['sequence']
                        if event['kind'] == 'done':
                            return
                    continue
                record = await asyncio.to_thread(self.server.db.get_task, task_id)
                if record is None:
                    yield encode_event('error', dict(task_id=task_id, terminal=False,
                        error=dict(code='task_not_found', retryable=False,
                                   message='Task record and retained events are no longer available')))
                    return
                if record['state'] in TERMINAL and after >= record['event_seq']:
                    # A fully consumed stream or a task created before event
                    # persistence was installed has no additional events.
                    return
            except Exception:
                yield encode_event('error', dict(task_id=task_id, terminal=False,
                    error=dict(code='event_store_unavailable', retryable=True)))
                return
            if time.monotonic() - heartbeat >= 10:
                yield b': keep-alive\n\n'
                heartbeat = time.monotonic()
            if self.server.event_notifications is not None:
                await self.server.event_notifications.wait(task_id, after)
            else:
                await asyncio.sleep(.1)

    async def __call__(self, scope, receive, send):
        async def bounded_send(message):
            await asyncio.wait_for(send(message), timeout=self.send_seconds)
        try:
            await asyncio.wait_for(super().__call__(scope, receive, bounded_send), timeout=self.connection_seconds)
        except asyncio.TimeoutError:
            try:
                await bounded_send({'type': 'http.response.body', 'body': b'', 'more_body': False})
            except (asyncio.TimeoutError, OSError):
                pass
        except (OSError, ClientDisconnect):
            # Closing the transport neither cancels computation nor invents a
            # terminal event. Clients reconnect using the last received ID.
            pass
        finally:
            try:
                await self.body_iterator.aclose()
            finally:
                self.server.stream_connections -= 1

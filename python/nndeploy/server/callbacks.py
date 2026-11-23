"""Deliver terminal task notifications independently of model execution."""
import asyncio
import ipaddress
import logging
from urllib.parse import urlsplit


def validate_url(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 2048 or any(ord(c) <= 32 for c in value):
        raise ValueError('callback_url must be an HTTP(S) URL of at most 2048 characters')
    try:
        url = urlsplit(value)
        port = url.port
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username is not None
                or url.password is not None or url.fragment or port == 0):
            raise ValueError()
    except ValueError:
        raise ValueError('callback_url requires HTTP(S), a host, and no credentials or fragment') from None
    return value


def task_values(url, scope):
    return dict(callback_url=url, callback_scope=scope if url else None,
                callback_state='pending' if url else None, callback_attempts=0, callback_next_at=0)


def fingerprint(inputs, parameters, owner, callback_url=None):
    import hashlib
    from .db import encode
    # Preserve fingerprints of requests accepted before callbacks were added.
    value = [inputs, parameters, owner]
    if callback_url is not None:
        value.append(callback_url)
    return hashlib.sha256(encode(value).encode()).hexdigest()


def _attempt_limit(record):
    return (record.get('service') or {}).get('callback_max_attempts', 5)


def claim_progress(db, scope):
    from sqlalchemy import select, update
    from .db import tasks, TERMINAL
    import json
    with db.engine.begin() as connection:
        now = db.clock(connection)
        conditions = (tasks.c.callback_scope == scope, tasks.c.callback_state == 'pending',
                      tasks.c.callback_next_at <= now, ~tasks.c.state.in_(TERMINAL),
                      tasks.c.service.is_not(None), tasks.c.progress_next_at <= now,
                      tasks.c.event_seq > tasks.c.progress_sequence)
        rows = connection.execute(select(tasks.c.task_id, tasks.c.service, tasks.c.event_seq,
                                          tasks.c.progress_sequence).where(*conditions)
                                  .order_by(tasks.c.progress_next_at).limit(32)).all()
        for row in rows:
            service = json.loads(row.service)
            if not service.get('progress_callbacks'):
                continue
            result = connection.execute(update(tasks).where(*conditions,
                tasks.c.task_id == row.task_id, tasks.c.progress_sequence == row.progress_sequence).values(
                progress_sequence=row.event_seq, progress_next_at=now + service.get('progress_interval', 5),
                callback_next_at=now + 60))
            if result.rowcount == 1:
                identity = row.task_id
                break
        else:
            return None
    record = db.get_task(identity)
    record['_progress_callback'] = True
    record['_claimed_sequence'] = row.event_seq
    return record


def claim(db, scope):
    from sqlalchemy import select, update
    from .db import tasks, TERMINAL
    with db.engine.begin() as connection:
        now = db.clock(connection)
        conditions = (tasks.c.callback_scope == scope, tasks.c.callback_state == 'pending',
                      tasks.c.callback_next_at <= now, tasks.c.state.in_(TERMINAL))
        row = connection.execute(select(tasks.c.task_id, tasks.c.callback_attempts, tasks.c.service).where(*conditions)
                                 .order_by(tasks.c.callback_next_at, tasks.c.task_id).limit(1)).first()
        if row is None:
            return None
        values = dict(callback_attempts=row.callback_attempts + 1, callback_next_at=now + 60)
        import json
        maximum = _attempt_limit({'service': json.loads(row.service) if row.service else {}})
        if row.callback_attempts >= maximum:
            values = dict(callback_state='failed', callback_error='Delivery attempt limit reached')
        result = connection.execute(update(tasks).where(*conditions, tasks.c.task_id == row.task_id,
            tasks.c.callback_attempts == row.callback_attempts).values(**values))
        if result.rowcount != 1 or row.callback_attempts >= maximum:
            return None
    return db.get_task(row.task_id)


def send(record):
    import requests
    import socket
    url = record['callback_url']
    target = urlsplit(url)
    # Local receivers are supported; special-purpose/link-local destinations
    # (including cloud metadata endpoints) are not callback receivers.
    for info in socket.getaddrinfo(target.hostname, target.port or (443 if target.scheme == 'https' else 80), type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(info[4][0])
        if address.is_link_local or address.is_multicast or address.is_unspecified:
            raise ValueError('Unsupported callback address')
    payload = {key: record.get(key) for key in ('task_id', 'request_id', 'name', 'state', 'outputs', 'error', 'finished_at')}
    payload['event'] = 'task.finished'
    service = record.get('service') or {}
    if service.get('callback_formatter'):
        from importlib import import_module
        module, function = service['callback_formatter'].split(':')
        payload = getattr(import_module(module), function)(record)
    with requests.Session() as session:
        session.trust_env = False
        with session.post(url, json=payload, headers={'X-NNDeploy-Task-ID': record['task_id']},
                          timeout=(3, service.get('callback_timeout', 5)),
                          allow_redirects=service.get('callback_redirects', False), stream=True) as response:
            return response.status_code


def deliver(db, record):
    from sqlalchemy import update
    from .db import tasks
    status, error = None, None
    if record.get('_progress_callback'):
        from .db import TERMINAL
        current = db.get_task(record['task_id'])
        if current is None or current['state'] in TERMINAL:
            with db.engine.begin() as connection:
                connection.execute(update(tasks).where(tasks.c.task_id == record['task_id'],
                    tasks.c.progress_sequence == record['_claimed_sequence']).values(callback_next_at=0))
            return
    try:
        record['_progress'] = db.latest_progress(record['task_id'])
        status = send(record)
        if not 200 <= status < 300:
            error = f'Callback returned HTTP {status}'
    except Exception as exc:
        # Do not persist/log URL query strings or response bodies as errors.
        error = f'Callback delivery failed ({type(exc).__name__})'
    if record.get('_progress_callback'):
        from .db import encode
        with db.engine.begin() as connection:
            connection.execute(update(tasks).where(tasks.c.task_id == record['task_id'],
                tasks.c.progress_sequence == record['_claimed_sequence']).values(
                callback_next_at=0, progress_delivery=encode(dict(state='delivered' if error is None else 'failed', attempts=1,
                                               status_code=status, error=error))))
        return
    attempts = record['callback_attempts']
    state = 'delivered' if error is None else 'failed' if attempts >= _attempt_limit(record) else 'pending'
    delays = (record.get('service') or {}).get('callback_retry_delays') or (1, 5, 30, 120, 120)
    with db.engine.begin() as connection:
        connection.execute(update(tasks).where(tasks.c.task_id == record['task_id'],
            tasks.c.callback_state == 'pending', tasks.c.callback_attempts == attempts).values(
                callback_state=state, callback_status_code=status, callback_error=error,
                callback_next_at=db.clock(connection) + delays[min(attempts - 1, len(delays) - 1)]))


class Callbacks:
    def __init__(self, db, scope):
        self.db, self.scope = db, scope
        self.stopping = asyncio.Event()
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def run(self):
        while not self.stopping.is_set():
            try:
                record = await asyncio.to_thread(claim, self.db, self.scope)
                if record is None:
                    record = await asyncio.to_thread(claim_progress, self.db, self.scope)
                if record is not None:
                    await asyncio.to_thread(deliver, self.db, record)
                    continue
            except Exception:
                logging.exception('Callback dispatch failed')
            try:
                await asyncio.wait_for(self.stopping.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass

    async def close(self):
        self.stopping.set()
        if self.task is not None:
            await asyncio.shield(self.task)

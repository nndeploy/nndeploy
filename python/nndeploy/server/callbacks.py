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


def claim(db, scope):
    from sqlalchemy import select, update
    from .db import tasks, TERMINAL
    with db.engine.begin() as connection:
        now = db.clock(connection)
        conditions = (tasks.c.callback_scope == scope, tasks.c.callback_state == 'pending',
                      tasks.c.callback_next_at <= now, tasks.c.state.in_(TERMINAL))
        row = connection.execute(select(tasks.c.task_id, tasks.c.callback_attempts).where(*conditions)
                                 .order_by(tasks.c.callback_next_at, tasks.c.task_id).limit(1)).first()
        if row is None:
            return None
        values = dict(callback_attempts=row.callback_attempts + 1, callback_next_at=now + 60)
        if row.callback_attempts >= 5:
            values = dict(callback_state='failed', callback_error='Delivery attempt limit reached')
        result = connection.execute(update(tasks).where(*conditions, tasks.c.task_id == row.task_id,
            tasks.c.callback_attempts == row.callback_attempts).values(**values))
        if result.rowcount != 1 or row.callback_attempts >= 5:
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
    with requests.Session() as session:
        session.trust_env = False
        with session.post(url, json=payload, headers={'X-NNDeploy-Task-ID': record['task_id']},
                          timeout=(3, 5), allow_redirects=False, stream=True) as response:
            return response.status_code


def deliver(db, record):
    from sqlalchemy import update
    from .db import tasks
    status, error = None, None
    try:
        status = send(record)
        if not 200 <= status < 300:
            error = f'Callback returned HTTP {status}'
    except Exception as exc:
        # Do not persist/log URL query strings or response bodies as errors.
        error = f'Callback delivery failed ({type(exc).__name__})'
    attempts = record['callback_attempts']
    state = 'delivered' if error is None else 'failed' if attempts >= 5 else 'pending'
    with db.engine.begin() as connection:
        connection.execute(update(tasks).where(tasks.c.task_id == record['task_id'],
            tasks.c.callback_state == 'pending', tasks.c.callback_attempts == attempts).values(
                callback_state=state, callback_status_code=status, callback_error=error,
                callback_next_at=db.clock(connection) + (1, 5, 30, 120, 120)[min(attempts - 1, 4)]))


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

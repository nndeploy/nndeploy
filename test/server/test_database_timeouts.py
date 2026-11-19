"""Real database contention must release callers and preserve transaction semantics."""
import threading
import time
import uuid

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.exc import SQLAlchemyError

from nndeploy.server import db as database
from nndeploy.server.db import DB, server_meta
from test_db import db
from test_editor_history import location


def test_locked_write_times_out_and_rolls_back_before_retry(db, monkeypatch):
    monkeypatch.setattr(database, 'DATABASE_TIMEOUT_SECONDS', .4, raising=False)
    bounded = DB(db.url).init_schema()
    identity, side = uuid.uuid4().hex, uuid.uuid4().hex
    with db.engine.begin() as connection:
        connection.execute(insert(server_meta).values(key=identity, value='original'))
    outcomes = []
    entered = threading.Event()
    def write():
        try:
            with bounded.engine.begin() as connection:
                entered.set()
                connection.execute(insert(server_meta).values(key=side, value='uncommitted'))
                connection.execute(update(server_meta).where(server_meta.c.key == identity).values(value='updated'))
            outcomes.append('committed')
        except SQLAlchemyError as error:
            outcomes.append(error)
    caller = threading.Thread(target=write)
    try:
        with db.engine.begin() as blocker:
            blocker.execute(update(server_meta).where(server_meta.c.key == identity).values(value='locked'))
            caller.start()
            assert entered.wait(2)
            caller.join(timeout=1.5)
            assert not caller.is_alive(), 'Database lock wait must release the calling thread'
            assert len(outcomes) == 1 and isinstance(outcomes[0], SQLAlchemyError), outcomes
        caller.join(timeout=3)
        with db.engine.connect() as connection:
            assert connection.execute(select(server_meta.c.value).where(server_meta.c.key == side)).scalar_one_or_none() is None
        # A fresh transaction after the timeout is usable, including the pool.
        outcomes.clear()
        write()
        assert outcomes == ['committed']
        assert bounded.ping()
    finally:
        caller.join(timeout=5)
        bounded.close()


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True, '30', 86401, 10**1000])
def test_database_timeout_rejects_invalid_values(tmp_path, value):
    with pytest.raises(ValueError, match='finite positive'):
        DB(tmp_path / 'invalid.db', timeout_seconds=value)


def test_slow_server_query_invalidates_connection_and_pool_recovers(db):
    backend = db.engine.dialect.name
    if backend == 'sqlite':
        pytest.skip('SQLite has no server socket')
    bounded = DB(db.url, timeout_seconds=.4).init_schema()
    try:
        with bounded.engine.connect() as connection:
            started = time.monotonic()
            with pytest.raises(SQLAlchemyError):
                connection.exec_driver_sql('SELECT pg_sleep(3)' if backend == 'postgresql' else 'SELECT SLEEP(3)')
            assert time.monotonic() - started < 2
            assert connection.invalidated
        assert bounded.ping()
    finally:
        bounded.close()


class ReplyProxy:
    """Forward a real DB connection, optionally discard only its server replies."""
    def __init__(self, host, port):
        import socket
        self.host, self.port = host, port
        self.drop = threading.Event()
        self.stop = threading.Event()
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen()
        self.listener.settimeout(.1)
        self.address = self.listener.getsockname()
        self.sessions = []
        self.thread = threading.Thread(target=self.accept)
        self.thread.start()

    def accept(self):
        import socket
        while not self.stop.is_set():
            try:
                client, _ = self.listener.accept()
            except TimeoutError:
                continue
            thread = threading.Thread(target=self.forward, args=(client,))
            self.sessions.append(thread)
            thread.start()

    def forward(self, client):
        import selectors
        import socket
        with client, socket.create_connection((self.host, self.port), timeout=2) as upstream, selectors.DefaultSelector() as poll:
            upstream.settimeout(2)
            client.settimeout(2)
            poll.register(client, selectors.EVENT_READ, upstream)
            poll.register(upstream, selectors.EVENT_READ, client)
            try:
                while not self.stop.is_set():
                    for key, _ in poll.select(.1):
                        data = key.fileobj.recv(65536)
                        if not data:
                            return
                        if key.fileobj is upstream and self.drop.is_set():
                            continue
                        key.data.sendall(data)
            except (ConnectionError, OSError):
                pass  # A timed-out connection is intentionally discarded.

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.listener.close()
        for thread in self.sessions:
            thread.join(timeout=3)
        assert not self.thread.is_alive() and all(not thread.is_alive() for thread in self.sessions)


def test_lost_commit_reply_times_out_and_replay_finds_original_task(db):
    from sqlalchemy import event
    backend = db.engine.dialect.name
    if backend == 'sqlite':
        pytest.skip('SQLite has no server socket')
    proxy = ReplyProxy(db.url.host or '127.0.0.1', db.url.port or (5432 if backend == 'postgresql' else 3306))
    bounded = DB(db.url.set(host=proxy.address[0], port=proxy.address[1]), timeout_seconds=.4).init_schema()
    identity, name, key = uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex
    try:
        @event.listens_for(bounded.engine, 'commit', once=True)
        def lose_reply(connection):
            proxy.drop.set()
        started = time.monotonic()
        with pytest.raises(SQLAlchemyError):
            bounded.create_task(identity, name, time.time(), {'text': 'once'}, {}, idempotency_key=key)
        assert time.monotonic() - started < 2
        proxy.drop.clear()
        assert db.get_task(identity)['inputs'] == {'text': 'once'}
        replay = bounded.create_task(uuid.uuid4().hex, name, time.time(), {'text': 'once'}, {}, idempotency_key=key)
        assert replay == identity
        assert bounded.ping()
    finally:
        proxy.drop.clear()
        bounded.close()
        proxy.close()


def test_connection_pool_wait_is_bounded_and_recovers(db):
    from sqlalchemy.exc import TimeoutError as PoolTimeout
    bounded = DB(db.url, timeout_seconds=.4).init_schema()
    connections = []
    try:
        for _ in range(bounded.engine.pool.size() + bounded.engine.pool._max_overflow):
            connections.append(bounded.engine.connect())
        started = time.monotonic()
        with pytest.raises(PoolTimeout):
            bounded.ping()
        assert time.monotonic() - started < 1.5
        connections.pop().close()
        assert bounded.ping()
    finally:
        for connection in connections:
            connection.close()
        bounded.close()


def test_database_handshake_timeout_does_not_leave_pool_unusable(db):
    backend = db.engine.dialect.name
    if backend == 'sqlite':
        pytest.skip('SQLite has no server socket')
    proxy = ReplyProxy(db.url.host or '127.0.0.1', db.url.port or (5432 if backend == 'postgresql' else 3306))
    proxy.drop.set()
    bounded = DB(db.url.set(host=proxy.address[0], port=proxy.address[1]), timeout_seconds=.4)
    try:
        started = time.monotonic()
        with pytest.raises(SQLAlchemyError):
            bounded.ping()
        # libpq uses integer connect_timeout with a minimum of two seconds.
        assert time.monotonic() - started < 4
        proxy.drop.clear()
        assert bounded.ping()
    finally:
        proxy.drop.clear()
        bounded.close()
        proxy.close()


def test_editor_locked_admission_returns_original_identity_for_retry(tmp_path, location):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from nndeploy.server import Server
    from test_server_api import text_graph
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True),
                    database_url=location, database_timeout_seconds=.4)
    identity = str(uuid.uuid4())
    responses = []
    with TestClient(server.app) as client:
        def submit():
            responses.append(client.post('/api/queue', json=text_graph(), headers={'Idempotency-Key': identity}))
        caller = threading.Thread(target=submit)
        try:
            with server.db.engine.begin() as blocker:
                blocker.execute(update(server_meta).where(server_meta.c.key == server.queue.history_store.counter_key).values(value='10'))
                caller.start()
                caller.join(timeout=2)
                assert not caller.is_alive(), 'An editor request must not stay blocked on a database lock'
                assert responses[0].status_code == 503, responses[0].text
                error = responses[0].json()['error']
                assert error['code'] == 'editor_admission_unconfirmed'
                assert error['task_id'] == identity and 'accepted' not in error
            submit()
            submit()
            assert [reply.status_code for reply in responses[1:]] == [202, 202]
            assert {reply.json()['result']['task_id'] for reply in responses[1:]} == {identity}
            assert len(server.queue.get_current_queue()['PENDING']) == 1
        finally:
            caller.join(timeout=5)


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '86401'])
def test_database_timeout_cli_validation(value):
    from nndeploy.server.app import cli
    with pytest.raises(SystemExit):
        cli(['--database-timeout-seconds', value])


def test_database_timeout_cli_and_server_option(tmp_path):
    from nndeploy.server import Server
    from nndeploy.server.app import cli
    args = cli(['--database-timeout-seconds', '4'])
    server = Server(resources=tmp_path, data_dir=tmp_path, database_timeout_seconds=args.database_timeout_seconds)
    try:
        assert server.db.timeout_seconds == 4
    finally:
        import asyncio
        asyncio.run(server.close())

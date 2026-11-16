"""Automatic maintenance must durably record intent before changing data."""
import asyncio
import json
import threading

import pytest
from sqlalchemy import select

from nndeploy.server.audit import records
from test_audit import make
from test_db import db
from test_file_cleanup import upload, expire as expire_file
from test_retention import create, expire as expire_task


def system_records(server):
    with server.db.engine.connect() as connection:
        return [dict(row) for row in connection.execute(select(records).where(
            records.c.scope == server.audit.scope, records.c.method == 'SYSTEM'
        ).order_by(records.c.id)).mappings()]


async def cycle(server):
    """Stop at the real maintenance loop's next timer, after exactly one batch."""
    original = server.maintenance_stop
    class OneBatch:
        async def wait(self):
            server.shutting_down = True
            return True
    server.maintenance_stop = OneBatch()
    try:
        await server._maintain_tasks()
    finally:
        server.shutting_down = False
        server.maintenance_stop = original


@pytest.mark.parametrize('kind', ['tasks', 'files'])
def test_automatic_cleanup_requires_its_own_intent_and_recovers(db, tmp_path, monkeypatch, kind):
    server = make(db, tmp_path)
    if kind == 'tasks':
        identity = create(db)
        db.finish_task(identity, 'succeeded', db.clock(), outputs={'text': 'private-output'})
        expire_task(db, identity)
        exists = lambda: db.get_task(identity)
    else:
        identity = upload(server.storage)
        expire_file(db, identity)
        exists = lambda: db.get_file(identity)
    route = 'Server.cleanup_' + kind
    original_begin = server.audit._begin
    attempts = []
    def unavailable(request_id, method, context=None):
        if method == 'SYSTEM' and context['route'] == route:
            attempts.append(request_id)
            raise OSError('private-audit-error')
        return original_begin(request_id, method, context)

    async def scenario():
        try:
            with monkeypatch.context() as patch:
                patch.setattr(server.audit, '_begin', unavailable)
                # The background operation cannot hide inside a caller's audit.
                with server.audit.track():
                    await cycle(server)
            assert attempts and exists() is not None
            assert not any(row['route'] == route for row in system_records(server))
            await cycle(server)
            assert exists() is None
            row, = [row for row in system_records(server) if row['route'] == route]
            assert row['state'] == 'complete' and row['status_code'] == 200
            assert row['subject'] is None and not row['authenticated'] and not row['administrator']
            assert 'private-output' not in json.dumps(row) and 'private-audit-error' not in json.dumps(row)
        finally:
            await server.close()
    asyncio.run(scenario())


def test_every_automatic_maintenance_action_has_a_separate_record(db, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from nndeploy.server.logging_taskid import get_request_id
    server = make(db, tmp_path)
    seen = []
    def observe(route, result):
        def action(*args):
            row, = [r for r in system_records(server) if r['route'] == 'Server.' + route]
            assert row['route'] == 'Server.' + route and row['state'] == 'started'
            seen.append(route)
            return result
        return action
    server.queue = SimpleNamespace(cleanup_history=observe('cleanup_editor_history', 0))
    server.editor_history_retention_seconds = 60
    server.devices = SimpleNamespace(recover=observe('devices.recover', 0))
    monkeypatch.setattr(server.db, 'cleanup_tasks', observe('cleanup_tasks', 0))
    monkeypatch.setattr(server.storage, 'cleanup', observe('cleanup_files', {'deleted': 0}))
    monkeypatch.setattr(server.controls, 'reconcile', observe('controls.reconcile', 0))
    monkeypatch.setattr(server.controls, 'cleanup', observe('controls.cleanup', 0))
    monkeypatch.setattr(server.audit, '_cleanup', observe('audit.cleanup', 0))
    async def scenario():
        try:
            with server.audit.track():
                await cycle(server)
            assert seen == ['cleanup_editor_history', 'cleanup_tasks', 'cleanup_files',
                            'controls.reconcile', 'controls.cleanup', 'audit.cleanup', 'devices.recover']
            rows = system_records(server)
            assert len(rows) == 7 and len({row['request_id'] for row in rows}) == 7
            assert all(row['state'] == 'complete' and row['status_code'] == 200 for row in rows)
            assert get_request_id() is None
        finally:
            await server.close()
    asyncio.run(scenario())


def test_failed_background_action_is_recorded_and_other_cleanup_continues(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    identity = upload(server.storage)
    expire_file(db, identity)
    def fail():
        raise OSError('private-cleanup-error')
    monkeypatch.setattr(server.db, 'cleanup_tasks', fail)
    async def scenario():
        try:
            await cycle(server)
            row, = [r for r in system_records(server) if r['route'] == 'Server.cleanup_tasks']
            assert row['state'] == 'complete' and row['status_code'] == 500
            assert 'private-cleanup-error' not in json.dumps(row)
            assert db.get_file(identity) is None
            assert all(r['status_code'] == 200 for r in system_records(server) if r['route'] != row['route'])
        finally:
            await server.close()
    asyncio.run(scenario())


def test_unconfirmed_background_outcome_keeps_deleted_data_deleted(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    identity = upload(server.storage)
    expire_file(db, identity)
    finish = server.audit._finish
    def unavailable(request_id, context, *args):
        if context['route'] == 'Server.cleanup_files':
            raise OSError('private-completion-error')
        return finish(request_id, context, *args)
    async def scenario():
        try:
            with monkeypatch.context() as patch:
                patch.setattr(server.audit, '_finish', unavailable)
                await cycle(server)
            assert db.get_file(identity) is None
            row, = [r for r in system_records(server) if r['route'] == 'Server.cleanup_files']
            assert row['state'] == 'started' and row['expires_at'] is None
            await cycle(server)
            rows = [r for r in system_records(server) if r['route'] == row['route']]
            assert len(rows) == 2 and rows[0]['state'] == 'started' and rows[1]['state'] == 'complete'
        finally:
            await server.close()
    asyncio.run(scenario())


def test_server_close_waits_for_background_audit_sql(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    begin, close = server.audit._begin, server.db.close
    def blocked(request_id, method, context=None):
        if method == 'SYSTEM' and context['route'] == 'Server.cleanup_tasks':
            entered.set()
            assert release.wait(10)
            assert not closed.is_set()
        return begin(request_id, method, context)
    def close_db():
        closed.set()
        close()
    monkeypatch.setattr(server.audit, '_begin', blocked)
    monkeypatch.setattr(server.db, 'close', close_db)
    async def scenario():
        server.task_maintenance = asyncio.create_task(server._maintain_tasks())
        stopping = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            stopping = asyncio.create_task(server.close())
            await asyncio.sleep(.02)
            assert not stopping.done() and not closed.is_set()
        finally:
            release.set()
            await asyncio.wait_for(stopping if stopping is not None else server.close(), 5)
        assert closed.is_set() and not server.audit.pending
    asyncio.run(scenario())


def test_disabled_audit_keeps_background_cleanup_available(db, tmp_path):
    server = make(db, tmp_path, audit_config=None)
    identity = upload(server.storage)
    expire_file(db, identity)
    async def scenario():
        try:
            await cycle(server)
            assert db.get_file(identity) is None
            assert system_records(server) == []
        finally:
            await server.close()
    asyncio.run(scenario())

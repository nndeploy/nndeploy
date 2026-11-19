"""Recover known control outcomes without executing their actions again."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import threading
import uuid

import httpx
import pytest
from sqlalchemy import select, update

from nndeploy.server import Server
from nndeploy.server.control import receipts
from nndeploy.server.db import encode
from test_db import db
from test_control_http import AUTH, make
from test_shared_server import until


REQUEST_ID = 'req_' + 'c' * 32


def crash_after_result(url, directory, marker, committed, failed=False):
    async def scenario():
        server = Server(resources=directory, data_dir=directory, database_url=url, admin_token='management-token')
        original = server.controls.finish
        def finish(*args):
            if committed:
                original(*args)
            Path(marker).write_text('finish reached')
            threading.Event().wait(60)
            raise OSError('test database unavailable')
        server.controls.finish = finish
        async def action():
            with (Path(directory) / 'calls').open('a') as output:
                output.write('action\n')
            if failed:
                from nndeploy.server.errors import ServerError
                raise ServerError(409, 'entry_busy', 'Original failure')
            return {'name': 'model', 'status': 'ready', 'original': True}
        await server.controls.request(REQUEST_ID, 'management', 'model', 'start', {}, 'recover', action, 0)
        await asyncio.Event().wait()
    asyncio.run(scenario())


@pytest.mark.parametrize('committed,failed', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('recover_on_start', [False, True])
def test_crashed_controller_replays_known_result_and_never_reexecutes(db, tmp_path, committed, failed, recover_on_start):
    directory, marker = tmp_path / 'server', tmp_path / 'finish'
    process = multiprocessing.get_context('spawn').Process(target=crash_after_result,
        args=(db.url.render_as_string(hide_password=False), directory, marker, committed, failed))
    async def scenario():
        server = None
        try:
            process.start()
            await until(marker.exists)
            process.kill()
            await asyncio.to_thread(process.join, 5)
            assert not process.is_alive()
            server = make(db, directory)
            async def forbidden(name):
                pytest.fail('A recovered control action executed again')
            server.start_entry = forbidden
            if recover_on_start:
                await server.start()
                with server.db.engine.connect() as connection:
                    state = connection.execute(select(receipts.c.state).where(receipts.c.scope == server.controls.scope,
                        receipts.c.request_id == REQUEST_ID)).scalar_one()
                    assert state == ('failed' if failed else 'succeeded')
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                result = await client.get(server.controls.url(REQUEST_ID), headers=AUTH)
                assert result.status_code == (409 if failed else 200), result.text
                expected = result.json()
                if failed:
                    assert expected['error']['code'] == 'entry_busy'
                else:
                    assert expected['original'] is True
                replay = await client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': 'recover'}))
                assert replay.status_code == result.status_code and replay.json() == expected
                assert (directory / 'calls').read_text() == 'action\n'
                assert not list(server.controls.results.glob('*.json'))
                assert not (directory / 'control-locks' / REQUEST_ID[4:]).exists()
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                await asyncio.to_thread(process.join, 5)
            if server:
                await server.close()
    asyncio.run(scenario())


def prepare_result(server, *, key='recover'):
    record = server.controls.reserve('req_' + uuid.uuid4().hex, 'management', 'model', 'start', {}, key)
    server.controls.checkpoint(record, 200, {'status': 'ready', 'original': True})
    return record, server.controls.result_path(record['request_id'])


@pytest.mark.parametrize('damage', ['checksum', 'truncated', 'fingerprint', 'database', 'owner', 'scope', 'request_id'])
def test_invalid_or_mismatched_result_never_changes_receipt(db, tmp_path, damage):
    server = make(db, tmp_path)
    try:
        record, path = prepare_result(server)
        original = path.read_bytes()
        payload = json.loads(original)
        if damage == 'checksum':
            payload['response']['original'] = False
        elif damage == 'database':
            payload['database'] = 'different-database'
        else:
            payload['record'][damage] = 'different'
        if damage not in ('checksum', 'truncated'):
            payload.pop('sha256')
            payload['sha256'] = hashlib.sha256(encode(payload).encode()).hexdigest()
        path.write_bytes(original[:30] if damage == 'truncated' else encode(payload).encode())
        assert server.controls.get(record['request_id'], 'management')['state'] == 'running'
        server.controls.reconcile()
        assert server.controls.get(record['request_id'], 'management')['response'] is None
        assert path.exists()  # Preserve rejected evidence; do not fabricate a result.
    finally:
        asyncio.run(server.close())


def test_result_recovery_does_not_require_same_boot_or_guard_file(db, tmp_path, monkeypatch):
    from nndeploy.server import control_guard
    server = make(db, tmp_path)
    try:
        record, path = prepare_result(server)
        monkeypatch.setattr(control_guard, '_boot_id', lambda: None)
        with db.engine.begin() as connection:
            connection.execute(update(receipts).where(receipts.c.identity == record['identity']).values(state='unconfirmed'))
        result = server.controls.get(record['request_id'], 'management')
        assert result['state'] == 'succeeded' and json.loads(result['response'])['original'] is True
        assert not path.exists()
    finally:
        asyncio.run(server.close())


def test_bounded_reconciliation_continues_past_bad_file_and_retries_db_failure(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    try:
        records = [prepare_result(server, key=f'key-{index}')[0] for index in range(7)]
        (server.controls.results / 'invalid.json').write_bytes(b'broken')
        original = server.controls.finish
        failed = False
        def transient(*args):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError('Database unavailable')
            return original(*args)
        monkeypatch.setattr(server.controls, 'finish', transient)
        with pytest.raises(OSError):
            server.controls.reconcile(limit=2)
        for _ in range(12):
            assert 0 <= server.controls.reconcile(limit=2) <= 2
        with db.engine.connect() as connection:
            assert set(connection.execute(select(receipts.c.state).where(
                receipts.c.identity.in_([record['identity'] for record in records]))).scalars()) == {'succeeded'}
        assert [path.name for path in server.controls.results.glob('*.json')] == ['invalid.json']
    finally:
        asyncio.run(server.close())


def test_checkpoint_atomic_publication_and_immutable_result(db, tmp_path, monkeypatch):
    import nndeploy.server.control as module
    server = make(db, tmp_path)
    try:
        record = server.controls.reserve(REQUEST_ID, 'management', 'model', 'start', {}, 'atomic')
        def fail_link(*args):
            raise OSError('Atomic publication unavailable')
        with monkeypatch.context() as patch:
            patch.setattr(module.os, 'link', fail_link)
            with pytest.raises(OSError):
                server.controls.checkpoint(record, 200, {'original': True})
        assert not list(server.controls.results.iterdir())
        server.controls.checkpoint(record, 200, {'original': True})
        path = server.controls.result_path(REQUEST_ID)
        original = path.read_bytes()
        if module.os.name != 'nt':
            assert path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(RuntimeError, match='changed'):
            server.controls.checkpoint(record, 200, {'original': False})
        assert path.read_bytes() == original
        assert json.loads(server.controls.get(REQUEST_ID, 'management')['response']) == {'original': True}
    finally:
        asyncio.run(server.close())


def test_checkpoint_failure_still_commits_response_to_healthy_database(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    async def scenario():
        async def action(name):
            return {'status': 'ready'}
        def no_disk(*args):
            raise OSError('Disk full')
        server.start_entry = action
        monkeypatch.setattr(server.controls, 'checkpoint', no_disk)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/api/server/model/start', headers=AUTH)
                assert response.status_code == 200 and response.json()['status'] == 'ready'
                assert (await client.get(response.headers['location'], headers=AUTH)).content == response.content
        finally:
            await server.close()
    asyncio.run(scenario())


def test_concurrent_control_results_cannot_overwrite_first_commit(db, tmp_path):
    server = make(db, tmp_path)
    try:
        record = server.controls.reserve(REQUEST_ID, 'management', 'model', 'start', {}, 'conflict')
        ready = threading.Barrier(8)
        def finish(index):
            ready.wait(timeout=5)
            try:
                server.controls.finish(record, 200, {'winner': index})
                return index
            except RuntimeError:
                return None
        with ThreadPoolExecutor(8) as pool:
            winners = [value for value in pool.map(finish, range(8)) if value is not None]
        assert len(winners) == 1
        assert json.loads(server.controls.get(REQUEST_ID, 'management')['response']) == {'winner': winners[0]}
    finally:
        asyncio.run(server.close())


def test_expired_control_has_only_one_replacement_and_rejects_old_finish(db, tmp_path):
    server = make(db, tmp_path)
    try:
        old = server.controls.reserve(REQUEST_ID, 'management', 'model', 'start', {}, 'expired')
        server.controls.finish(old, 200, {'old': True})
        with db.engine.begin() as connection:
            connection.execute(update(receipts).where(receipts.c.identity == old['identity']).values(expires_at=0))
        ready = threading.Barrier(8)
        def reserve(index):
            ready.wait(timeout=5)
            return server.controls.reserve('req_' + uuid.uuid4().hex, 'management', 'model', 'stop', {}, 'expired')
        with ThreadPoolExecutor(8) as pool:
            records = list(pool.map(reserve, range(8)))
        assert len({record['request_id'] for record in records}) == 1
        assert records[0]['request_id'] != old['request_id']
        with pytest.raises(RuntimeError, match='no longer'):
            server.controls.finish(old, 200, {'late': True})
        assert server.controls.get(records[0]['request_id'], 'management')['state'] == 'running'
    finally:
        asyncio.run(server.close())


def test_response_snapshot_is_independent_before_checkpoint_thread(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    shared = {'status': 'ready', 'schema': {'version': 1}}
    checkpoint = server.controls.checkpoint
    def changing_schema(record, status, response):
        shared['schema']['version'] = 2
        checkpoint(record, status, response)
    monkeypatch.setattr(server.controls, 'checkpoint', changing_schema)
    async def scenario():
        async def action(name):
            return shared
        server.start_entry = action
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                result = await client.post('/api/server/model/start', headers=AUTH)
                assert result.status_code == 200 and result.json()['schema']['version'] == 1
                assert shared['schema']['version'] == 2
                assert (await client.get(result.headers['location'], headers=AUTH)).content == result.content
        finally:
            await server.close()
    asyncio.run(scenario())


def test_checkpoint_io_keeps_http_responsive_and_shutdown_joins_it(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    entered, release = threading.Event(), threading.Event()
    checkpoint = server.controls.checkpoint
    calls = []
    event_loop_thread = threading.get_ident()
    def blocked(*args):
        assert threading.get_ident() != event_loop_thread
        entered.set()
        assert release.wait(10)
        checkpoint(*args)
    monkeypatch.setattr(server.controls, 'checkpoint', blocked)
    async def scenario():
        restored = None
        async def action(name):
            calls.append(name)
            return {'status': 'ready'}
        server.start_entry = action
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                headers = dict(AUTH, **{'Idempotency-Key': 'slow-result'})
                result = await client.post('/api/server/model/start?wait_timeout_seconds=0', headers=headers)
                assert result.status_code == 202
                await until(entered.is_set)
                assert (await asyncio.wait_for(client.get('/health/live'), 1)).status_code == 200
                duplicate = await asyncio.wait_for(client.post('/api/server/model/start?wait_timeout_seconds=0', headers=headers), 1)
                assert duplicate.status_code == 202 and duplicate.json()['request_id'] == result.json()['request_id']
                closing = asyncio.create_task(server.close())
                await until(lambda: server.shutting_down)
                assert not closing.done()
                release.set()
                await asyncio.wait_for(closing, 5)
                assert calls == ['model']
                restored = make(db, tmp_path)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored.app), base_url='http://test') as other:
                    response = await other.get(result.headers['location'], headers=AUTH)
                    assert response.status_code == 200 and response.json()['status'] == 'ready'
        finally:
            release.set()
            await server.close()
            if restored:
                await restored.close()
    asyncio.run(scenario())


def test_terminal_receipt_removes_only_its_guard_after_boot_change(db, tmp_path, monkeypatch):
    from nndeploy.server import control_guard
    server = make(db, tmp_path)
    try:
        record = server.controls.reserve(REQUEST_ID, 'management', 'model', 'start', {}, 'terminal')
        http = control_guard.ControlGuard.create(server.data_dir, REQUEST_ID[4:], REQUEST_ID[4:])
        runtime_id = uuid.uuid4().hex
        runtime = control_guard.ControlGuard.create(server.data_dir, runtime_id, runtime_id)
        http.close()
        runtime.close()
        server.controls.finish(record, 200, {'ok': True})
        monkeypatch.setattr(control_guard, '_boot_id', lambda: None)
        assert server.controls.get(REQUEST_ID, 'management')['state'] == 'succeeded'
        assert not http.path.exists()
        assert runtime.path.exists()
    finally:
        asyncio.run(server.close())


def test_multiple_servers_reconcile_same_file_without_changing_result(db, tmp_path):
    servers = [make(db, tmp_path) for _ in range(4)]
    try:
        record, path = prepare_result(servers[0])
        ready = threading.Barrier(len(servers))
        def recover(server):
            ready.wait(timeout=5)
            return server.controls.get(record['request_id'], 'management')
        with ThreadPoolExecutor(len(servers)) as pool:
            list(pool.map(recover, servers))
        results = [server.controls.get(record['request_id'], 'management') for server in servers]
        assert {result['state'] for result in results} == {'succeeded'}
        assert len({result['response'] for result in results}) == 1
        assert json.loads(results[0]['response']) == {'status': 'ready', 'original': True}
        assert not path.exists()
    finally:
        async def close():
            await asyncio.gather(*(server.close() for server in servers))
        asyncio.run(close())

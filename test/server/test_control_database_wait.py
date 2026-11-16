"""Slow receipt I/O must not pin HTTP callers or create unbounded DB work."""
import asyncio
import threading

import httpx
import pytest

from nndeploy.server import control
from test_control_http import AUTH, make, complete
from test_db import db
from test_shared_server import until


def test_slow_admission_returns_uncertainty_and_same_key_retries_share_one_write(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    release, entered = threading.Event(), threading.Event()
    writes, actions = [], []
    reserve = server.controls.reserve
    def blocked(*args):
        writes.append(args[0])
        entered.set()
        assert release.wait(10)
        return reserve(*args)
    async def action(name):
        actions.append(name)
        return {'name': name, 'status': 'ready'}
    monkeypatch.setattr(server.controls, 'reserve', blocked)
    monkeypatch.setattr(server, 'start_entry', action)
    monkeypatch.setattr(control, 'DATABASE_WAIT_SECONDS', .05, raising=False)
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                headers = dict(AUTH, **{'Idempotency-Key': 'slow-start'})
                responses = await asyncio.wait_for(asyncio.gather(*(
                    client.post('/api/server/model/start?wait_timeout_seconds=0', headers=headers) for _ in range(8))), 1)
                assert entered.is_set() and len(writes) == 1 and not actions
                assert {response.status_code for response in responses} == {503}
                details = [response.json()['error'] for response in responses]
                assert all(value['retryable'] and 'accepted' not in value for value in details)
                assert all('status_url' not in value for value in details)
                assert {response.headers['idempotency-key'] for response in responses} == {'slow-start'}
                conflict = await client.post('/api/server/model/stop', headers=headers)
                assert conflict.status_code == 409
                release.set()
                await until(lambda: actions == ['model'])
                replay = await client.post('/api/server/model/start', headers=headers)
                assert replay.status_code == 200 and actions == ['model']
                assert (await complete(client, replay.headers['location'])).content == replay.content
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_slow_status_reads_are_bounded_coalesced_and_leave_deploy_available(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    release = threading.Event()
    reads = []
    get = server.controls.get
    def blocked(*args):
        reads.append(args)
        assert release.wait(10)
        return get(*args)
    monkeypatch.setattr(control, 'DATABASE_WAIT_SECONDS', .05, raising=False)
    async def scenario():
        try:
            await server.start()
            server.add(name='model', deploy=lambda values: values)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                ready = await client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': 'lookup'}))
                assert ready.status_code == 200
                location = ready.headers['location']
                monkeypatch.setattr(server.controls, 'get', blocked)
                responses = await asyncio.wait_for(asyncio.gather(*(client.get(location, headers=AUTH) for _ in range(8))), 1)
                assert {response.status_code for response in responses} == {503} and len(reads) == 1
                assert (await asyncio.wait_for(server.deploy('model', {'ok': True}), 1))['outputs'] == {'ok': True}
                release.set()
                assert (await complete(client, location)).status_code == 200
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_control_capacity_rejects_new_work_without_cancelling_unknown_writes(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    release = threading.Event()
    actions, writes = [], []
    reserve = server.controls.reserve
    def blocked(*args):
        writes.append(args[0])
        assert release.wait(10)
        return reserve(*args)
    async def action(name):
        actions.append(name)
        return {'name': name, 'status': 'ready'}
    monkeypatch.setattr(server.controls, 'reserve', blocked)
    monkeypatch.setattr(server, 'start_entry', action)
    monkeypatch.setattr(control, 'DATABASE_WAIT_SECONDS', .05, raising=False)
    monkeypatch.setattr(control, 'MAX_CONTROL_OPERATIONS', 2, raising=False)
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                for name in ('first', 'second'):
                    response = await asyncio.wait_for(client.post(f'/api/server/{name}/start',
                        headers=dict(AUTH, **{'Idempotency-Key': name})), 1)
                    assert response.status_code == 503 and response.json()['error']['retryable']
                rejected = await asyncio.wait_for(client.post('/api/server/third/start', headers=AUTH), 1)
                assert rejected.status_code == 503 and rejected.json()['error']['accepted'] is False
                assert len(writes) == 2 and not actions
                release.set()
                await until(lambda: len(actions) == 2)
                accepted = await client.post('/api/server/third/start', headers=AUTH)
                assert accepted.status_code == 200 and actions.count('third') == 1
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_disconnect_before_receipt_confirmation_keeps_one_original_operation(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    release, entered = threading.Event(), threading.Event()
    reserve = server.controls.reserve
    actions = []
    def blocked(*args):
        entered.set()
        assert release.wait(10)
        return reserve(*args)
    async def action(name):
        actions.append(name)
        return {'name': name, 'status': 'ready'}
    monkeypatch.setattr(server.controls, 'reserve', blocked)
    monkeypatch.setattr(server, 'start_entry', action)
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                headers = dict(AUTH, **{'Idempotency-Key': 'disconnect-before-acceptance'})
                request = asyncio.create_task(client.post('/api/server/model/start', headers=headers))
                await until(entered.is_set)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                release.set()
                response = await client.post('/api/server/model/start', headers=headers)
                assert response.status_code == 200 and actions == ['model']
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_accepted_control_returns_202_when_result_lookup_is_slow(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    release = threading.Event()
    action_release = asyncio.Event()
    original = server.controls.get
    def blocked(*args):
        assert release.wait(10)
        return original(*args)
    async def action(name):
        await action_release.wait()
        return {'name': name, 'status': 'ready'}
    monkeypatch.setattr(control, 'DATABASE_WAIT_SECONDS', .05, raising=False)
    monkeypatch.setattr(server.controls, 'get', blocked)
    monkeypatch.setattr(server, 'start_entry', action)
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await asyncio.wait_for(client.post('/api/server/model/start?wait_timeout_seconds=0.05',
                    headers=dict(AUTH, **{'Idempotency-Key': 'read-after-acceptance'})), 1)
                assert response.status_code == 202
                query = await asyncio.wait_for(client.get(response.headers['location'], headers=AUTH), 1)
                assert query.status_code == 503 and query.json()['error']['retryable']
                release.set()
                action_release.set()
                assert (await complete(client, response.headers['location'])).status_code == 200
        finally:
            release.set()
            action_release.set()
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('existing', [False, True], ids=['generated-key', 'older-receipt'])
def test_unconfirmed_admission_returns_a_safe_key_even_when_receipt_id_is_unknown(db, tmp_path, monkeypatch, existing):
    server = make(db, tmp_path)
    release = threading.Event()
    actions = []
    reserve = server.controls.reserve
    def blocked(*args):
        assert release.wait(10)
        return reserve(*args)
    async def action(name):
        actions.append(name)
        return {'name': name, 'status': 'ready'}
    monkeypatch.setattr(control, 'DATABASE_WAIT_SECONDS', .05, raising=False)
    monkeypatch.setattr(server, 'start_entry', action)
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                headers = dict(AUTH, **({'Idempotency-Key': 'already-completed'} if existing else {}))
                invalid = await client.post('/api/server/model/start',
                    headers=dict(AUTH, **{'Idempotency-Key': 'invalid\r\nInjected: true'}))
                assert invalid.status_code == 422 and not actions
                if existing:
                    original = await client.post('/api/server/model/start', headers=headers)
                    assert original.status_code == 200
                monkeypatch.setattr(server.controls, 'reserve', blocked)
                response = await client.post('/api/server/model/start', headers=headers)
                assert response.status_code == 503 and 'location' not in response.headers
                assert 'status_url' not in response.json()['error']
                key = response.headers['idempotency-key']
                release.set()
                replay = await client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': key}))
                assert replay.status_code == 200 and actions == ['model']
                assert replay.headers['idempotency-key'] == key
                if existing:
                    assert replay.headers['location'] == original.headers['location']
                assert (await complete(client, replay.headers['location'])).content == replay.content
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_lookup_capacity_is_bounded_and_shutdown_joins_inflight_database_work(db, tmp_path, monkeypatch):
    server = make(db, tmp_path)
    release = threading.Event()
    reads, closed = [], []
    original_get, original_close = server.controls.get, server.db.close
    def blocked(*args):
        reads.append(args)
        assert release.wait(10)
        assert not closed
        return original_get(*args)
    def close_db():
        closed.append(True)
        original_close()
    async def action(name):
        return {'name': name, 'status': 'ready'}
    monkeypatch.setattr(control, 'DATABASE_WAIT_SECONDS', .05, raising=False)
    monkeypatch.setattr(control, 'MAX_CONTROL_QUERIES', 1, raising=False)
    monkeypatch.setattr(server, 'start_entry', action)
    monkeypatch.setattr(server.db, 'close', close_db)
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                urls = [(await client.post(f'/api/server/{name}/start', headers=AUTH)).headers['location']
                        for name in ('first', 'second')]
                monkeypatch.setattr(server.controls, 'get', blocked)
                for url in urls:
                    response = await asyncio.wait_for(client.get(url, headers=AUTH), 1)
                    assert response.status_code == 503 and response.json()['error']['retryable']
                assert len(reads) == 1
                closing = asyncio.create_task(server.close())
                await asyncio.sleep(.03)
                assert not closing.done() and not closed
                release.set()
                await asyncio.wait_for(closing, 3)
                assert closed == [True]
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())

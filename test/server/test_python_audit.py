"""Direct Server calls share durable audit without duplicating transport records."""
import asyncio
import json
import threading

import httpx
import pytest
from sqlalchemy import select

from nndeploy.server.audit import records
from nndeploy.server.errors import ServerError
from nndeploy.server.logging_taskid import get_request_id
from test_audit import make
from test_db import db
from test_control import configuration
from test_server_api import text_graph


def operations(server, *, include_start=False):
    with server.db.engine.connect() as connection:
        return [dict(row) for row in connection.execute(select(records).where(
            records.c.scope == server.audit.scope, records.c.method == 'PYTHON',
            True if include_start else records.c.route != 'Server.start'
        ).order_by(records.c.id)).mappings()]


@pytest.mark.parametrize('source', ['code', 'graph'])
def test_direct_deploy_correlates_task_and_audits_only_metadata(db, tmp_path, source):
    seen = []
    def echo(inputs):
        row, = operations(server)
        assert row['state'] == 'started' and row['route'] == 'Server.deploy'
        seen.append(get_request_id())
        return inputs
    options = dict(deploy=echo) if source == 'code' else dict(workflow=text_graph(), config=configuration('default'))
    server = make(db, tmp_path, name='echo', **options)
    async def scenario():
        await server.start()
        try:
            startup, = operations(server, include_start=True)
            assert startup['route'] == 'Server.start' and startup['state'] == 'complete'
            result = await server.deploy('echo', {'text': 'private-request-text'})
            assert result['outputs'] == {'text': 'private-request-text'}
            row, = operations(server)
            assert row['state'] == 'complete' and row['status_code'] == 200
            assert row['resource_id'] == result['task_id'] and row['entry'] == 'echo'
            assert server.db.get_task(result['task_id'])['request_id'] == row['request_id']
            assert row['subject'] is None and not row['authenticated'] and not row['administrator']
            assert 'private-request-text' not in json.dumps(row)
            if source == 'code':
                assert seen == [row['request_id']]
            assert get_request_id() is None
        finally:
            await server.close()
    asyncio.run(scenario())


def test_submit_retry_and_batch_preserve_task_identity_and_parent_operation(db, tmp_path):
    calls = []
    server = make(db, tmp_path, name='echo', deploy=lambda x: calls.append(x) or x)
    async def scenario():
        await server.start()
        try:
            first, waiter = await server.submit('echo', {'text': 'one'}, idempotency_key='private-idempotency')
            await waiter
            second, waiter = await server.submit('echo', {'text': 'one'}, idempotency_key='private-idempotency')
            await waiter
            assert first == second and len(calls) == 1
            rows = operations(server)
            assert len(rows) == 2 and all(row['resource_id'] == first and row['status_code'] == 202 for row in rows)
            assert rows[0]['request_id'] != rows[1]['request_id']
            assert server.db.get_task(first)['request_id'] == rows[0]['request_id']
            result = await server.deploy_batch('echo', [{'inputs': {'text': str(index)}} for index in range(3)])
            rows = operations(server)
            assert len(rows) == 3 and rows[-1]['route'] == 'Server.deploy_batch'
            assert rows[-1]['resource_id'] is None
            assert all(item['status_code'] == 200 for item in result['items'])
            assert all(server.db.get_task(item['task_id'])['request_id'] == rows[-1]['request_id'] for item in result['items'])
            assert 'private-idempotency' not in json.dumps(rows)
        finally:
            await server.close()
    asyncio.run(scenario())


def test_direct_controls_and_failed_reload_are_audited_without_workflow_contents(db, tmp_path):
    server = make(db, tmp_path, name='echo', workflow=text_graph(), config=configuration('old'))
    async def scenario():
        await server.start()
        try:
            await server.stop_entry('echo')
            await server.start_entry('echo')
            await server.reload_entry('echo', config=configuration('new'))
            with pytest.raises(ValueError, match='Unknown node'):
                await server.reload_entry('echo', workflow={'key_': 'private-invalid-node'})
            assert (await server.deploy('echo', {}))['outputs'] == {'text': 'new'}
            rows = operations(server)
            assert [row['route'] for row in rows] == ['Server.stop_entry', 'Server.start_entry',
                'Server.reload_entry', 'Server.reload_entry', 'Server.deploy']
            assert all(row['state'] == 'complete' for row in rows)
            assert rows[3]['status_code'] >= 400 and rows[2]['status_code'] == 200
            assert 'private-invalid-node' not in json.dumps(rows)
        finally:
            await server.close()
    asyncio.run(scenario())


def test_unconfirmed_python_audit_blocks_action_and_finish_failure_does_not_rewrite_result(db, tmp_path, monkeypatch):
    calls = []
    server = make(db, tmp_path, name='echo', deploy=lambda x: calls.append(x) or x)
    def fail(*args):
        raise RuntimeError('private-database-error')
    async def scenario():
        await server.start()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(server.audit, '_begin', fail)
                with pytest.raises(ServerError) as error:
                    await server.deploy('echo', {})
                assert error.value.code == 'audit_unavailable' and not calls
            with monkeypatch.context() as patch:
                patch.setattr(server.audit, '_finish', fail)
                result = await server.deploy('echo', {'text': 'success'})
            assert result['outputs'] == {'text': 'success'} and len(calls) == 1
            row, = operations(server)
            assert row['state'] == 'started' and row['finished_at'] is None
            assert server.db.get_task(result['task_id'])['request_id'] == row['request_id']
        finally:
            await server.close()
    asyncio.run(scenario())


def test_cancelled_python_waiter_is_audited_without_cancelling_execution(db, tmp_path):
    entered, release = threading.Event(), threading.Event()
    def held(inputs):
        entered.set()
        assert release.wait(10)
        return inputs
    server = make(db, tmp_path, name='echo', deploy=held)
    async def scenario():
        await server.start()
        waiter = asyncio.create_task(server.deploy('echo', {'text': 'kept'}))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            row, = operations(server)
            with db.engine.connect() as connection:
                from nndeploy.server.db import tasks
                task_id = connection.execute(select(tasks.c.task_id).where(tasks.c.request_id == row['request_id'])).scalar_one()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            row, = operations(server)
            assert row['state'] == 'interrupted' and row['status_code'] == 499
            assert not server.db.get_task(task_id)['cancel_requested']
            release.set()
            # A new observer can still wait for the original execution.
            _, observer = server._record_future(server.db.get_task(task_id))
            assert (await asyncio.wait_for(observer, 5))['outputs'] == {'text': 'kept'}
        finally:
            release.set()
            await asyncio.gather(waiter, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())


def test_http_call_does_not_duplicate_python_audit(db, tmp_path):
    server = make(db, tmp_path, name='echo', deploy=lambda inputs: inputs)
    async def scenario():
        await server.start()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/v1/server/echo/deploy', json={'inputs': {'text': 'hello'}})
                assert response.status_code == 200
                assert not operations(server)
                assert server.db.get_task(response.json()['task_id'])['request_id'] == response.headers['x-request-id']
        finally:
            await server.close()
    asyncio.run(scenario())


def test_direct_duplex_inputs_and_cancellation_keep_independent_operation_records(db, tmp_path):
    from test_duplex import echo, config
    server = make(db, tmp_path, name='echo', deploy=echo, config=config())
    async def scenario():
        await server.start()
        try:
            identity, result = await server.submit('echo', {})
            await server.send_input(identity, {'text': 'private-frame'}, sequence=1)
            await server.send_input(identity, sequence=2, end=True)
            assert (await result)['outputs'] == {'text': 'private-frame'}
            cancelled, result = await server.submit('echo', {})
            await server.cancel_task(cancelled)
            with pytest.raises(ServerError) as error:
                await result
            assert error.value.code == 'task_cancelled'
            rows = operations(server)
            assert [row['route'] for row in rows] == ['Server.submit', 'Server.send_input',
                'Server.send_input', 'Server.submit', 'Server.cancel_task']
            assert [row['resource_id'] for row in rows] == [identity, identity, identity, cancelled, cancelled]
            assert all(row['state'] == 'complete' for row in rows)
            assert 'private-frame' not in json.dumps(rows)
        finally:
            await server.close()
    asyncio.run(scenario())


def test_startup_audit_failure_prevents_model_initialization(db, tmp_path, monkeypatch):
    initialized = []
    class Model:
        def init(self):
            initialized.append(True)
        def deploy(self, inputs):
            return inputs
    server = make(db, tmp_path, name='echo', handler_factory=Model)
    def fail(*args):
        raise RuntimeError('Unavailable audit database')
    monkeypatch.setattr(server.audit, '_begin', fail)
    async def scenario():
        try:
            with pytest.raises(ServerError) as error:
                await server.start()
            assert error.value.code == 'audit_unavailable' and not initialized
        finally:
            await server.close()
    asyncio.run(scenario())


def test_parallel_python_calls_keep_separate_ids_and_do_not_leak_context(db, tmp_path):
    server = make(db, tmp_path, name='echo', deploy=lambda x: x, workers=2)
    async def scenario():
        await server.start()
        try:
            results = await asyncio.gather(*(server.deploy('echo', {'value': index}) for index in range(8)))
            rows = operations(server)
            assert len(rows) == 8 and len({row['request_id'] for row in rows}) == 8
            assert {row['resource_id'] for row in rows} == {result['task_id'] for result in results}
            assert [result['outputs']['value'] for result in results] == list(range(8))
            assert get_request_id() is None
        finally:
            await server.close()
    asyncio.run(scenario())


def test_detached_http_work_and_another_server_do_not_bypass_python_audit(db, tmp_path):
    first = make(db, tmp_path / 'first', name='echo', deploy=lambda x: x)
    other = make(db, tmp_path / 'other', name='echo', deploy=lambda x: x)
    release = asyncio.Event()
    pending = []
    @first.app.post('/v1/deferred')
    async def deferred():
        async def later():
            await release.wait()
            return await first.deploy('echo', {'text': 'later'})
        pending.append(first.own(later()))
        return {'accepted': True}
    @first.app.post('/v1/other')
    async def another_server():
        return await other.deploy('echo', {'text': 'other'})
    async def scenario():
        await first.start()
        await other.start()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=first.app), base_url='http://test') as client:
                response = await client.post('/v1/deferred')
                assert response.status_code == 200 and not operations(first)
                release.set()
                result = await asyncio.wait_for(pending[0], 5)
                row, = operations(first)
                assert row['resource_id'] == result['task_id']
                assert row['request_id'] != response.headers['x-request-id']
                response = await client.post('/v1/other')
                row, = operations(other)
                assert row['resource_id'] == response.json()['task_id']
                assert row['request_id'] != response.headers['x-request-id']
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)
            await first.close()
            await other.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('kind', ['files', 'tasks'])
def test_explicit_cleanup_requires_audit_before_deleting(db, tmp_path, monkeypatch, kind):
    from test_file_cleanup import upload, expire as expire_file
    from test_retention import create, expire as expire_task
    server = make(db, tmp_path)
    if kind == 'files':
        identity = upload(server.storage)
        expire_file(db, identity)
        exists = lambda: db.get_file(identity)
    else:
        identity = create(db)
        db.finish_task(identity, 'succeeded', db.clock(), outputs={})
        expire_task(db, identity)
        exists = lambda: db.get_task(identity)
    cleanup = getattr(server, 'cleanup_' + kind)
    def fail(*args):
        raise RuntimeError('Audit unavailable')
    async def scenario():
        try:
            with monkeypatch.context() as patch:
                patch.setattr(server.audit, '_begin', fail)
                with pytest.raises(ServerError) as error:
                    await cleanup()
                assert error.value.code == 'audit_unavailable' and exists() is not None
            await cleanup()
            assert exists() is None
            row, = operations(server)
            assert row['route'] == 'Server.cleanup_' + kind and row['state'] == 'complete'
        finally:
            await server.close()
    asyncio.run(scenario())


def test_response_background_action_is_audited_after_http_body_has_been_sent(db, tmp_path):
    from starlette.background import BackgroundTask
    from starlette.responses import JSONResponse
    server = make(db, tmp_path, name='echo', deploy=lambda x: x)
    results = []
    async def later():
        results.append(await server.deploy('echo', {'text': 'background'}))
    @server.app.post('/v1/background')
    async def background():
        return JSONResponse({'accepted': True}, background=BackgroundTask(later))
    async def scenario():
        await server.start()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/v1/background')
            assert response.status_code == 200 and len(results) == 1
            row, = operations(server)
            assert row['resource_id'] == results[0]['task_id']
            assert row['request_id'] != response.headers['x-request-id']
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('ending', ['close', 'disconnect', 'denial'])
def test_actions_after_websocket_end_get_independent_audit(db, tmp_path, ending):
    from nndeploy.server.request import RequestMiddleware
    server = make(db, tmp_path, name='echo', deploy=lambda x: x)
    results, sent = [], []

    async def app(scope, receive, send):
        results.append(await server.deploy('echo', {'text': 'connected'}))
        assert operations(server) == []
        if ending == 'disconnect':
            await receive()
        elif ending == 'close':
            await send({'type': 'websocket.close', 'code': 1000})
        else:
            await send({'type': 'websocket.http.response.start', 'status': 403, 'headers': []})
            await send({'type': 'websocket.http.response.body', 'body': b''})
        results.append(await server.deploy('echo', {'text': 'disconnected'}))

    async def receive():
        return {'type': 'websocket.disconnect', 'code': 1000}

    async def send(message):
        sent.append(message)

    async def scenario():
        await server.start()
        try:
            scope = dict(type='websocket', path='/v1/audit-boundary', root_path='')
            await RequestMiddleware(app, server)(scope, receive, send)
            row, = operations(server)
            assert row['resource_id'] == results[1]['task_id']
            assert row['request_id'] != scope['state']['request_id']
            assert server.db.get_task(results[0]['task_id'])['request_id'] == scope['state']['request_id']
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('transport', ['http', 'websocket'])
def test_custom_route_outside_http_audit_scope_still_audits_python_calls(db, tmp_path, transport):
    from nndeploy.server.request import RequestMiddleware
    server = make(db, tmp_path, name='echo', deploy=lambda x: x)
    results = []

    async def app(scope, receive, send):
        results.append(await server.deploy('echo', {'text': 'custom'}))
        if transport == 'http':
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b''})
        else:
            await send({'type': 'websocket.close', 'code': 1000})

    async def receive():
        raise AssertionError('No receive required')

    async def send(message):
        pass

    async def scenario():
        await server.start()
        try:
            scope = dict(type=transport, path='/custom-deploy', root_path='', method='POST')
            await RequestMiddleware(app, server)(scope, receive, send)
            row, = operations(server)
            assert row['resource_id'] == results[0]['task_id']
            assert row['request_id'] != scope['state']['request_id']
            assert server.db.get_task(results[0]['task_id'])['request_id'] == row['request_id']
        finally:
            await server.close()
    asyncio.run(scenario())

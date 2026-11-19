"""Durable HTTP intent, privacy, authorization, failure and retention checks."""
import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from nndeploy.server import Server
from nndeploy.server.audit import records
from nndeploy.server.dependencies import get_deploy
from nndeploy.server.protocol import DeployRequest
from test_auth import credentials
from test_db import db
from test_redis_queue import redis_service, backend


ADMIN = {'Authorization': 'Bearer audit-admin'}
URL = '/api/server/audit/records'


def get_record(db, request_id):
    with db.engine.connect() as connection:
        return dict(connection.execute(select(records).where(records.c.request_id == request_id)).mappings().one())


def make(db, path, **kwargs):
    return Server(resources=path, data_dir=path, database_url=db.url.render_as_string(hide_password=False),
                  admin_token='audit-admin', audit_config=kwargs.pop('audit_config', {}), **kwargs)


def test_request_identity_is_durable_before_algorithm_and_only_metadata_is_stored(db, tmp_path, credentials):
    auth, headers = credentials
    seen = []

    def echo(values):
        with db.engine.connect() as connection:
            row = connection.execute(select(records).where(records.c.scope == server.audit.scope,
                records.c.state == 'authorized', records.c.subject == 'alice')).mappings().one()
        seen.append(row['request_id'])
        assert row['route'] == '/v1/server/echo/messages'
        return values

    router = APIRouter()
    @router.post('/messages')
    async def messages(deploy=Depends(get_deploy)):
        return await deploy(DeployRequest(inputs={'text': 'private-body-sentinel'}))

    server = make(db, tmp_path, name='echo', deploy=echo, router=router, auth_config=auth)
    with TestClient(server.app) as client:
        request_headers = dict(headers['alice'], **{'X-Request-ID': 'private-incoming-id', 'Idempotency-Key': 'private-idempotency'})
        result = client.post('/v1/server/echo/messages?secret=private-query-sentinel', headers=request_headers)
        assert result.status_code == 200, result.text
        request_id = result.headers['x-request-id']
        record = get_record(db, request_id)
        assert record['state'] == 'complete' and record['status_code'] == 200
        assert record['subject'] == 'alice' and record['authenticated'] and not record['administrator']
        assert record['route'] == '/v1/server/echo/messages'
        assert record['entry'] == 'echo' and record['resource_id'] == result.json()['task_id']
        assert record['duration_seconds'] >= 0 and record['expires_at'] > record['finished_at']
        assert seen == [request_id]
        retry = client.post('/v1/server/echo/messages', headers=request_headers)
        assert retry.json()['task_id'] == result.json()['task_id']
        assert get_record(db, retry.headers['x-request-id'])['resource_id'] == result.json()['task_id']
        assert len(seen) == 1
        denied = client.post('/v1/server/echo/deploy', json={'inputs': {}}, headers=headers['reader'])
        assert denied.status_code == 403
        assert get_record(db, denied.headers['x-request-id'])['subject'] == 'reader'
        anonymous = client.get('/v1/models')
        assert anonymous.status_code == 401
        assert get_record(db, anonymous.headers['x-request-id'])['subject'] is None
        assert client.get(URL, headers=headers['alice']).status_code == 401
        queried = client.get(URL, params={'request_id': request_id}, headers=ADMIN)
        assert queried.status_code == 200 and queried.headers['cache-control'] == 'no-store'
        assert queried.json()['items'][0]['request_id'] == request_id
        with db.engine.connect() as connection:
            saved = json.dumps([dict(row) for row in connection.execute(select(records).where(records.c.scope == server.audit.scope)).mappings()])
        for secret in ('private-body-sentinel', 'private-incoming-id', 'private-idempotency', 'private-query-sentinel',
                       headers['alice']['Authorization'].split()[1], 'audit-admin'):
            assert secret not in saved
        live = client.get('/health/live')
        with db.engine.connect() as connection:
            assert connection.execute(select(records.c.id).where(records.c.request_id == live.headers['x-request-id'])).first() is None


@pytest.mark.parametrize('phase', ['_begin', '_identify'])
def test_audit_failure_before_action_rejects_without_executing(db, tmp_path, credentials, monkeypatch, phase):
    auth, headers = credentials
    calls = []
    server = make(db, tmp_path, name='echo', deploy=lambda x: calls.append(x) or x, auth_config=auth)
    def fail(*args):
        raise RuntimeError('private-database-error')
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.audit, phase, fail)
        response = client.post('/v1/server/echo/deploy', json={'inputs': {}}, headers=headers['alice'])
        assert response.status_code == 503 and response.json()['error']['code'] == 'audit_unavailable'
        assert 'private-database-error' not in response.text and not calls
        assert client.get('/health/live').status_code == 200
        if phase == '_identify':
            row = get_record(db, response.headers['x-request-id'])
            assert row['status_code'] == 503 and row['resource_id'] is None


def test_audit_completion_failure_preserves_intent_and_does_not_rewrite_business_result(db, tmp_path, credentials, monkeypatch):
    auth, headers = credentials
    server = make(db, tmp_path, name='echo', deploy=lambda x: x, auth_config=auth)
    finish = server.audit._finish
    def fail(request_id, context, *args):
        if context['route'] == '/v1/server/{name}/deploy':
            raise RuntimeError('private-final-write-error')
        return finish(request_id, context, *args)
    with TestClient(server.app) as client:
        monkeypatch.setattr(server.audit, '_finish', fail)
        response = client.post('/v1/server/echo/deploy', json={'inputs': {'text': 'done'}}, headers=headers['alice'])
        assert response.status_code == 200 and response.json()['outputs']['text'] == 'done'
        request_id = response.headers['x-request-id']
        row = get_record(db, request_id)
        assert row['state'] == 'authorized' and row['status_code'] is None and row['expires_at'] is None
        assert row['subject'] == 'alice'
        assert db.get_task(response.json()['task_id'])['request_id'] == request_id
        assert 'nndeploy_audit_errors_total{operation="fail"} 1.0' in client.get('/metrics', headers=ADMIN).text
    restored = make(db, tmp_path)
    with TestClient(restored.app) as client:
        found = client.get(URL, headers=ADMIN, params={'request_id': request_id}).json()['items']
        assert len(found) == 1 and found[0]['state'] == 'authorized'


def test_audit_admission_wait_and_queue_are_bounded_and_close_joins_sql(db, tmp_path, credentials, monkeypatch):
    auth, headers = credentials
    server = make(db, tmp_path, auth_config=auth, audit_config={'max_pending': 1, 'wait_seconds': .1})
    entered, release = threading.Event(), threading.Event()
    begin = server.audit._begin
    def blocked(*args):
        entered.set()
        assert release.wait(timeout=10)
        begin(*args)
    monkeypatch.setattr(server.audit, '_begin', blocked)

    async def scenario():
        closing = None
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                first = await client.get('/v1/models', headers=headers['alice'])
                assert entered.is_set() and first.status_code == 503
                assert len(server.audit.pending) == 1
                second = await client.get('/v1/models', headers=headers['alice'])
                assert second.status_code == 503 and second.json()['error']['code'] == 'audit_busy'
                assert (await client.get('/health/live')).status_code == 200
                closing = asyncio.create_task(server.close())
                await asyncio.sleep(.05)
                assert not closing.done()
                release.set()
                await asyncio.wait_for(closing, 5)
                row = get_record(db, first.headers['x-request-id'])
                assert row['state'] == 'started' and row['resource_id'] is None
                assert not server.audit.pending and server.audit.closed
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_audit_pages_and_retention_preserve_unconfirmed_and_other_workspace(db, tmp_path):
    first, other = make(db, tmp_path / 'a'), make(db, tmp_path / 'b')
    async def scenario():
        try:
            for index in range(4):
                identity = 'req_' + f'{index:032x}'
                first.audit._begin(identity, 'GET')
                if index != 0:
                    first.audit._finish(identity, {}, 200, True, .1)
            other.audit._begin('req_' + 'f' * 32, 'POST')
            page = await first.audit.read(2)
            assert len(page['items']) == 2 and page['next_before'] is not None
            rest = await first.audit.read(2, page['next_before'])
            assert len(rest['items']) == 2 and rest['next_before'] is None
            assert not set(x['id'] for x in page['items']) & set(x['id'] for x in rest['items'])
            with db.engine.begin() as connection:
                connection.execute(update(records).where(records.c.scope == first.audit.scope, records.c.state == 'complete').values(expires_at=0))
            assert await first.audit.cleanup() == 3
            kept = (await first.audit.read())['items']
            assert len(kept) == 1 and kept[0]['state'] == 'started'
            assert len((await other.audit.read())['items']) == 1
        finally:
            await first.close()
            await other.close()
    asyncio.run(scenario())


def test_editor_session_and_forbidden_key_are_audited_without_cookie_or_token(tmp_path, monkeypatch, credentials):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    auth, headers = credentials
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
                    admin_token='audit-admin', auth_config=auth, audit_config={})
    with TestClient(server.app) as client:
        denied = client.get('/api/resources', headers=headers['alice'])
        assert denied.status_code == 403
        row = get_record(server.db, denied.headers['x-request-id'])
        assert row['subject'] == 'alice' and not row['administrator']
        assert row['route'] == '/api/resources'
        login = client.post('/api/auth/session', headers=ADMIN)
        assert login.status_code == 200
        assert get_record(server.db, login.headers['x-request-id'])['administrator']
        read = client.get('/api/resources')
        assert read.status_code == 200
        row = get_record(server.db, read.headers['x-request-id'])
        assert row['administrator'] and row['subject'] == '@administrator'
        logout = client.delete('/api/auth/session', headers={'Origin': 'http://testserver'})
        assert logout.status_code == 200
        assert get_record(server.db, logout.headers['x-request-id'])['administrator']
        with server.db.engine.connect() as connection:
            saved = json.dumps([dict(row) for row in connection.execute(select(records)).mappings()])
        assert login.headers['set-cookie'].split('=', 1)[1].split(';')[0] not in saved
        assert 'audit-admin' not in saved


def test_shared_replicas_query_same_audit_scope_with_different_workspaces(backend, tmp_path, credentials):
    from test_shared_server import shared_deploy, entry_config
    database, redis = backend
    auth, headers = credentials
    common = tmp_path / 'files'
    common.mkdir()
    roots = [tmp_path / str(i) for i in range(2)]
    for root in roots:
        root.mkdir()
        (root / 'files').symlink_to(common, target_is_directory=True)
    servers = [make(database, root, name='echo', deploy=shared_deploy, config=entry_config(),
                    auth_config=auth, redis_config=redis) for root in roots]
    async def scenario():
        try:
            for server in servers:
                await server.start()
            assert servers[0].config_scope != servers[1].config_scope
            assert servers[0].audit.scope == servers[1].audit.scope
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as a, \
                       httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as b:
                response = await a.post('/v1/server/echo/deploy', json={'inputs': {}}, headers=headers['alice'])
                assert response.status_code == 200, response.text
                queried = await b.get(URL, headers=ADMIN, params={'request_id': response.headers['x-request-id']})
                found = queried.json()['items']
                assert len(found) == 1 and found[0]['resource_id'] == response.json()['task_id']
                assert found[0]['subject'] == 'alice' and found[0]['state'] == 'complete'
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_stream_audit_describes_http_outcome_separately_from_failed_task(db, tmp_path, credentials):
    auth, headers = credentials
    def stream(values):
        yield {'text': 'private-token-sentinel'}
        raise RuntimeError('private-model-exception')
    server = make(db, tmp_path, name='echo', deploy=stream, auth_config=auth, config={'streaming': {'enabled': True}})
    with TestClient(server.app) as client:
        response = client.post('/v1/server/echo/deploy', json={'inputs': {}, 'mode': 'stream'},
                               headers=dict(headers['alice'], Accept='text/event-stream'))
        assert response.status_code == 200
        row = get_record(db, response.headers['x-request-id'])
        assert row['state'] == 'complete' and row['status_code'] == 200
        assert db.get_task(row['resource_id'])['state'] == 'failed'
        assert 'private-token-sentinel' not in json.dumps(row) and 'private-model-exception' not in json.dumps(row)


def test_cancelled_http_request_is_audited_without_cancelling_accepted_task(db, tmp_path, credentials):
    from test_shared_server import until
    auth, headers = credentials
    entered, release = threading.Event(), threading.Event()
    def echo(values):
        entered.set()
        assert release.wait(timeout=10)
        return values
    server = make(db, tmp_path, name='echo', deploy=echo, auth_config=auth)
    async def scenario():
        try:
            await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                request = asyncio.create_task(client.post('/v1/server/echo/deploy', json={'inputs': {}}, headers=headers['alice']))
                await until(entered.is_set)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                with db.engine.connect() as connection:
                    row = dict(connection.execute(select(records).where(records.c.scope == server.audit.scope,
                        records.c.method == 'POST', records.c.route == '/v1/server/{name}/deploy')).mappings().one())
                assert row['state'] == 'interrupted' and row['status_code'] == 499
                assert row['subject'] == 'alice'
                from nndeploy.server.db import tasks
                with db.engine.connect() as connection:
                    identity = connection.execute(select(tasks.c.task_id).where(tasks.c.request_id == row['request_id'])).scalar_one()
                release.set()
                await until(lambda: db.get_task(identity)['state'] == 'succeeded')
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_cli_enables_audit_with_existing_database_and_native_workflow(tmp_path, monkeypatch):
    import subprocess
    import sys
    import requests
    from test_cli import unused_port
    from test_server_api import text_graph
    monkeypatch.setenv('TEST_AUDIT_ADMIN', 'cli-audit-admin')
    workflow, config = tmp_path / 'echo.json', tmp_path / 'audit.json'
    workflow.write_text(json.dumps(text_graph()))
    config.write_text(json.dumps({'retention_seconds': 120}))
    port = unused_port()
    address = f'http://127.0.0.1:{port}'
    log_path = tmp_path / 'server.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'nndeploy.server', '--api-only', '--workflow', str(workflow),
            '--resources', str(tmp_path / 'resources'), '--port', str(port), '--audit-config', str(config),
            '--admin-token-env', 'TEST_AUDIT_ADMIN'], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while True:
                assert process.poll() is None, log_path.read_text()
                try:
                    if requests.get(address + '/health/ready', timeout=.2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, log_path.read_text()
                time.sleep(.05)
            result = requests.post(address + '/v1/server/echo/deploy', json={'inputs': {'source': 'native'}}, timeout=5)
            assert result.status_code == 200, result.text
            deadline = time.monotonic() + 5
            while True:
                result_log = requests.get(address + URL, params={'request_id': result.headers['x-request-id']},
                    headers={'Authorization': 'Bearer cli-audit-admin'}, timeout=5)
                assert result_log.status_code == 200, result_log.text
                row = result_log.json()['items'][0]
                if row['state'] == 'complete':
                    break
                assert time.monotonic() < deadline
                time.sleep(.02)
            assert row['resource_id'] == result.json()['task_id'] and row['subject'] is None
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

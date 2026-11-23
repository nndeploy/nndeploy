"""Identity quotas through HTTP and atomic admission against real Redis."""
import asyncio
import uuid

import httpx
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nndeploy.server import Server
from nndeploy.server.dependencies import Identity, current_identity, limited_identity, get_deploy
from nndeploy.server.errors import ServerError
from nndeploy.server.protocol import DeployRequest
from nndeploy.server.rate_limit import RequestLimiter, RequestRate
from nndeploy.server.redis import RedisConfig
from test_auth import credentials
from test_redis_queue import redis_service, backend


def test_http_routes_share_subject_budget_and_rejections_do_not_deploy(tmp_path, credentials, monkeypatch):
    config, headers = credentials
    config['request_rate'] = {'requests': 2, 'window_seconds': 60}
    monkeypatch.setenv('TEST_SECOND_ALICE', 's' * 32)
    config['keys'].append(dict(config['keys'][0], token_env='TEST_SECOND_ALICE'))
    clock = [100.0]
    monkeypatch.setattr('nndeploy.server.rate_limit.monotonic', lambda: clock[0])
    calls = []
    router = APIRouter()

    @router.post('/custom')
    async def custom(deploy=Depends(get_deploy), caller=Depends(limited_identity)):
        return await deploy(DeployRequest(inputs={'text': caller.subject}))

    server = Server(name='echo', deploy=lambda x: calls.append(x) or x, router=router,
                    auth_config=config, admin_token='admin', resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        assert client.get('/v1/models').status_code == 401
        assert client.get('/v1/models', headers={'Authorization': 'Bearer invalid'}).status_code == 401
        assert not server.request_limiter.buckets
        # Both mounted authorization and the route dependency use the same
        # cached admission; get_deploy must not debit a second time either.
        assert client.post('/v1/server/echo/custom', headers=headers['alice']).status_code == 200
        assert client.get('/v1/models', headers={'Authorization': 'Bearer ' + 's' * 32}).status_code == 200
        requests = [
            ('GET', '/v1/models', {}),
            ('GET', '/v1/server/echo/schema', {}),
            ('POST', '/v1/server/echo/custom', {}),
            ('POST', '/v1/server/echo/deploy', {'json': {'inputs': {}}}),
            ('POST', '/v1/server/echo/deploy/batch', {'json': {'items': [{'inputs': {}}]}}),
            ('GET', '/v1/tasks/missing', {}),
            ('GET', '/v1/tasks/missing/events', {}),
            ('POST', '/v1/tasks/missing/cancel', {}),
            ('POST', '/v1/files', {'files': {'file': ('x.txt', b'test', 'text/plain')}}),
            ('GET', '/v1/files/missing', {}),
            ('GET', '/v1/files/missing/content', {}),
            ('DELETE', '/v1/files/missing', {}),
            ('POST', '/v1/files/missing/download-url', {}),
        ]
        for method, url, kwargs in requests:
            response = client.request(method, url, headers=headers['alice'], **kwargs)
            assert response.status_code == 429, (url, response.text)
            assert response.headers['Retry-After'] == '30'
            assert response.json()['error']['code'] == 'request_rate_exceeded'
        assert calls == [{'text': 'alice'}]
        assert client.get('/v1/models', headers=headers['bob']).status_code == 200
        assert client.get('/health/live').status_code == 200
        assert client.get('/health/ready').status_code == 200
        admin = {'Authorization': 'Bearer admin'}
        assert client.get('/v1/models', headers=admin).status_code == 200
        assert client.get('/v1/models', headers=admin).status_code == 200
        assert client.get('/v1/models', headers=admin).status_code == 429
        assert client.get('/api/server', headers=admin).status_code == 200
        clock[0] += 30
        assert client.get('/v1/models', headers=headers['alice']).status_code == 200
        assert client.get('/v1/models', headers=headers['alice']).status_code == 429
    assert server.request_limiter.closed
    assert not server.request_limiter.buckets


def test_custom_identity_override_keeps_rate_and_entry_permissions(tmp_path, credentials):
    config, _ = credentials
    config['request_rate'] = {'requests': 2}
    router = APIRouter()

    @router.get('/custom')
    def custom():
        return {'ok': True}

    server = Server(name='echo', deploy=lambda x: x, router=router, auth_config=config, resources=tmp_path, data_dir=tmp_path)
    server.app.dependency_overrides[current_identity] = lambda: Identity('external', frozenset(), frozenset())
    with TestClient(server.app) as client:
        # Valid external identity, but no deploy permission. Forbidden attempts
        # still consume quota once authentication succeeds.
        assert client.get('/v1/server/echo/custom').status_code == 403
        assert client.get('/v1/models').status_code == 200
        assert client.get('/v1/models').status_code == 429


@pytest.mark.parametrize('value', [
    {'requests': 0}, {'requests': True}, {'requests': 1_000_001},
    {'requests': 1, 'window_seconds': float('nan')},
    {'requests': 1, 'window_seconds': float('inf')},
    {'requests': 1, 'window_seconds': 0}, {'requests': 1, 'window_seconds': 86401},
    {'requests': 1, 'window_seconds': '60'}, {'requests': 1, 'unknown': 1},
])
def test_invalid_rate_policy_is_rejected(value):
    with pytest.raises(ValidationError):
        RequestRate.model_validate(value)


def test_local_buckets_refill_expire_and_never_evict_live_identity(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('nndeploy.server.rate_limit.monotonic', lambda: clock[0])

    async def scenario():
        limiter = RequestLimiter(RequestRate(requests=1, window_seconds=2))
        limiter.max_subjects = 2
        await limiter.take('alice')
        await limiter.take('bob')
        with pytest.raises(ServerError, match='capacity') as failure:
            await limiter.take('charlie')
        assert failure.value.status_code == 503
        with pytest.raises(ServerError) as failure:
            await limiter.take('alice')
        assert failure.value.status_code == 429
        clock[0] += 1
        with pytest.raises(ServerError) as failure:
            await limiter.take('alice')
        assert failure.value.retry_after == 1
        clock[0] += 1
        await limiter.take('charlie')  # bob has expired, alice is retained
        await limiter.take('alice')
        await limiter.close()
        with pytest.raises(ServerError) as failure:
            await limiter.take('alice')
        assert failure.value.status_code == 503
        disabled = RequestLimiter(None)
        for _ in range(100):
            await disabled.take('alice')
        assert not disabled.buckets and disabled.client is None
        await disabled.close()
    asyncio.run(scenario())


def redis_config(monkeypatch, service, namespace=None):
    monkeypatch.setenv('TEST_RATE_REDIS', service.url)
    return RedisConfig(url_env='TEST_RATE_REDIS', namespace=namespace or 'rate-' + uuid.uuid4().hex,
                       socket_timeout_seconds=.2)


def test_redis_atomic_across_replicas_and_policy_mismatch(redis_service, monkeypatch):
    config = redis_config(monkeypatch, redis_service)

    async def scenario():
        a = RequestLimiter(RequestRate(requests=7, window_seconds=86400), config)
        b = RequestLimiter(a.config, config)
        changed = RequestLimiter(RequestRate(requests=8, window_seconds=86400), config)
        isolated = RequestLimiter(a.config, config.model_copy(update={'namespace': config.namespace + '-other'}))
        try:
            results = await asyncio.gather(*( (a if i % 2 else b).take('alice') for i in range(60)), return_exceptions=True)
            assert sum(x is None for x in results) == 7
            assert all(x is None or isinstance(x, ServerError) and x.status_code == 429 for x in results)
            await b.take('bob')
            await isolated.take('alice')
            with pytest.raises(ServerError) as failure:
                await changed.take('alice')
            assert failure.value.code == 'rate_limit_config_mismatch'
            await a.close()
            # A replacement replica must not get a fresh local allowance.
            replacement = RequestLimiter(b.config, config)
            try:
                with pytest.raises(ServerError) as failure:
                    await replacement.take('alice')
                assert failure.value.status_code == 429
            finally:
                await replacement.close()
        finally:
            await asyncio.gather(*(x.close() for x in (a, b, changed, isolated)))
    asyncio.run(scenario())


def test_redis_refill_and_idle_expiry(redis_service, monkeypatch):
    config = redis_config(monkeypatch, redis_service)

    async def scenario():
        limiter = RequestLimiter(RequestRate(requests=2, window_seconds=1), config)
        try:
            await limiter.take('alice')
            await limiter.take('alice')
            with pytest.raises(ServerError) as failure:
                await limiter.take('alice')
            assert failure.value.status_code == 429
            keys = await limiter.client.keys(config.namespace + ':request-rate:*')
            assert len(keys) == 1 and 'alice' not in keys[0]
            assert 0 < await limiter.client.pttl(keys[0]) <= 1000
            await asyncio.sleep(.55)
            # One token refills before the key expires; the burst does not
            # reset to full at an arbitrary fixed-window boundary.
            assert await limiter.client.exists(keys[0]) == 1
            await limiter.take('alice')
            with pytest.raises(ServerError):
                await limiter.take('alice')
            await asyncio.sleep(1.05)
            assert await limiter.client.exists(keys[0]) == 0
            await limiter.take('alice')
            await limiter.take('alice')
        finally:
            await limiter.close()
    asyncio.run(scenario())


def test_http_shared_rate_outage_rejects_without_local_fallback(redis_service, monkeypatch, tmp_path, credentials):
    redis = redis_config(monkeypatch, redis_service)
    config, headers = credentials
    config['request_rate'] = {'requests': 1, 'window_seconds': 86400}

    async def scenario():
        # HTTP quota admission needs no loaded model. Omit application startup
        # here so queue recovery cannot mask the limiter's own Redis outage.
        servers = [Server(resources=tmp_path / str(i), data_dir=tmp_path / str(i), auth_config=config,
                          redis_config=redis.model_dump()) for i in range(2)]
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as a, \
                       httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as b:
                assert (await a.get('/v1/models', headers=headers['alice'])).status_code == 200
                assert (await b.get('/v1/models', headers=headers['alice'])).status_code == 429
                redis_service.stop()
                for client in (a, b):
                    response = await client.get('/v1/models', headers=headers['bob'])
                    assert response.status_code == 503, response.text
                    assert response.json()['error']['code'] == 'rate_limit_unavailable'
                    assert '127.0.0.1' not in response.text
                    assert (await client.get('/health/live')).status_code == 200
                    assert (await client.get('/v1/models')).status_code == 401
                assert all(not s.request_limiter.buckets for s in servers)
                redis_service.start()
                # This fixture intentionally has persistence off: restarting
                # Redis resets quotas, unlike merely restarting an API replica.
                assert (await b.get('/v1/models', headers=headers['bob'])).status_code == 200
        finally:
            if redis_service.process.poll() is not None:
                redis_service.start()
            await asyncio.gather(*(s.close() for s in servers))
    asyncio.run(scenario())


def test_shared_deploy_replay_consumes_request_quota_not_another_task(backend, tmp_path, credentials):
    from sqlalchemy import select, func
    from nndeploy.server.db import tasks
    from test_shared_server import shared_deploy, entry_config

    db, redis = backend
    auth, headers = credentials
    auth['request_rate'] = {'requests': 2, 'window_seconds': 86400}
    servers = [Server(name='echo', deploy=shared_deploy, config=entry_config(),
                      resources=tmp_path / 'shared', data_dir=tmp_path / 'shared', database_url=db.url.render_as_string(hide_password=False),
                      redis_config=redis, auth_config=auth) for _ in range(2)]

    async def scenario():
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as a, \
                       httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as b:
                request_headers = dict(headers['alice'], **{'Idempotency-Key': uuid.uuid4().hex})
                url = '/v1/server/echo/deploy'
                payload = {'inputs': {'text': 'one computation'}}
                first = await a.post(url, json=payload, headers=request_headers)
                assert first.status_code == 200, first.text
                replay = await b.post(url, json=payload, headers=request_headers)
                assert replay.status_code == 200, replay.text
                identity = first.json()['task_id']
                assert replay.json()['task_id'] == identity
                rejected = await b.post(url, json={'inputs': {'text': 'must not run'}}, headers=headers['alice'])
                assert rejected.status_code == 429, rejected.text
                with db.engine.connect() as connection:
                    # DB fixtures can be reused by the integration runner;
                    # restrict the count to these entries' actual queue key.
                    count = connection.execute(select(func.count()).select_from(tasks).where(
                        tasks.c.queue_key == db.get_task(identity)['queue_key'])).scalar_one()
                assert count == 1
                assert (await b.post(url, json=payload, headers=headers['bob'])).status_code == 200
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())

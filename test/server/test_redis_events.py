"""Redis sequence hints accelerate readers without replacing durable events."""
import asyncio
import uuid

import httpx
import pytest

from nndeploy.server import Server
from nndeploy.server.redis import RedisConfig, RedisEvents
from test_redis_queue import backend, redis_service
from test_streaming import output_stream, parse_events, stream_config


def test_sequence_cache_is_bounded_shared_and_database_scoped(backend):
    db, settings = backend
    config = RedisConfig.model_validate(settings)
    first = RedisEvents(config, db.database_identity())
    second = RedisEvents(config, db.database_identity())
    foreign = RedisEvents(config, 'other-database')
    identity = 'task_' + uuid.uuid4().hex
    async def scenario():
        try:
            await asyncio.to_thread(first.notify, identity, 2)
            # The notification preceded this wait; the cached sequence closes
            # that race without retaining or copying event payloads to Redis.
            assert await second.wait(identity, 1)
            await asyncio.to_thread(first.notify, identity, 3)
            key = first.prefix + identity
            assert await second.reader.xrange(key) == [('3-0', {'sequence': '3'})]
            assert 0 < await second.reader.ttl(key) <= 60
            assert not await foreign.wait(identity, 1)
            await second.reader.delete(key)
            assert not await second.wait(identity, 3)
        finally:
            await first.close()
            await second.close()
            await foreign.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('unavailable', [False, True])
def test_http_stream_and_replay_use_db_when_notifications_fail(backend, tmp_path, monkeypatch, unavailable):
    db, config = backend
    servers = [Server(name='echo', deploy=output_stream, config=stream_config(), data_dir=tmp_path / 'shared',
                      database_url=db.url.render_as_string(hide_password=False), redis_config=config) for _ in range(2)]
    if unavailable:
        def fail(*args, **kwargs):
            raise OSError('notification connection unavailable')
        async def fail_async(*args, **kwargs):
            raise OSError('notification connection unavailable')
        for server in servers:
            monkeypatch.setattr(server.event_notifications.writer, 'pipeline', fail)
            monkeypatch.setattr(server.event_notifications.reader, 'xread', fail_async)
    async def scenario():
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as client:
                response = await client.post('/v1/server/echo/deploy', json={'mode': 'stream'})
                assert response.status_code == 200, response.text
                identity = response.headers['x-task-id']
                events = parse_events(response.text)
                assert [event['sequence'] for event in events] == [1, 2, 3, 4, 5]
                assert events[-1]['kind'] == 'done'
                assert db.get_task(identity)['state'] == 'succeeded'
            signals = servers[1].event_notifications
            if unavailable:
                assert not await signals.wait(identity, 0)
            else:
                assert await signals.reader.xrange(signals.prefix + identity) == [('3-0', {'sequence': '3'})]
            # Loss/expiry of the entire cache must not affect replay or history.
            await signals.reader.delete(signals.prefix + identity)
            await servers[0].close()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as client:
                response = await client.get(f'/v1/tasks/{identity}/events', headers={'Last-Event-ID': '2'})
                assert response.status_code == 200, response.text
                assert [event['sequence'] for event in parse_events(response.text)] == [3, 4, 5]
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())

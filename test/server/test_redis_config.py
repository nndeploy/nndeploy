"""Redis key configuration retains namespace and database isolation."""
import asyncio

import pytest
from pydantic import ValidationError

from nndeploy.server.redis import RedisConfig, RedisEvents


def test_colon_namespace_is_supported():
    assert RedisConfig(namespace='agentx:dev').namespace == 'agentx:dev'


@pytest.mark.parametrize('namespace', ['bad*', 'bad?', 'bad[0]', 'bad{tag}', 'bad\nkey', ''])
def test_namespace_cannot_inject_key_patterns_or_hash_tags(namespace):
    with pytest.raises(ValidationError):
        RedisConfig(namespace=namespace)


def test_event_prefix_keeps_namespace_and_database_isolation(monkeypatch):
    monkeypatch.setenv('NNDEPLOY_REDIS_URL', 'redis://127.0.0.1:1/0')
    events = []
    try:
        for namespace, database in [('first', 'db-a'), ('second', 'db-a'), ('first', 'db-b')]:
            config = RedisConfig(namespace=namespace, event_stream_prefix='agentx_dev:run_stream:')
            events.append(RedisEvents(config, database))
        assert len({item.prefix for item in events}) == 3
        assert all(item.prefix.startswith('agentx_dev:run_stream:') for item in events)
    finally:
        async def close():
            for item in events:
                await item.close()
        asyncio.run(close())


@pytest.mark.parametrize('prefix', ['', 'x*', 'x?', 'x[1]', '{slot}', 'x\n'])
def test_event_prefix_rejects_unsafe_key_characters(prefix):
    with pytest.raises(ValidationError):
        RedisConfig(event_stream_prefix=prefix)

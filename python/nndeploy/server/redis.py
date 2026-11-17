"""Optional Redis connections for Server queues and coordination."""
import os
import asyncio
import time

from pydantic import Field
from .config import StrictModel


class RedisConfig(StrictModel):
    url_env: str = 'NNDEPLOY_REDIS_URL'
    namespace: str = Field(default='nndeploy', pattern=r'^[A-Za-z0-9_.:-]{1,64}$')
    event_stream_prefix: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_.:-]{1,128}$')
    socket_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    lease_seconds: float = Field(default=30.0, ge=1, le=3600)
    max_inflight_tasks: int = Field(default=64, ge=1)

    def connect(self):
        import redis.asyncio as redis
        url = os.environ.get(self.url_env)
        if not url:
            raise ValueError(f'Missing Redis URL environment variable: {self.url_env}')
        return redis.Redis.from_url(url, decode_responses=True,
            socket_connect_timeout=self.socket_timeout_seconds,
            socket_timeout=self.socket_timeout_seconds, health_check_interval=10)


class RedisEvents:
    """Short-lived sequence cache and wakeups; event content stays in the DB.

    A one-item Stream avoids subscription races and bounds retained data per
    active task. Missing/expired signals and Redis failures use ordinary DB
    polling, so notifications never determine task state or ownership.
    """
    def __init__(self, config, database_identity):
        import redis
        self.prefix = f'{config.event_stream_prefix or ""}{config.namespace}:db:{database_identity}:events:'
        self.reader = config.connect()
        self.writer = redis.Redis.from_url(os.environ[config.url_env], decode_responses=True,
            socket_connect_timeout=config.socket_timeout_seconds, socket_timeout=config.socket_timeout_seconds)
        self.closed = False
        self.retry_after = 0.0

    def notify(self, task_id, sequence):
        # Called on the existing event-writer thread only after DB commit.
        # Optional notification failures must not fail successful computation.
        if self.closed or time.monotonic() < self.retry_after:
            return
        try:
            key = self.prefix + task_id
            with self.writer.pipeline(transaction=True) as pipeline:
                pipeline.xadd(key, {'sequence': sequence}, id=f'{sequence}-0', maxlen=1, approximate=False)
                pipeline.expire(key, 60)
                pipeline.execute()
        except Exception:
            self.retry_after = time.monotonic() + 1

    async def wait(self, task_id, after):
        if not self.closed:
            try:
                # DB event sequence is the Stream ID: an event arriving before
                # the reader starts waiting remains observable in the cache.
                return bool(await self.reader.xread({self.prefix + task_id: f'{after}-0'}, count=1, block=100))
            except Exception:
                pass
        await asyncio.sleep(.1)
        return False

    async def close(self):
        self.closed = True
        self.writer.close()
        await self.reader.aclose()


def source_identity(source):
    """Fingerprint code or the original workflow before host-specific asset copies."""
    import hashlib
    import inspect
    from .db import encode
    if source.get('graph') is not None:
        raise ValueError('Shared queues require a workflow or an importable factory, not a direct Graph')
    if source.get('workflow') is not None:
        value = source['workflow']
    else:
        target = next(source[key] for key in ('graph_factory', 'handler_factory', 'deploy') if source.get(key) is not None)
        name = getattr(target, '__qualname__', '')
        module = getattr(target, '__module__', '')
        if not name or '<locals>' in name or '<lambda>' in name or not module:
            raise ValueError('Shared queues require importable functions or factories')
        try:
            code = inspect.getsource(target)
        except (TypeError, OSError) as exc:
            raise ValueError('Shared code must expose inspectable source for worker matching') from exc
        value = [module, name, code]
    return hashlib.sha256(encode(value).encode()).hexdigest()


def entry_fingerprint(source, config):
    import hashlib
    from .db import encode
    schemas = [source[key].model_json_schema() if source.get(key) is not None else None for key in ('input_model', 'output_model')]
    value = config.model_dump()
    # HTTP wait policy changes neither accepted tasks nor model execution.
    # Excluding it also preserves fingerprints of configurations saved before it existed.
    value['execution'].pop('http_wait_timeout_seconds', None)
    return hashlib.sha256(encode([source['identity'], value, schemas]).encode()).hexdigest()

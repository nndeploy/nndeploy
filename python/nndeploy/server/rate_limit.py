"""HTTP request quotas shared by identity, independently of execution slots."""
from collections import OrderedDict
import hashlib
import math
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field

from .errors import ServerError


class RequestRate(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    requests: int = Field(ge=1, le=1_000_000)
    window_seconds: float = Field(default=60.0, ge=0.001, le=86400)


# Redis time, refill, debit and expiry are one atomic operation. Store the
# policy too: replicas with inconsistent policies must not reset each other's
# counters. No API keys or request content enter Redis.
_TAKE = """
local capacity = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local state = redis.call('HMGET', KEYS[1], 'tokens', 'time', 'capacity', 'window')
local tokens = capacity
if state[1] then
    if tonumber(state[3]) ~= capacity or tonumber(state[4]) ~= window then
        return {-1, 0}
    end
    now = math.max(now, tonumber(state[2]))
    tokens = math.min(capacity, tonumber(state[1]) + (now - tonumber(state[2])) * capacity / window)
end
local accepted = 0
local retry = 0
if tokens >= 1 then
    tokens = tokens - 1
    accepted = 1
else
    retry = math.ceil((1 - tokens) * window / capacity * 1000)
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'time', tostring(now),
           'capacity', ARGV[1], 'window', ARGV[2])
redis.call('PEXPIRE', KEYS[1], math.ceil(window * 1000))
return {accepted, retry}
"""


class RequestLimiter:
    # Bound memory for applications supplying dynamic Identity dependencies.
    # Never evict a live bucket to admit a fresh identity with a full budget.
    max_subjects = 10_000

    def __init__(self, config, redis_config=None):
        self.config = config
        self.redis_config = redis_config
        self.client = None
        self.buckets = OrderedDict()
        self.closed = False

    async def take(self, subject):
        if self.config is None:
            return
        if self.closed:
            raise ServerError(503, 'rate_limit_unavailable', 'Request quota is unavailable', retryable=True)
        if self.redis_config is not None:
            try:
                if self.client is None:
                    self.client = self.redis_config.connect()
                # Null and the literal string "null" must not share a bucket.
                digest = hashlib.sha256(('anonymous' if subject is None else 'subject:' + subject).encode()).hexdigest()
                key = f'{self.redis_config.namespace}:request-rate:{digest}'
                accepted, delay_ms = await self.client.eval(
                    _TAKE, 1, key, self.config.requests, self.config.window_seconds)
            except Exception:
                # Redis URLs and credentials must not enter public errors.
                raise ServerError(503, 'rate_limit_unavailable', 'Request quota is unavailable', retryable=True) from None
            if accepted == -1:
                raise ServerError(503, 'rate_limit_config_mismatch',
                                  'Server replicas have inconsistent request quota settings')
            delay = delay_ms / 1000
        else:
            now = monotonic()
            window = self.config.window_seconds
            # Last-touch order is expiry order because all buckets use one
            # window. Bound cleanup per request, including expired identities.
            for _ in range(128):
                if not self.buckets or next(iter(self.buckets.values()))[1] + window > now:
                    break
                self.buckets.popitem(last=False)
            previous = self.buckets.get(subject)
            if previous is None and len(self.buckets) >= self.max_subjects:
                raise ServerError(503, 'rate_limit_capacity', 'Request quota identity capacity is full', retryable=True)
            tokens = self.config.requests if previous is None else min(
                self.config.requests, previous[0] + (now - previous[1]) * self.config.requests / window)
            accepted = tokens >= 1
            delay = 0 if accepted else (1 - tokens) * window / self.config.requests
            self.buckets[subject] = (tokens - 1 if accepted else tokens, now)
            self.buckets.move_to_end(subject)
        if not accepted:
            raise ServerError(429, 'request_rate_exceeded', 'Request rate limit exceeded',
                              retryable=True, retry_after=max(1, math.ceil(delay)))

    async def close(self):
        self.closed = True
        self.buckets.clear()
        if self.client is not None:
            await self.client.aclose()

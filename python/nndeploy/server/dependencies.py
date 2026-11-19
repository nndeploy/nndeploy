"""Common application dependencies and explicit caller permissions."""
from dataclasses import dataclass
import hashlib
import os
import secrets
from typing import Literal

from fastapi import Depends, Request
from starlette.requests import HTTPConnection
from starlette.exceptions import WebSocketException
from pydantic import BaseModel, ConfigDict, Field

from .errors import ServerError
from .rate_limit import RequestRate


class APIKey(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    subject: str = Field(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$')
    token_env: str = Field(min_length=1)
    entries: list[str] = Field(default_factory=list)
    permissions: list[Literal['deploy', 'tasks:read', 'tasks:cancel', 'files:read', 'files:write']] = Field(default_factory=list)


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    keys: list[APIKey] = Field(min_length=1)
    max_sync_connections_per_subject: int | None = Field(default=None, ge=1)
    request_rate: RequestRate | None = None
    max_inflight_tasks_per_subject: int | None = Field(default=None, ge=1, le=1_000_000)


@dataclass(frozen=True)
class Identity:
    subject: str | None
    permissions: frozenset[str]
    entries: frozenset[str]
    administrator: bool = False

    def allows(self, permission, name=None):
        return self.administrator or (permission in self.permissions and (
            name is None or '*' in self.entries or name in self.entries))

    def require(self, permission, name=None):
        if not self.allows(permission, name):
            raise ServerError(403, 'forbidden', 'This operation is not permitted')

    def owns(self, record):
        return self.administrator or record.get('owner') == self.subject


class Authentication:
    """Only token digests are retained; credential values never enter config/DB."""
    def __init__(self, config=None, admin_token=None):
        self.enabled = config is not None
        self.keys = []
        self.admin_digest = self.digest(admin_token) if admin_token else None
        self.max_sync_connections_per_subject = None
        self.request_rate = None
        self.max_inflight_tasks_per_subject = None
        if config is not None:
            value = AuthConfig.model_validate(config)
            self.max_sync_connections_per_subject = value.max_sync_connections_per_subject
            self.request_rate = value.request_rate
            self.max_inflight_tasks_per_subject = value.max_inflight_tasks_per_subject
            seen = set()
            for key in value.keys:
                token = os.environ.get(key.token_env)
                if not token or len(token) < 32:
                    raise ValueError('API key environment variables must contain at least 32 characters')
                digest = self.digest(token)
                if digest in seen or digest == self.admin_digest:
                    raise ValueError('API key credentials must be distinct from one another and the management token')
                seen.add(digest)
                self.keys.append((digest, Identity(key.subject, frozenset(key.permissions), frozenset(key.entries))))

    @staticmethod
    def digest(token):
        return hashlib.sha256(token.encode()).digest()

    def authenticate(self, authorization):
        scheme, _, token = authorization.partition(' ')
        if authorization:
            if scheme.lower() != 'bearer' or not token:
                raise ServerError(401, 'unauthorized', 'A valid Bearer token is required')
            digest = self.digest(token)
            if self.admin_digest and secrets.compare_digest(digest, self.admin_digest):
                return Identity('@administrator', frozenset(), frozenset(), True)
            for expected, identity in self.keys:
                if secrets.compare_digest(digest, expected):
                    return identity
            raise ServerError(401, 'unauthorized', 'A valid Bearer token is required')
        if self.enabled:
            raise ServerError(401, 'unauthorized', 'A valid Bearer token is required')
        return Identity(None, frozenset(('deploy', 'tasks:read', 'tasks:cancel', 'files:read', 'files:write')), frozenset(('*',)))


def get_server(request: HTTPConnection):
    return request.app.state.server


def current_identity(request: HTTPConnection):
    try:
        caller = get_server(request).authentication.authenticate(request.headers.get('authorization', ''))
        request.scope.setdefault('state', {})['audit_identity'] = caller
        return caller
    except ServerError:
        if request.scope['type'] == 'websocket':
            raise WebSocketException(code=1008) from None
        raise


async def limited_identity(request: Request, caller: Identity = Depends(current_identity)):
    await get_server(request).audit.identify(request.scope, caller)
    await get_server(request).request_limiter.take(caller.subject)
    return caller


def get_db(request: Request):
    return get_server(request).db


def get_storage(request: Request):
    return get_server(request).storage


def _entry_context(request):
    context = getattr(request.state, 'server_entry', None)
    if context is None:
        raise RuntimeError('This dependency requires a router added with Server.add(router=...)')
    return context


def get_entry(request: Request):
    """Current entry description; does not expose the running model or worker."""
    name, _ = _entry_context(request)
    return get_server(request).describe(name)


def get_redis(request: Request):
    """Current entry's shared Redis client, when Redis scheduling is configured."""
    name, _ = _entry_context(request)
    entry = get_server(request)._entry(name)
    client = getattr(entry.get('queue'), 'client', None)
    if client is None:
        raise ServerError(503, 'redis_unavailable', 'This entry has no active Redis connection')
    return client


def get_deploy(request: Request):
    """An authenticated Deploy callable bound to this HTTP route's entry.

    Algorithm routes translate their body into DeployRequest and await this
    callable. Sync returns the task result; async and stream return HTTP responses.
    Idempotency-Key, Accept and ownership come from the original request.
    """
    from .api.deploy import deploy_request
    from .protocol import DeployRequest
    name, caller = _entry_context(request)
    server = get_server(request)

    async def deploy(body: DeployRequest):
        caller.require('deploy', name)
        if not isinstance(body, DeployRequest):
            raise TypeError('Algorithm routes must translate requests into DeployRequest')
        return await deploy_request(server, name, body, caller, request=request,
                                    idempotency_key=request.headers.get('idempotency-key'),
                                    accept=request.headers.get('accept', '*/*'))
    return deploy

"""Editor access reuses management identity; browser sessions stay process-local."""
import asyncio
import secrets
import time
from urllib.parse import urlsplit

from fastapi import Request
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse

from .errors import ServerError
from .dependencies import Identity


def origin_key(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Origins must be explicit HTTP(S) origins')
    if parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('Origins cannot contain paths, queries or fragments')
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)


class EditorAccess:
    lifetime = 8 * 60 * 60
    max_sessions = 128

    def __init__(self, authentication, origins, scope):
        self.authentication = authentication
        # Cookies are not port-scoped. Distinct local Servers must not overwrite
        # each other's browser sessions when both use the same hostname.
        self.cookie_name = 'nndeploy_editor_' + authentication.digest(str(scope)).hex()[:16]
        self.required = authentication.enabled or authentication.admin_digest is not None
        self.origins = {origin_key(value) for value in origins}
        self.sessions = {}

    def check_origin(self, connection, *, required=False):
        origin = connection.headers.get('origin')
        if not origin:
            if required:
                raise ServerError(403, 'origin_required', 'Browser session writes require an Origin header')
            return
        try:
            scheme = {'ws': 'http', 'wss': 'https'}.get(connection.url.scheme, connection.url.scheme)
            own = origin_key(f'{scheme}://{connection.url.netloc}')
            allowed = origin_key(origin) in self.origins | {own}
        except ValueError:
            allowed = False
        if not allowed:
            raise ServerError(403, 'origin_forbidden', 'This browser origin is not permitted')

    def session(self, connection):
        token = connection.cookies.get(self.cookie_name, '')
        digest = self.authentication.digest(token)
        deadline = self.sessions.get(digest, 0)
        if deadline <= time.monotonic():
            self.sessions.pop(digest, None)
            return None
        return digest

    def require(self, connection, *, management=False):
        authorization = connection.headers.get('authorization', '')
        identity = Identity(None, frozenset(), frozenset())
        if authorization:
            identity = self.authentication.authenticate(authorization)
            connection.scope.setdefault('state', {})['audit_identity'] = identity
            if not identity.administrator:
                raise ServerError(403, 'forbidden', 'Editor access requires management permission')
        elif self.session(connection) is None:
            if self.required:
                raise ServerError(401, 'unauthorized', 'Management sign-in required')
            if management:
                identity = Identity('@administrator', frozenset(), frozenset(), True)
        else:
            identity = Identity('@administrator', frozenset(), frozenset(), True)
        # Cookie-authorized writes must prove their browser origin. Bearer clients
        # need no cookie/CSRF mechanism, but a supplied Origin is always checked.
        connection.scope.setdefault('state', {})['audit_identity'] = identity
        self.check_origin(connection, required=not authorization and self.session(connection) is not None and (
            connection.scope['type'] == 'websocket' or connection.scope.get('method') not in ('GET', 'HEAD', 'OPTIONS')))

    def register(self, app):
        @app.get('/api/auth/session')
        async def status(request: Request):
            self.check_origin(request)
            return JSONResponse({'required': self.required, 'authenticated': self.session(request) is not None,
                                 'management_available': not self.required or self.authentication.admin_digest is not None},
                                headers={'Cache-Control': 'no-store'})

        @app.post('/api/auth/session')
        async def login(request: Request):
            self.check_origin(request)
            # A cookie cannot renew itself: each new session needs the token.
            identity = self.authentication.authenticate(request.headers.get('authorization', ''))
            request.state.audit_identity = identity
            if not identity.administrator:
                raise ServerError(403, 'forbidden', 'A management token is required')
            await request.app.state.server.audit.identify(request.scope, identity)
            now = time.monotonic()
            self.sessions = {key: value for key, value in self.sessions.items() if value > now}
            old = self.session(request)
            if len(self.sessions) >= self.max_sessions and old is None:
                raise ServerError(429, 'session_limit', 'Too many active editor sessions')
            if old is not None:
                self.sessions.pop(old, None)
            token = secrets.token_urlsafe(32)
            self.sessions[self.authentication.digest(token)] = now + self.lifetime
            response = JSONResponse({'authenticated': True, 'expires_in': self.lifetime}, headers={'Cache-Control': 'no-store'})
            response.set_cookie(self.cookie_name, token, max_age=self.lifetime, httponly=True,
                                secure=request.url.scheme == 'https', samesite='strict', path='/api')
            return response

        @app.delete('/api/auth/session')
        async def logout(request: Request):
            self.check_origin(request, required=True)
            identity = Identity('@administrator', frozenset(), frozenset(), True) if self.session(request) is not None else Identity(None, frozenset(), frozenset())
            await request.app.state.server.audit.identify(request.scope, identity)
            self.sessions.pop(self.authentication.digest(request.cookies.get(self.cookie_name, '')), None)
            response = JSONResponse({'authenticated': False}, headers={'Cache-Control': 'no-store'})
            response.delete_cookie(self.cookie_name, path='/api', httponly=True, samesite='strict',
                                   secure=request.url.scheme == 'https')
            return response


class EditorAccessMiddleware:
    """Check before body parsing; revoke connected sockets on expiry or sign-out."""
    def __init__(self, app, access):
        self.app, self.access = app, access

    async def __call__(self, scope, receive, send):
        if scope['type'] not in ('http', 'websocket') or not (
                scope['path'] == '/api' or scope['path'].startswith('/api/')):
            return await self.app(scope, receive, send)
        connection = HTTPConnection(scope)
        if scope['path'] == '/api/auth/session':
            return await self.app(scope, receive, send)
        try:
            self.access.require(connection)
            await connection.app.state.server.audit.identify(scope, scope['state']['audit_identity'])
        except ServerError as exc:
            if scope['type'] == 'websocket':
                return await send({'type': 'websocket.close', 'code': 1013 if exc.status_code >= 500 else 1008})
            return await JSONResponse(exc.response(), status_code=exc.status_code)(scope, receive, send)

        if scope['type'] == 'http':
            async def safe_send(message):
                if message['type'] == 'http.response.start':
                    message['headers'] = list(message.get('headers', [])) + [
                        (b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
                        (b'content-security-policy', b"default-src 'none'; sandbox")]
                await send(message)
            return await self.app(scope, receive, safe_send)

        # One pending receive, even during idle checks; no lost messages on timeout.
        pending = None
        revoked = False
        async def checked_receive():
            nonlocal pending, revoked
            while True:
                if revoked:
                    return {'type': 'websocket.disconnect', 'code': 1008}
                try:
                    self.access.require(connection)
                except ServerError:
                    revoked = True
                    await send({'type': 'websocket.close', 'code': 1008})
                    return {'type': 'websocket.disconnect', 'code': 1008}
                if pending is None:
                    pending = asyncio.create_task(receive())
                done, _ = await asyncio.wait([pending], timeout=1)
                if done:
                    message = pending.result()
                    pending = None
                    # Revalidate messages arriving during the idle wait.
                    try:
                        self.access.require(connection)
                    except ServerError:
                        revoked = True
                        await send({'type': 'websocket.close', 'code': 1008})
                        return {'type': 'websocket.disconnect', 'code': 1008}
                    return message

        async def checked_send(message):
            nonlocal revoked
            if revoked:
                return
            try:
                self.access.require(connection)
            except ServerError:
                revoked = True
                await send({'type': 'websocket.close', 'code': 1008})
                return
            await send(message)
        try:
            await self.app(scope, checked_receive, checked_send)
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

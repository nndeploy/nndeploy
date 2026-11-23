"""Entry-scoped algorithm HTTP routes sharing the Server dependencies."""
from fastapi import APIRouter, Depends, Request
from fastapi.routing import APIRoute
from ..dependencies import Identity, limited_identity


def prepare_router(server, name, router, *, prefix=None, permissions=None, authorize_entry=True):
    # Assemble and validate before changing the application or its entries.
    if server.started or server.starting_task is not None:
        raise ValueError('Add algorithm routes before starting the Server')
    if not isinstance(router, APIRouter):
        raise ValueError('Algorithm router must be a FastAPI APIRouter')
    if router.on_startup or router.on_shutdown or type(router.lifespan_context) is not type(APIRouter().lifespan_context):
        raise ValueError('Use the entry handler lifecycle instead of router lifecycle hooks')
    for route in router.routes:
        if not isinstance(route, APIRoute):
            raise ValueError('Algorithm routers currently support HTTP routes only')
        first = route.path.lstrip('/').split('/')[0]
        if not first or '{' in first or first in ('deploy', 'schema'):
            raise ValueError('Algorithm routes need a literal first path segment other than deploy or schema')

    prefix = f'/v1/server/{name}' if prefix is None else prefix
    if not isinstance(prefix, str) or (prefix and (not prefix.startswith('/') or prefix.endswith('/') or '{' in prefix)):
        raise ValueError('Router prefix must be empty or an absolute literal path without a trailing slash')
    allowed = {'deploy', 'tasks:read', 'tasks:cancel', 'files:read', 'files:write'}
    declared = {(method, route.path) for route in router.routes for method in route.methods}
    permissions = dict(permissions or {})
    if permissions.keys() - declared or any(value not in allowed for value in permissions.values()):
        raise ValueError('Route permissions must name declared methods/paths and supported permissions')

    async def authorize(request: Request, caller: Identity = Depends(limited_identity)):
        route_path = request.scope['route'].path
        relative = route_path if (request.method, route_path) in declared else route_path[len(prefix):]
        caller.require(permissions.get((request.method, relative), 'deploy'), name)
        request.state.server_entry = (name, caller)

    mounted = APIRouter()
    # Shared resource routers perform identity/entry/ownership checks themselves.
    mounted.include_router(router, prefix=prefix, dependencies=[Depends(authorize)] if authorize_entry else [])
    # OpenAPI resolves included routers on both eager and lazy FastAPI versions.
    # Normalize parameter names: /{id} and /{task_id} are the same route.
    import re
    canonical = lambda path: re.sub(r'\{[^}]+\}', '{}', path)
    existing = {(method.upper(), canonical(path))
                for path, operations in server.app.openapi().get('paths', {}).items()
                for method in operations if method.upper() in {'GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'}}
    existing.update((method, canonical(route.path)) for route in server.app.routes
                    if isinstance(route, APIRoute) for method in route.methods)
    existing.update(getattr(server, '_algorithm_routes', set()))
    added = set()
    for route in router.routes:
        for method in route.methods:
            key = method, canonical(prefix + route.path)
            if key in existing:
                raise ValueError(f'Algorithm route conflicts with an existing route: {prefix + route.path}')
            existing.add(key)
            added.add(key)
    mounted.nndeploy_route_keys = added
    return mounted


def mount_router(server, mounted):
    """Mount a validated router before the editor's SPA catch-all."""
    from starlette.routing import Mount
    routes = server.app.router.routes
    insert_at = next((index for index, route in enumerate(routes)
                      if isinstance(route, Mount) and route.path == ''), len(routes))
    previous_count = len(routes)
    server.app.include_router(mounted)
    added = routes[previous_count:]
    del routes[previous_count:]
    routes[insert_at:insert_at] = added
    server.app.openapi_schema = None
    server._algorithm_routes = getattr(server, '_algorithm_routes', set()) | mounted.nndeploy_route_keys

"""Entry-scoped algorithm HTTP routes sharing the Server dependencies."""
from fastapi import APIRouter, Depends, Request
from fastapi.routing import APIRoute
from ..dependencies import Identity, limited_identity


def prepare_router(server, name, router):
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

    async def authorize(request: Request, caller: Identity = Depends(limited_identity)):
        caller.require('deploy', name)
        request.state.server_entry = (name, caller)

    mounted = APIRouter()
    mounted.include_router(router, prefix=f'/v1/server/{name}', dependencies=[Depends(authorize)])
    return mounted

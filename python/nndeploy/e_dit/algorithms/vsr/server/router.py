"""Algorithm-specific submission; shared video resources live in Server."""
from fastapi import APIRouter, Depends, Request
from nndeploy.server.dependencies import limited_identity
from .protocol import VideoRestorationRequest


def create_router(service):
    router = APIRouter(tags=['VSR'])
    schema = VideoRestorationRequest.model_json_schema()
    definitions = schema.pop('$defs', {})
    def inline(value):
        if isinstance(value, dict):
            if '$ref' in value:
                return inline(definitions[value['$ref'].split('/')[-1]])
            return {key: inline(item) for key, item in value.items()}
        if isinstance(value, list):
            return [inline(item) for item in value]
        return value

    @router.post('/v1/videos/restorations', openapi_extra={'requestBody': {
        'required': True, 'content': {'application/json': {'schema': inline(schema)}}}})
    async def restore(request: Request, caller=Depends(limited_identity)):
        try:
            body = await request.json()
        except (ValueError, TypeError) as error:
            return {'code': 1, 'message': str(error)}
        return await service.submit(body, caller)

    return router

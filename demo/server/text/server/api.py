from fastapi import APIRouter, Depends

from nndeploy.server.dependencies import get_deploy
from nndeploy.server.protocol import DeployRequest

from .protocol import TextRequest, TextResponse

router = APIRouter()


@router.post('/text', response_model=TextResponse)
async def text(body: TextRequest, deploy=Depends(get_deploy)):
    parameters = {} if body.uppercase is None else {'uppercase': body.uppercase}
    result = await deploy(DeployRequest(inputs={'text': body.text}, parameters=parameters, mode=body.mode))
    if isinstance(result, dict):
        return {'id': result['task_id'], 'text': result['outputs']['text']}
    return result  # Async returns the standard 202 response and task query URL.

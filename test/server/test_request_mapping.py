"""Business requests retain normal task validation, ownership and execution."""
import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

from nndeploy.server import Server
from nndeploy.server.request_mapping import RequestMapping
from nndeploy.server.dependencies import get_entry, get_deploy


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)
    image: str = Field(alias='imagePath')
    threshold: float = Field(default=0.5, ge=0, le=1)


def mapping():
    return RequestMapping(DetectRequest, {'image': 'inputs.image', 'threshold': 'parameters.detector.threshold'})


def detect(inputs, detector):
    return {'image': inputs['image'], 'threshold': detector['threshold']}


def test_mapping_defaults_aliases_roundtrip_and_no_mutation():
    convert = mapping()
    original = {'inputs': {'image': 'input.png'}, 'parameters': {}}
    request = convert.parse(original)
    assert request.inputs == {'image': 'input.png'}
    assert request.parameters == {'detector': {'threshold': .5}}
    assert original['parameters'] == {}
    assert convert.from_deploy(request).image == 'input.png'
    assert convert.to_deploy({'imagePath': 'input.png'}).parameters == request.parameters
    with pytest.raises(ValueError):
        convert.parse({'inputs': {'image': 'x'}, 'threshold': .3})
    with pytest.raises(ValueError, match='without a business mapping'):
        convert.parse({'inputs': {'image': 'x', 'hidden': True}})
    with pytest.raises(ValueError):
        convert.to_deploy({'image': 'x', 'threshold': 2})


def test_overlapping_or_missing_mappings_rejected():
    with pytest.raises(ValueError, match='overlap'):
        RequestMapping(DetectRequest, {'image': 'inputs.value', 'threshold': 'inputs.value.threshold'})
    with pytest.raises(ValueError, match='Every'):
        RequestMapping(DetectRequest, {'image': 'inputs.image'})


def test_business_and_deploy_envelope_execute_same_request(tmp_path):
    server = Server(name='detect', deploy=detect, resources=tmp_path, data_dir=tmp_path,
                    router=mapping().router('/detect'), router_prefix='/v1',
                    config={'inputs': {'image': {'type': 'string'}},
                            'parameters': {'detector': {'type': 'json'}}})
    with TestClient(server.app) as client:
        response = client.post('/v1/detect', json={'imagePath': 'x.png'})
        assert response.status_code == 200, response.text
        assert response.json()['outputs'] == {'image': 'x.png', 'threshold': .5}
        response = client.post('/v1/detect', json={'inputs': {'image': 'x.png'},
                                                 'parameters': {'detector': {'threshold': .7}}})
        assert response.status_code == 200, response.text
        assert response.json()['outputs']['threshold'] == .7
        assert client.post('/v1/detect', json={'image': 'x', 'unknown': 1}).status_code == 422
        schema = client.get('/openapi.json').json()
        assert 'DetectRequest' in schema['components']['schemas']
        assert 'imagePath' in schema['components']['schemas']['DetectRequest']['properties']
        assert client.post('/v1/server/detect/deploy', json={
            'inputs': {'image': 'x'}, 'parameters': {'detector': {'threshold': .2}}}).status_code == 200


def test_complete_path_collision_is_atomic(tmp_path):
    server = Server(resources=tmp_path, data_dir=tmp_path)
    router = APIRouter()
    @router.get('/v1/models')
    def models():
        return {}
    count = len(server.app.routes)
    with pytest.raises(ValueError, match='conflicts'):
        server.add(name='bad', deploy=lambda value: value, router=router, router_prefix='')
    assert 'bad' not in server.entries
    assert len(server.app.routes) == count
    server.db.close()


def test_route_permission_is_independent_of_deploy(tmp_path, monkeypatch):
    monkeypatch.setenv('MAPPING_READER', 'r' * 32)
    auth={'keys': [{'subject': 'reader', 'token_env': 'MAPPING_READER',
                    'entries': ['detect'], 'permissions': ['tasks:read']}]}
    router = APIRouter()
    @router.get('/status')
    def status(entry=Depends(get_entry)):
        return {'name': entry['name']}
    @router.post('/submit')
    def submit():
        return {}
    @router.post('/read-only')
    async def read_only(deploy=Depends(get_deploy)):
        from nndeploy.server.protocol import DeployRequest
        return await deploy(DeployRequest(inputs={}))
    server=Server(name='detect', deploy=lambda value: value, resources=tmp_path, data_dir=tmp_path, auth_config=auth,
                  router=router, router_prefix='/v1/video', router_permissions={('GET', '/status'): 'tasks:read', ('POST', '/read-only'): 'tasks:read'})
    with TestClient(server.app) as client:
        headers={'Authorization': 'Bearer ' + 'r' * 32}
        assert client.get('/v1/video/status', headers=headers).status_code == 200
        assert client.post('/v1/video/submit', headers=headers).status_code == 403
        assert client.post('/v1/video/read-only', headers=headers).status_code == 403
        assert client.get('/v1/video/status').status_code == 401


def test_mapping_cannot_bypass_internal_validation(tmp_path):
    server = Server(name='detect', deploy=detect, resources=tmp_path, data_dir=tmp_path,
        router=mapping().router(), config={'inputs': {'image': {'type': 'string'}}})
    with TestClient(server.app) as client:
        # The business model accepts threshold, but the entry did not expose it.
        response = client.post('/v1/server/detect/request', json={'imagePath': 'x'})
        assert response.status_code == 422


def test_hidden_route_collision_is_rejected(tmp_path):
    router = APIRouter()
    router.add_api_route('/hidden', lambda: {}, include_in_schema=False)
    server = Server(name='first', deploy=lambda value: value, resources=tmp_path, data_dir=tmp_path,
                    router=router, router_prefix='/v1')
    with pytest.raises(ValueError, match='conflicts'):
        server.add(name='second', deploy=lambda value: value, router=router, router_prefix='/v1')
    import asyncio
    asyncio.run(server.close())

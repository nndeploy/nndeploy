import copy
import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from nndeploy.server import Server
from test_server_api import image_graph, text_graph


def test_workflow_and_assets_survive_draft_deletion_and_restart(tmp_path):
    drafts = tmp_path / 'drafts'
    drafts.mkdir()
    asset = drafts / 'input.png'
    Image.new('RGB', (8, 10), (90, 50, 20)).save(asset)
    workflow = image_graph()
    workflow['node_repository_'][0]['path_'] = 'input.png'
    original = copy.deepcopy(workflow)
    path = drafts / 'image.json'
    path.write_text(json.dumps(workflow))
    config = {'name': 'image', 'outputs': {'image': {'type': 'image', 'node': ['encode']}}}
    sidecar = path.with_suffix('.server.json')
    sidecar.write_text(json.dumps(config))
    data_dir = tmp_path / 'server'
    with TestClient(Server(workflow=path, data_dir=data_dir).app) as client:
        assert json.loads(path.read_text()) == original
        path.unlink()
        sidecar.unlink()
        asset.unlink()
        output = client.post('/v1/server/image/deploy', json={})
        assert output.status_code == 200, output.text
    with TestClient(Server(data_dir=data_dir).app) as client:
        output = client.post('/v1/server/image/deploy', json={})
        assert output.status_code == 200, output.text
        identity = output.json()['outputs']['image']['file_id']
        response = client.get('/v1/files/' + identity + '/content')
        with Image.open(io.BytesIO(response.content)) as image:
            assert image.size == (8, 10) and image.getpixel((0, 0)) == (90, 50, 20)


def test_explicit_stop_stays_stopped_after_restart(tmp_path):
    server = Server(workflow=text_graph(), data_dir=tmp_path, admin_token='test')
    with TestClient(server.app) as client:
        assert client.post('/api/server/echo/stop', headers={'Authorization': 'Bearer test'}).status_code == 200
    with TestClient(Server(data_dir=tmp_path, admin_token='test').app) as client:
        assert client.get('/v1/server/echo/schema').json()['status'] == 'stopped'
        assert client.post('/api/server/echo/start', headers={'Authorization': 'Bearer test'}).status_code == 200
        assert client.post('/v1/server/echo/deploy', json={'inputs': {'source': 'again'}}).json()['outputs']['sink'] == 'again'


def test_failed_reload_preserves_restart_configuration(tmp_path):
    import asyncio
    async def scenario():
        import pytest
        server = Server(workflow=text_graph(), data_dir=tmp_path)
        await server.start()
        broken = text_graph()
        broken['node_repository_'][0]['key_'] = 'unknown.Node'
        with pytest.raises(Exception):
            await server.reload_entry('echo', workflow=broken)
        await server.close()
    asyncio.run(scenario())
    with TestClient(Server(data_dir=tmp_path).app) as client:
        result = client.post('/v1/server/echo/deploy', json={'inputs': {'source': 'original'}})
        assert result.status_code == 200, result.text
        assert result.json()['outputs']['sink'] == 'original'

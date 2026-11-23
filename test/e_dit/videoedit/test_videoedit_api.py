"""Exercise original HTTP protocols through real Server DB, Worker and storage."""
import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nndeploy.server import Server
from nndeploy.e_dit.algorithms.wan.server import add_videoedit


class FakeGraph:
    def init(self):
        from nndeploy.base import StatusCode
        return StatusCode.Ok
    def deinit(self):
        pass
    def infer_files(self, output, **sampling):
        from nndeploy.server.request_context import current_request
        context = current_request()
        from nndeploy.e_dit.registry import get_model
        get_model('wan_videoedit').sampling(sampling)
        if sampling['prompt'] == 'slow':
            for step in range(100):
                context.publish({'progress': step})
                time.sleep(.05)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sampling['video_input_path'], output)
        paths = [str(output)]
        if sampling['save_crop_only']:
            crop = output.with_name(output.stem + '_crop_only' + output.suffix)
            shutil.copyfile(output, crop)
            paths.append(str(crop))
        context.publish({'progress': 98, 'steps': sampling['num_inference_steps'],
                         'guidance': sampling['guidance_scale']})
        return paths


def graph_factory(variant, params):
    return FakeGraph()


def make_inputs(path):
    path.mkdir(exist_ok=True)
    video = path/'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5, (32, 32))
    assert writer.isOpened()
    for step in range(5):
        writer.write(np.full((32,32,3), 50+step, np.uint8))
    writer.release()
    mask = path/'mask.npy'; np.save(mask, np.ones((5,32,32), dtype=np.uint8))
    reference = path/'reference.png'; Image.new('RGB', (32,32)).save(reference)
    return dict(taskId='one', prompt='edit', video_input_path=str(video), mask_input_path=str(mask),
                reference_image_path=str(reference), useClip=False)


def configured(tmp_path, contract='gateway', isolation='thread', **kwargs):
    server = Server(resources=tmp_path/'server', data_dir=tmp_path/'server', **kwargs)
    add_videoedit(server, {'normal': {'model_path': 'unused'}, 'dmd': {'model_path': 'unused-dmd'}},
                  graph_factory=graph_factory, contract=contract, execution={'isolation': isolation})
    return server


def wait(client, identity, **kwargs):
    for _ in range(200):
        response = client.get('/v1/videos/'+identity, **kwargs)
        assert response.status_code == 200, response.text
        value = response.json()
        if value['status'] in ('completed', 'failed', 'cancelled', 'expired'):
            return value
        time.sleep(.05)
    raise AssertionError('Task did not finish')


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_repair_dmd_files_and_duplicate_task(tmp_path, isolation):
    server = configured(tmp_path, isolation=isolation)
    body = make_inputs(tmp_path/'inputs')
    body.update(model='dmd', save_crop_only=True, num_inference_steps=30)
    with TestClient(server.app) as client:
        response = client.post('/v1/videos/repairs', json=body)
        assert response.status_code == 200, response.text
        assert response.json()['variant'] == 'dmd'
        value = wait(client, 'one')
        assert value['status'] == 'completed', value
        assert value['progress'] == 100
        content = client.get('/v1/videos/one/content')
        assert content.status_code == 200 and content.content == Path(body['video_input_path']).read_bytes()
        record = server.db.external_task('videoedit', 'one', None, server.config_scope)
        assert len(record['outputs']['artifacts']) == 2
        assert server.db.latest_progress(record['task_id'])['steps'] == 4
        duplicate = client.post('/v1/videos/repairs', json=body)
        assert duplicate.status_code == 409, duplicate.text
        assert client.get('/admin/queue').json()['counts']['completed'] == 1


def test_busy_cancel_and_deadline(tmp_path):
    server = configured(tmp_path)
    body = make_inputs(tmp_path/'inputs');body['prompt']='slow'
    with TestClient(server.app) as client:
        assert client.post('/v1/videos/repairs', json=body).json()['code'] == 0
        other = {**body, 'taskId': 'two', 'model': 'dmd'}
        assert client.post('/v1/videos/repairs', json=other).json()['code'] == 2
        assert client.delete('/v1/videos/one').status_code == 200
        value = wait(client, 'one')
        assert value['status'] == 'cancelled' and value['error']['code'] == 'task_cancelled'
        other['timeout']=1
        assert client.post('/v1/videos/repairs', json=other).json()['code'] == 0
        value = wait(client, 'two')
        assert value['status'] == 'expired' and value['error']['code'] == 'execution_timeout'


def test_backend_failed_submission_and_missing_task_id(tmp_path):
    server=configured(tmp_path, contract='backend')
    body=make_inputs(tmp_path/'inputs')
    with TestClient(server.app) as client:
        invalid={**body,'infer_len':4}
        response=client.post('/v1/videos/repairs',json=invalid)
        assert response.status_code == 200 and response.json()['code']==1
        assert client.get('/v1/videos/one').json()['status']=='failed'
        body['taskId']='valid'
        assert client.post('/v1/videos/repairs',json=body).json()=={'code':0,'message':'ok'}
        assert wait(client,'valid')['status']=='completed'
        body.pop('taskId')
        assert client.post('/v1/videos/repairs',json=body).json()['code']==1


def test_external_id_recovered_after_restart(tmp_path):
    body=make_inputs(tmp_path/'inputs')
    with TestClient(configured(tmp_path).app) as client:
        client.post('/v1/videos/repairs',json=body)
        assert wait(client,'one')['status']=='completed'
    with TestClient(configured(tmp_path).app) as client:
        assert client.get('/v1/videos/one').json()['status']=='completed'
        assert client.get('/v1/videos/one/content').status_code==200
        assert client.post('/v1/videos/repairs',json=body).status_code==409

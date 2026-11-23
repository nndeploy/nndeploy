"""Differential checks against definitions read from the original repository.

Only pure protocol definitions and status routes are loaded, so tests need no
SGLang scheduler, weights or LLM dependencies. No reference source is rewritten.
"""
import ast
from copy import deepcopy
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Path as APIPath, Query
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, model_validator
import pytest

from nndeploy.e_dit.algorithms.vsr.server import protocol
from nndeploy.e_dit.algorithms.vsr.server.service import VSRService
from nndeploy.server import Server
from nndeploy.server.api.videos import prepare_video_resources
import io
from nndeploy.e_dit.algorithms.vsr.config import VSRConfig


def load_nodes(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', None)
        if name in names:
            nodes.append(node)
    assert len(nodes) == len(names), names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


@pytest.fixture
def reference():
    root = Path(os.environ.get('SGLANG_ROOT', '/home/zhouhao6/vsr/sglang'))
    folder = root / 'python/sglang/multimodal_gen/runtime/entrypoints/openai'
    if not folder.is_dir():
        pytest.skip('Set SGLANG_ROOT to run upstream contract tests')
    ns = dict(globals(), Path=APIPath, parse_resolution=protocol.parse_resolution,
              router=APIRouter(prefix='/v1/videos'))
    load_nodes(folder/'protocol.py', {'VideoResponse','VideoListResponse','VideoRepairMinioConfig'}, ns)
    load_nodes(folder/'video_api.py', {'_VIDEO_REPAIR_FIELD_ALIASES','_VIDEO_REPAIR_MINIO_FIELD_ALIASES',
        '_normalize_aliases','_normalize_video_repair_payload','_job_error_message','_job_reason'}, ns)
    load_nodes(folder/'vsr_api.py', {'VideoRestorationRequest','_validate_tiling'}, ns)
    return ns, folder


@pytest.mark.parametrize('body', [
    {'video_input_path': '/input.mp4'},
    {'videoUrl': 'http://video', 'taskId': 'camel', 'task_id': 'snake'},
    {'video_input_path': '/x', 'read_queue': '2', 'preserve_audio': 'false'},
    {'video_input_path': '/x', 'minioConfig': {'endpoint':'x','bucket':'a','bucketName':'b',
        'rootUser':'u','accessKey':'key','rootPass':'p','secretKey':'secret','useSSL':True}},
    {'video_input_path': '/x', 'prompt': 'bad'},
    {'video_input_path': '/x', 'read_queue': 17},
    {'video_input_path': '/x', 'outputObjectKey': 'key'},
    {'video_input_path': '/x', 'videoUrl': 'http://video'},
    {'video_input_path': '/x', 'target_resolution': '13x20'},
    {'video_input_path': '/x', 'callbackUrl': 'ftp://callback'},
    {'video_input_path': '/x', 'taskId': '../../bad'},
    {'video_input_path': '/x', 'maskUrl': 'unused'},
    {'video_input_path': '/x', 'minioConfig': []}, {}, [], None,
])
def test_request_contract(reference, body):
    ns, _ = reference
    results=[]
    for model in (ns['VideoRestorationRequest'], protocol.VideoRestorationRequest):
        try:
            results.append(('ok', model.model_validate(body).model_dump()))
        except (ValueError, TypeError) as error:
            results.append(('error', str(error)))
    assert results[0] == results[1]


@pytest.mark.parametrize('status', ['queued', 'running', 'cancelling', 'completed', 'failed'])
def test_response_routes_contract(reference, tmp_path, status):
    ns, folder = reference
    record = dict(id='id', model='vsr', status=status, progress=0 if status == 'queued' else 1, created_at=123,
                  file_path=None, size='130x98', seconds='1.0',
                  request_cancel_path=str(tmp_path/'cancel'))
    if status=='completed':
        path=tmp_path/'output.mp4'; path.write_bytes(b'video')
        record.update(progress=100,file_path=str(path),completed_at=456,inference_time_s=1.2)
    if status=='failed':
        record.update(error={'message':'Request timed out.'}, completed_at=456)
    class ReferenceStore:
        async def get(self, identity):
            return deepcopy(record) if identity == 'id' else None
        async def list_values(self):
            return [deepcopy(record)]
    ns.update(VIDEO_STORE=ReferenceStore(), _current_video_progress=lambda job: 100 if job['status']=='completed' else job.get('progress',0))
    load_nodes(folder/'video_api.py', {'retrieve_video','retrieve_video_progress','list_videos','download_video_content'}, ns)
    baseline=FastAPI(); baseline.include_router(ns['router'])
    server = Server(name='vsr', deploy=lambda inputs: inputs, resources=tmp_path/'server', data_dir=tmp_path/'server')
    service = VSRService(server, VSRConfig())
    prepare_video_resources(server).register(service)
    with TestClient(baseline) as a, TestClient(server.app) as b:
        server.db.create_task('internal', 'vsr', 123, service={'scope': server.config_scope, 'external_id': 'id'})
        if status != 'queued':
            server.db.start_task('internal', 124)
            server.db.append_task_event('internal', {'progress': 1, 'size': '130x98', 'seconds': '1.0'})
        else:
            record.update(size='', seconds='4')
        if status == 'completed':
            artifact = server.storage.save(io.BytesIO(b'video'), 'application/octet-stream', task_id='internal')
            path, _ = server.storage.resolve(artifact['file_id'])
            record['file_path'] = str(path)
            server.db.finish_task('internal', 'succeeded', 456, outputs={
                'size': '130x98', 'seconds': '1.0', 'inference_time_s': 1.2,
                'artifacts': [{'file': {'file_id': artifact['file_id'], 'local_path': str(path)}}]})
        elif status == 'failed':
            server.db.finish_task('internal', 'failed', 456, error={'message': 'Request timed out.'})
        elif status == 'cancelling':
            server.db.request_cancel('internal')
        for path in ('/v1/videos/id','/v1/videos/id/progress','/v1/videos/id/content',
                     '/v1/videos/missing','/v1/videos','/v1/videos?after=missing',
                     '/v1/videos?order=garbage&limit=1'):
            x,y=a.get(path),b.get(path)
            assert x.status_code==y.status_code, path
            assert x.content==y.content, (path,x.content,y.content)


@pytest.mark.parametrize('state', ['succeeded', 'failed', 'cancelled', 'expired'])
def test_durable_callback_reports_terminal_outcome(state):
    from nndeploy.e_dit.algorithms.vsr.server.callbacks import task_callback
    record = dict(task_id='internal', state=state, created_at=123, finished_at=456,
                  service={'external_id': 'external'}, outputs={}, error=None,
                  _progress={'progress': 1})
    job = dict(id='external', object='video', model='vsr', created_at=123,
               completed_at=456, file_path=None, url=None, error=None, reason=None)
    if state == 'succeeded':
        record['outputs'] = {'url': 'https://storage/video.mp4', 'inference_time_s': 2.5}
        job.update(status='completed', progress=100, completed_at=456,
                   url='https://storage/video.mp4', inference_time_s=2.5)
    else:
        code = {'failed': 'execution_failed', 'cancelled': 'task_cancelled', 'expired': 'execution_timeout'}[state]
        message = {'failed': 'upload failed', 'cancelled': 'Request cancelled.', 'expired': 'Request timed out.'}[state]
        record['error'] = {'code': code, 'message': message}
        job.update(status=state, progress=1, error=record['error'], reason=message)
    assert task_callback(record) == job

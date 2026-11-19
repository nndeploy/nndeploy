"""Source VideoEdit HTTP contracts without importing SGLang or model weights."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nndeploy.e_dit.algorithms.wan.server.schemas import (
    VideoRepairRequest, normalize_request, parse_request, submit_response,
    default_video_repair_output_object_key,
)
from nndeploy.e_dit.algorithms.wan.server.policy import resolve_variant, apply_variant_policy
from nndeploy.e_dit.algorithms.wan.server.callbacks import callback_payload


def payload(**options):
    return dict(taskId='edit-1', prompt='red jacket', videoUrl='input.mp4',
                maskUrl='mask.npy', referenceImageUrl='reference.png', **options)


def test_source_defaults_and_explicit_alias_priority():
    request = parse_request(payload(task_id='other', refFrameIdx=2))
    assert request.task_id == 'edit-1'
    assert request.ref_frame_idx == 2
    assert (request.seed, request.num_inference_steps, request.guidance_scale) == (42, 40, 5)
    assert request.decode_mode == 'stream' and request.use_clip
    # Source accepts only its explicit aliases; do not invent new ones.
    assert parse_request(payload(numInferenceSteps=9)).num_inference_steps == 40
    source = payload(minioConfig={'endpoint': 'localhost:9000', 'bucket': 'a', 'bucketName': 'b',
                                'rootUser': 'user', 'rootPass': 'secret', 'useSSL': True})
    request = parse_request(source)
    assert request.minio_config.bucket_name == 'b'
    assert request.minio_config.secure
    assert source['minioConfig']['rootPass'] == 'secret'


@pytest.mark.parametrize('field', ['chunks', 'strength', 'dropReferenceFrame', 'varySeedByWindow', 'generator_device'])
def test_removed_fields_rejected(field):
    with pytest.raises(ValueError, match='removed VideoEdit request fields'):
        parse_request(payload(**{field: 1}))


@pytest.mark.parametrize('options', [
    {'timeout': 0}, {'timeout': -2}, {'refFrameIdx': True}, {'refFrameIdx': 1.5},
    {'refFrameIdx': -1}, {'num_frames': 0}, {'num_frames': 2, 'refFrameIdx': 2},
    {'infer_len': 4}, {'overlap': 49}, {'bridgeOverlap': 2},
    {'stabilizeMaskUnion': True, 'stabilizeMaskShape': True},
])
def test_request_rejections(options):
    with pytest.raises(ValueError):
        parse_request(payload(**options))


def test_task_id_required_and_storage_configuration():
    value = payload();value.pop('taskId')
    with pytest.raises(ValueError, match='taskId is required'):
        parse_request(value)
    with pytest.raises(ValueError, match='minioConfig'):
        parse_request(payload(output_storage='s3'))
    assert parse_request(payload(output_storage='s3'), storage_enabled=True).output_storage == 's3'


def test_variant_policy_and_submission_responses():
    request = parse_request(payload(negative_prompt='bad'))
    assert apply_variant_policy(request, 'normal') is request
    dmd = apply_variant_policy(request, resolve_variant(' VideoEdit-DMD '))
    assert (dmd.num_inference_steps, dmd.guidance_scale, dmd.dynamic_cfg, dmd.negative_prompt) == (4, 1, False, None)
    assert request.num_inference_steps == 40
    with pytest.raises(ValueError, match='Unsupported'):
        resolve_variant('unregistered')
    assert submit_response(0, 'ok') == {'code': 0, 'message': 'ok'}
    assert submit_response(1, 'bad') == {'code': 1, 'message': 'bad', 'reason': 'bad'}
    assert submit_response(2, 'A task is running.') == {'code': 2, 'message': 'A task is running.'}


def test_callback_contract_and_output_priority(tmp_path):
    progress = tmp_path / 'progress.json'
    progress.write_text('{"progress": 73}')
    assert callback_payload('edit-1', {'status': 'running', 'progress': 1, 'progress_path': str(progress)})['progress'] == 73
    result = callback_payload('edit-1', {'status': 'completed', 'progress': 99,
        'output_object_key': 'result/key.mp4', 'url': 'https://other', 'file_path': '/tmp/other',
        'created_at': 10, 'completed_at': 14})
    assert result == {'status': 'succeeded', 'progress': 100, 'reason': '',
                      'output': json.dumps({'gen_video_url': 'result/key.mp4', 'duration': 4})}
    assert callback_payload('edit-1', {'status': 'running', 'progress': 42}) == {
        'status': 'running', 'progress': 42, 'reason': '', 'output': ''}
    assert callback_payload('edit-1', {'status': 'failed', 'progress': 42,
        'error': {'message': 'Request timed out.'}})['reason'] == 'Request timed out.'


def test_source_contract_equivalence():
    root = os.environ.get('SGLANG_VIDEOEDIT_SOURCE')
    if not root:
        pytest.skip('Set SGLANG_VIDEOEDIT_SOURCE to compare the source checkout')
    root = Path(root) / 'runtime/entrypoints/openai'
    # Load only pure definitions from source; no SGLang runtime import needed.
    names = {'VideoRepairMinioConfig', '_REMOVED_VIDEOEDIT_REQUEST_FIELDS',
        'VideoRepairRequest', '_VIDEO_REPAIR_FIELD_ALIASES', '_VIDEO_REPAIR_MINIO_FIELD_ALIASES',
        '_normalize_aliases', '_normalize_video_repair_payload', '_validate_video_repair_request',
        '_build_video_repair_callback_payload', '_job_error_message'}
    namespace = {'json': json}
    exec('from typing import *\nfrom pydantic import BaseModel, Field, StrictFloat, StrictInt, model_validator', namespace)
    namespace['cloud_storage'] = SimpleNamespace(is_enabled=lambda: False)
    namespace['_current_video_progress'] = lambda job: int(job.get('progress', 0))
    for path in (root/'protocol.py', root/'video_api.py'):
        nodes = []
        for node in ast.parse(path.read_text()).body:
            name = getattr(node, 'name', None)
            if isinstance(node, ast.Assign):
                name = getattr(node.targets[0], 'id', None)
            if name in names:
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    for body in (payload(), payload(refFrameIdx=3, num_frames=-1), payload(decodeMode='eager')):
        source = namespace['VideoRepairRequest'](**namespace['_normalize_video_repair_payload'](body))
        namespace['_validate_video_repair_request'](source)
        assert parse_request(body).model_dump() == source.model_dump()
    source_fields = namespace['VideoRepairRequest'].model_json_schema()
    assert VideoRepairRequest.model_json_schema() == source_fields
    for state in ('running', 'failed', 'completed'):
        job = dict(status=state, progress=32, inference_time_s=3.5, url='result.mp4', error={'message': 'failure'})
        assert callback_payload('edit-1', job) == namespace['_build_video_repair_callback_payload']('edit-1', job)

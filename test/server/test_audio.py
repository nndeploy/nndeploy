"""Real audio files cross the same Graph/code/Task/file interfaces as arrays."""
import asyncio
import io
import json
import subprocess
import uuid
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

pytest.importorskip('av')

from nndeploy.server import Server
from nndeploy.server.audio import read_audio, write_audio
from nndeploy.server.config import FieldConfig
from test_arrays import array_graph, tensor_graph, echo
from test_db import db
from test_redis_queue import backend, redis_service
from test_storage import s3_service
from test_auth import credentials


def field(mime='audio/wav', rate=16000, channels=1, **kwargs):
    return dict(type='array', dtype='float32', shape=[None] if channels == 1 else [None, channels],
                media_type=mime, sample_rate=rate, **kwargs)


def settings(kind='array', graph=False, mime='audio/wav', rate=16000, channels=1, **execution):
    value = dict(field(mime, rate, channels), type=kind)
    return {'inputs': {'value': dict(value, **({'edge': 'input'} if graph else {}))},
            'outputs': {'value': dict(value, **({'edge': 'output'} if graph else {}))}, 'execution': execution}


def pcm(samples, rate=16000):
    samples = np.asarray(samples, dtype='<i2')
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as output:
        output.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())
    return stream.getvalue()


def upload(client, data, mime='audio/wav'):
    result = client.post('/v1/files', files={'file': ('input', data, mime)})
    assert result.status_code == 201, result.text
    return {'file_id': result.json()['file_id']}


def download(client, reference, tmp_path, *, channels=1, mime='audio/wav'):
    result = client.get('/v1/files/' + reference['file_id'] + '/content')
    assert result.status_code == 200 and result.headers['content-type'].startswith(mime)
    path = tmp_path / 'decoded-source'
    path.write_bytes(result.content)
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                      check=True, capture_output=True, timeout=10).stdout)
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le', 'pipe:1'],
                         check=True, capture_output=True, timeout=10).stdout
    array = np.frombuffer(raw, dtype='<f4').reshape(-1, channels)
    return array.reshape(-1) if channels == 1 else array, info['streams'][0]


@pytest.mark.parametrize('kind,graph', [('array', False), ('array', True), ('tensor', True)])
@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_audio_samples_code_and_graph_keep_original_files_and_results(db, tmp_path, kind, graph, isolation):
    options = {'graph_factory': tensor_graph if kind == 'tensor' else array_graph} if graph else {'deploy': echo}
    server = Server(**options, config=settings(kind, graph, isolation=isolation), resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        outputs = []
        for sign in (1, -1):
            key = uuid.uuid4().hex
            samples = np.arange(-800, 800, dtype='int16') * sign
            data = pcm(samples)
            source = upload(client, data)
            result = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}},
                                 headers={'Idempotency-Key': key})
            assert result.status_code == 200, result.text
            target = result.json()['outputs']['value']
            actual, info = download(client, target, tmp_path)
            np.testing.assert_array_equal(actual, samples.astype('float32') / 32768)
            assert info['sample_rate'] == '16000' and info['channels'] == 1
            assert client.get('/v1/files/' + source['file_id'] + '/content').content == data
            assert db.get_task(result.json()['task_id'])['inputs']['value'] == source
            again = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}},
                                headers={'Idempotency-Key': key})
            assert again.json()['task_id'] == result.json()['task_id'] and again.json()['outputs'] == result.json()['outputs']
            outputs.append(target)
        assert outputs[0] != outputs[1]
        np.testing.assert_array_equal(download(client, outputs[0], tmp_path)[0], np.arange(-800, 800, dtype='float32') / 32768)


def test_audio_resampling_and_channel_mix_have_declared_shape(tmp_path):
    source = np.column_stack((np.full(800, 8192, dtype='int16'), np.full(800, -8192, dtype='int16')))
    server = Server(graph_factory=array_graph, config=settings(graph=True), resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        result = client.post('/v1/server/default/deploy', json={'inputs': {'value': upload(client, pcm(source, rate=8000))}})
        assert result.status_code == 200, result.text
        actual, info = download(client, result.json()['outputs']['value'], tmp_path)
        assert actual.shape == (1600,) and info['channels'] == 1 and info['sample_rate'] == '16000'
        np.testing.assert_allclose(actual, 0, atol=1e-7)


@pytest.mark.parametrize('mime', ['audio/wav', 'audio/flac', 'audio/mpeg', 'audio/ogg', 'audio/mp4'])
def test_audio_encoders_produce_independently_decodable_files(tmp_path, mime):
    config = settings(mime=mime)
    config['inputs']['value']['media_type'] = 'audio/wav'
    server = Server(deploy=echo, config=config, resources=tmp_path, data_dir=tmp_path)
    samples = (np.sin(np.arange(8000) * .05) * 4096).astype('int16')
    with TestClient(server.app) as client:
        result = client.post('/v1/server/default/deploy', json={'inputs': {'value': upload(client, pcm(samples))}})
        assert result.status_code == 200, result.text
        actual, info = download(client, result.json()['outputs']['value'], tmp_path, mime=mime)
        assert info['codec_type'] == 'audio' and info['channels'] == 1
        # Opus always decodes at 48 kHz; the algorithm's rate remains 16 kHz.
        encoded = client.get('/v1/files/' + result.json()['outputs']['value']['file_id'] + '/content').content
        normalized = read_audio(io.BytesIO(encoded), FieldConfig(**field(mime)))
        assert abs(len(normalized) - len(samples)) <= 1024
        assert np.mean(np.abs(normalized[:len(samples)])) > .04
        other = Server(deploy=echo, config=settings(mime=mime), resources=tmp_path / 'other', data_dir=tmp_path / 'other')
        with TestClient(other.app) as second:
            converted = second.post('/v1/server/default/deploy', json={'inputs': {'value': upload(second, encoded, mime)}})
            assert converted.status_code == 200, converted.text


@pytest.mark.parametrize('change', [dict(sample_rate=None), dict(sample_rate=0), dict(dtype='int16'),
    dict(shape=[]), dict(shape=[None, None]), dict(shape=[None, 9]), dict(shape=[None, 2, 3]),
    dict(shape=[0]), dict(type='file'), dict(media_type='application/x-npy')])
def test_audio_field_requires_explicit_valid_numeric_metadata(change):
    with pytest.raises(ValidationError):
        FieldConfig(**dict(field(), **change))


def test_audio_limits_and_bad_wav_fail_before_task_admission(tmp_path):
    config = settings()
    config['inputs']['value']['max_elements'] = 100
    server = Server(deploy=echo, config=config, resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        source = upload(client, pcm(np.zeros(101, dtype='int16')))
        result = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert result.status_code == 422 and result.json()['error']['code'] == 'invalid_file'
        assert server.inflight == 0
    good = pcm(np.zeros(100, dtype='int16'))
    for bad in (good[:-2], good + b'extra', b'not a WAV'):
        with pytest.raises(ValueError):
            read_audio(io.BytesIO(bad), FieldConfig(**field()))


def test_audio_nonfinite_and_encoded_byte_limit_are_not_published(tmp_path):
    schema = FieldConfig(**field())
    with pytest.raises(ValueError, match='finite'):
        write_audio(io.BytesIO(), np.array([np.nan], dtype='float32'), schema)
    with pytest.raises(ValueError, match='byte limit'):
        write_audio(io.BytesIO(), np.zeros(1000, dtype='float32'), schema, max_bytes=100)
    values = np.zeros(1000, dtype='float32')
    stream = io.BytesIO()
    write_audio(stream, values, schema)
    data = bytearray(stream.getvalue())
    offset = data.index(b'data') + 8
    data[offset:offset + 4] = np.float32(np.nan).tobytes()
    with pytest.raises(ValueError, match='finite'):
        read_audio(io.BytesIO(data), schema)


def test_audio_failure_does_not_leave_output_and_next_request_succeeds(tmp_path):
    def bad_once(values):
        value = values['value']
        if value[0] > 0:
            value[0] = np.nan
        return {'value': value}
    server = Server(deploy=bad_once, config=settings(), resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        first = upload(client, pcm(np.ones(800, dtype='int16')))
        failed = client.post('/v1/server/default/deploy', json={'inputs': {'value': first}})
        assert failed.status_code == 500
        second = upload(client, pcm(-np.ones(800, dtype='int16')))
        succeeded = client.post('/v1/server/default/deploy', json={'inputs': {'value': second}})
        assert succeeded.status_code == 200, succeeded.text
        np.testing.assert_array_equal(download(client, succeeded.json()['outputs']['value'], tmp_path)[0],
                                      np.full(800, -1 / 32768, dtype='float32'))


def test_non_audio_schema_fingerprints_do_not_gain_sample_rate():
    assert 'sample_rate' not in FieldConfig(type='string').model_dump()
    assert 'sample_rate' not in FieldConfig(type='array', dtype='float32', shape=[None]).model_dump()


def test_audio_shared_process_restores_input_and_output_from_minio(backend, s3_service, tmp_path, monkeypatch):
    import uuid
    database, redis_config = backend
    endpoint, environment, _ = s3_service
    monkeypatch.setenv('NNDEPLOY_AUDIO_ACCESS', environment['MINIO_ROOT_USER'])
    monkeypatch.setenv('NNDEPLOY_AUDIO_SECRET', environment['MINIO_ROOT_PASSWORD'])
    storage = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
                   prefix='audio-' + uuid.uuid4().hex, access_key_env='NNDEPLOY_AUDIO_ACCESS', secret_key_env='NNDEPLOY_AUDIO_SECRET')
    server = Server(graph_factory=tensor_graph, config=settings('tensor', True, isolation='process'), resources=tmp_path, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config, storage_config=storage)
    with TestClient(server.app) as client:
        expected = np.arange(-800, 800, dtype='int16')
        source = upload(client, pcm(expected))
        server.storage.resolve(source['file_id'])[0].unlink()
        result = client.post('/v1/server/default/deploy', json={'inputs': {'value': source}})
        assert result.status_code == 200, result.text
        assert database.get_task(result.json()['task_id'])['worker_token']
        target = result.json()['outputs']['value']
        server.storage.resolve(target['file_id'])[0].unlink()
        np.testing.assert_array_equal(download(client, target, tmp_path)[0], expected.astype('float32') / 32768)


class ReusingAudioEngine:
    def __init__(self):
        self.buffer = np.zeros(800, dtype='float32')

    async def deploy(self, values):
        np.copyto(self.buffer, values['value'])
        return {'value': self.buffer}


@pytest.mark.parametrize('cancel', [False, True])
def test_audio_engine_transcode_is_detached_nonblocking_and_joined_on_cancel(tmp_path, monkeypatch, cancel):
    import threading
    import nndeploy.server.audio as audio
    from test_engine import until
    server = Server(handler_factory=ReusingAudioEngine,
                    config=settings(scheduling='engine', max_concurrency_per_worker=2, cooperative_cancel=True), resources=tmp_path, data_dir=tmp_path)
    sources = [server.storage.save(io.BytesIO(pcm(np.full(800, value, dtype='int16'))), 'audio/wav') for value in (4096, -4096)]
    started, release = threading.Event(), threading.Event()
    original = audio.write_audio
    def delayed(*args, **kwargs):
        if not started.is_set():
            started.set()
            assert release.wait(15)
        return original(*args, **kwargs)
    monkeypatch.setattr(audio, 'write_audio', delayed)
    async def scenario():
        await server.start()
        try:
            first, future = await server.submit('default', {'value': {'file_id': sources[0]['file_id']}})
            await until(started.is_set)
            second = await asyncio.wait_for(server.deploy('default', {'value': {'file_id': sources[1]['file_id']}}), 5)
            with server.storage.open(second['outputs']['value']['file_id']) as stream:
                np.testing.assert_array_equal(read_audio(stream, FieldConfig(**field())), np.full(800, -.125))
            if cancel:
                cancellation = asyncio.create_task(server.cancel_task(first))
                await until(lambda: server.db.get_task(first)['cancel_requested'])
                await asyncio.sleep(.05)
                assert not future.done() and server.db.get_task(first)['state'] == 'running'
                release.set()
                await cancellation
                with pytest.raises(Exception):
                    await future
                assert server.db.get_task(first)['state'] == 'cancelled'
            else:
                release.set()
                result = await future
                with server.storage.open(result['outputs']['value']['file_id']) as stream:
                    np.testing.assert_array_equal(read_audio(stream, FieldConfig(**field())), np.full(800, .125))
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('scheduling', ['serial', 'concurrent', 'engine', 'batch'])
def test_audio_warmup_and_schedulers_share_file_conversion(tmp_path, scheduling):
    from test_warmup import counts
    from test_arrays import NumericBatch
    server = Server(resources=tmp_path, data_dir=tmp_path)
    source = server.storage.save(io.BytesIO(pcm(np.zeros(800, dtype='int16'))), 'audio/wav')
    execution = {'batching': {'enabled': True}} if scheduling == 'batch' else {'scheduling': scheduling}
    if scheduling in ('concurrent', 'engine'):
        execution['max_concurrency_per_worker'] = 2
    execution['warmup'] = [{'inputs': {'value': {'file_id': source['file_id']}}}]
    options = {'handler_factory': NumericBatch} if scheduling == 'batch' else {
        'handler_factory': ReusingAudioEngine} if scheduling == 'engine' else {'deploy': echo}
    server.add(config=settings(**execution), **options)
    with TestClient(server.app) as client:
        assert counts(server) == [0, 1, 0]
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': {'file_id': source['file_id']}}})
        assert response.status_code == 200, response.text
        np.testing.assert_array_equal(download(client, response.json()['outputs']['value'], tmp_path)[0],
                                      np.full(800, 1 if scheduling == 'batch' else 0))


def test_audio_codec_configuration_is_checked_before_algorithm_init(tmp_path):
    called = []
    def factory():
        called.append(True)
        return ReusingAudioEngine()
    with pytest.raises(ValueError, match='sample_rate'):
        with TestClient(Server(handler_factory=factory, config=settings(mime='audio/ogg', rate=44100), resources=tmp_path, data_dir=tmp_path).app):
            pass
    with pytest.raises(ValueError, match='mono or stereo'):
        with TestClient(Server(handler_factory=factory, config=settings(mime='audio/mpeg', channels=3), resources=tmp_path, data_dir=tmp_path).app):
            pass
    assert called == []


@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('editor', [False, True])
def test_audio_workflow_json_and_sidecar_use_existing_graph_without_rewriting(tmp_path, isolation, editor):
    from pathlib import Path
    from types import SimpleNamespace
    import shutil
    from demo.server import audio
    from demo.server.audio import build_graph, config
    plugins = tmp_path / ('editor' if editor else 'server') / 'plugin'
    plugins.mkdir(parents=True)
    shutil.copyfile(Path(audio.__file__), plugins / 'audio_gain.py')
    workflow = tmp_path / 'workflow.json'
    original = build_graph().serialize()
    workflow.write_text(original)
    sidecar = workflow.with_suffix('.server.json')
    saved = json.dumps(config(isolation))
    sidecar.write_text(saved)
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True)) if editor else Server(resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    server.add(workflow=workflow)
    with TestClient(server.app) as client:
        samples = np.arange(-800, 800, dtype='int16')
        result = client.post('/v1/server/audio/deploy', json={'inputs': {'audio': upload(client, pcm(samples))}})
        assert result.status_code == 200, result.text
        np.testing.assert_array_equal(download(client, result.json()['outputs']['audio'], tmp_path)[0],
                                      samples.astype('float32') / 65536)
    assert workflow.read_text() == original and sidecar.read_text() == saved


@pytest.mark.parametrize('scheduling', ['serial', 'engine'])
def test_audio_stream_events_keep_independent_files_after_buffer_reuse(tmp_path, scheduling):
    from test_array_files import array_stream
    from test_streaming import parse_events
    schema = settings(scheduling=scheduling)
    schema['streaming'] = {'enabled': True}
    with TestClient(Server(deploy=array_stream, config=schema, resources=tmp_path, data_dir=tmp_path).app) as client:
        source = upload(client, pcm(np.full(800, 4096, dtype='int16')))
        result = client.post('/v1/server/default/deploy', json={'mode': 'stream', 'inputs': {'value': source}})
        assert result.status_code == 200, result.text
        events = parse_events(result.text)
        assert events[-1]['data']['state'] == 'succeeded', result.text
        outputs = [event['data']['data']['value'] for event in events if event['kind'] == 'delta']
        assert len(outputs) == 2 and outputs[0] != outputs[1]
        np.testing.assert_array_equal(download(client, outputs[0], tmp_path)[0], np.full(800, .125))
        np.testing.assert_array_equal(download(client, outputs[1], tmp_path)[0], np.zeros(800))


def test_audio_ownership_checked_before_decoding(tmp_path, credentials, monkeypatch):
    import nndeploy.server.arrays as arrays
    auth, headers = credentials
    server = Server(name='echo', deploy=echo, config=settings(), resources=tmp_path, data_dir=tmp_path, auth_config=auth)
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('input.wav', pcm(np.zeros(800, dtype='int16')), 'audio/wav')},
                               headers=headers['alice'])
        assert uploaded.status_code == 201, uploaded.text
        def unexpected(*args, **kwargs):
            pytest.fail('Another caller must not decode this audio')
        monkeypatch.setattr(arrays, 'read_file', unexpected)
        rejected = client.post('/v1/server/echo/deploy', json={'inputs': {'value': {'file_id': uploaded.json()['file_id']}}},
                               headers=headers['bob'])
        assert rejected.status_code == 422, rejected.text


def test_audio_output_retention_releases_content_after_task_expiry(db, tmp_path):
    from test_file_cleanup import expire as expire_file
    from test_retention import expire as expire_task
    server = Server(deploy=echo, config=settings(), resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start()
        try:
            source = server.storage.save(io.BytesIO(pcm(np.zeros(800, dtype='int16'))), 'audio/wav')
            result = await server.deploy('default', {'value': {'file_id': source['file_id']}})
            target = result['outputs']['value']['file_id']
            expire_file(db, target)
            assert (await server.cleanup_files())['deleted'] == 0
            with server.storage.open(target) as stream:
                assert read_audio(stream, FieldConfig(**field())).shape == (800,)
            expire_task(db, result['task_id'])
            await server.cleanup_tasks()
            assert (await server.cleanup_files())['deleted'] == 1
            assert db.get_file(target) is None
        finally:
            await server.close()
    asyncio.run(scenario())

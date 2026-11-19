"""Public launcher owns one runtime for single or combined video services."""
import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nndeploy.server import Server
from nndeploy.server.app import cli, main
from nndeploy.server.services import ServicesConfig, load_services, register_services


class FakeVideoHandler:
    reports_progress = True
    service_status = {'normal': True, 'dmd': True}

    def __init__(self, *args, **kwargs):
        pass

    def init(self):
        pass

    def deinit(self):
        pass

    def deploy(self, inputs):
        from nndeploy.server.request_context import current_request, FileArtifact
        request = current_request()
        request.publish({'progress': 50, 'stage': 'inference'})
        path = request.directory / 'output.mp4'
        shutil.copyfile(inputs['request']['video_input_path'], path)
        return {'artifacts': [{'file': FileArtifact(path, 'video/mp4')}],
                'size': '32x32', 'seconds': '1.0'}


def fake_videoedit_factory(*args, **kwargs):
    return FakeVideoHandler


def definition(algorithm):
    options = {'params': {'model': 'wan_vsr', 'model_path': '/unused/vsr', 'wan_root': '/unused/wan'}} if algorithm == 'vsr' else {
        'models': {'normal': {'model_path': '/unused/videoedit'}}}
    return {'algorithm': algorithm, 'name': algorithm, 'options': options}


@pytest.mark.parametrize('arguments', [
    ['--services', 'services.json'], ['--data-dir', 'state'],
    ['--api-only', '--services', 'services.json', '--workflow', 'graph.json'],
    ['--api-only', '--services', 'services.json', '--config', 'config.json'],
    ['--api-only', '--services', 'services.json', '--name', 'wrong'],
    ['--database-backup', 'backup', '--database-url-env', 'DB', '--services', 'services.json'],
])
def test_service_cli_rejects_ambiguous_lifecycle_options(arguments):
    with pytest.raises(SystemExit):
        cli(arguments)


@pytest.mark.parametrize('config', [
    {'services': []},
    {'services': [definition('vsr')], 'data_dir': 'nested'},
    {'services': [definition('vsr'), definition('vsr')]},
    {'services': [definition('vsr'), {**definition('videoedit'), 'name': 'vsr'}]},
    {'services': [{**definition('vsr'), 'algorithm': 'arbitrary.module:factory'}]},
    {'services': [{**definition('vsr'), 'name': '../bad'}]},
])
def test_service_manifest_validation(config):
    with pytest.raises(ValidationError):
        ServicesConfig.model_validate(config)


@pytest.mark.parametrize('options', [
    {'params': {}},
    {'params': {'model': 'wan_vsr'}, 'data_dir': 'nested'},
    {'params': {'model': 'wan_vsr'}, 'handler_factory': 'module:factory'},
])
def test_video_options_are_validated_by_the_algorithm_before_installation(tmp_path, options):
    instance = Server(resources=tmp_path)
    try:
        manifest = {'services': [{**definition('vsr'), 'options': options}]}
        ServicesConfig.model_validate(manifest)  # The manifest is algorithm-independent.
        with pytest.raises(ValidationError):
            register_services(instance, manifest)
        assert not instance.entries
    finally:
        asyncio.run(instance.close())


@pytest.mark.parametrize('algorithms', [('vsr',), ('videoedit',), ('vsr', 'videoedit')])
def test_common_launcher_uses_one_database_and_lifecycle(tmp_path, monkeypatch, algorithms):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and FFprobe are required for video artifact validation')
    from nndeploy.server import app
    from nndeploy.e_dit.algorithms.vsr.server import app as vsr_app
    from nndeploy.e_dit.algorithms.wan.server import service as wan_service
    monkeypatch.setattr(vsr_app, 'VSRHandler', FakeVideoHandler)
    monkeypatch.setattr(wan_service, 'HandlerFactory', fake_videoedit_factory)
    manifest = tmp_path / 'services.json'
    manifest.write_text(json.dumps({'services': [definition(algorithm) for algorithm in algorithms]}))
    root = tmp_path / 'runtime'
    created = []
    def construct(**kwargs):
        instance = Server(**kwargs)
        created.append(instance)
        return instance
    monkeypatch.setattr(app, 'NnDeployServer', construct)
    def run(instance, *, host, port):
        assert host == '127.0.0.1' and port == 8123
        assert set(instance.entries) == set(algorithms)
        assert instance.storage.db is instance.db
        assert all(adapter.tasks.server is instance for adapter in instance.app.state.video_resources.adapters.values())
        with TestClient(instance.app) as client:
            assert client.get('/health/ready').status_code == 200
            for worker in (worker for entry in instance.entries.values() for worker in entry['workers']):
                assert worker.storage_options['database_url'] == instance.db.url.render_as_string(hide_password=False)
            source = tmp_path / 'input.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=32x32:rate=5:duration=1', '-c:v', 'libx264', str(source)], check=True)
            for algorithm in algorithms:
                path = '/v1/videos/restorations' if algorithm == 'vsr' else '/v1/videos/repairs'
                body = {'taskId': 'shared-id', 'video_input_path': str(source)}
                if algorithm == 'vsr':
                    body['target_resolution'] = '32x32'
                else:
                    body.update(prompt='edit', mask_input_path=str(source), reference_image_path=str(source))
                response = client.post(path, json=body)
                assert response.status_code == 200, response.text
                assert response.json()['code'] == 0, response.text
                for _ in range(300):
                    job = client.get('/v1/videos/shared-id', params={'entry': algorithm}).json()
                    if job['status'] in ('completed', 'failed'):
                        break
                    time.sleep(.01)
                assert job['status'] == 'completed', job
                assert client.get('/v1/videos/shared-id/content', params={'entry': algorithm}).content == source.read_bytes()
                assert instance.db.external_task(algorithm, 'shared-id', None, instance.config_scope)['state'] == 'succeeded'
            assert len(client.get('/v1/videos').json()['data']) == len(algorithms)
            if len(algorithms) == 2:
                assert client.get('/v1/videos/shared-id').status_code == 409
        assert all(not entry['workers'] for entry in instance.entries.values())
    monkeypatch.setattr(Server, 'run', run)
    arguments = ['--api-only', '--services', str(manifest), '--resources', str(root),
                 '--data-dir', str(tmp_path / 'files-work'), '--port', '8123']
    # An explicit shared SQL URL replaces the default local database.
    database = root / 'db' / 'nndeploy.db'
    if len(algorithms) == 2:
        database = tmp_path / 'external.db'
        monkeypatch.setenv('NNDEPLOY_TEST_SERVICE_DB', f'sqlite:///{database}')
        arguments += ['--database-url-env', 'NNDEPLOY_TEST_SERVICE_DB']
    main(arguments)
    assert len(created) == 1
    assert list(tmp_path.rglob('*.db')) == [database]


def test_invalid_manifest_does_not_create_a_server(tmp_path, monkeypatch):
    from nndeploy.server import app
    path = tmp_path / 'services.json'
    path.write_text('{"services": []}')
    monkeypatch.setattr(app, 'NnDeployServer', lambda **kwargs: pytest.fail('Server created before validation'))
    with pytest.raises(ValidationError):
        main(['--api-only', '--services', str(path), '--data-dir', str(tmp_path / 'unused')])
    assert not (tmp_path / 'unused').exists()


def test_registration_requires_unstarted_runtime(tmp_path):
    instance = Server(resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        try:
            await instance.start()
            with pytest.raises(ValueError, match='before starting'):
                register_services(instance, {'services': [definition('vsr')]})
        finally:
            await instance.close()
    asyncio.run(scenario())


def test_services_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / 'services.json'
    path.write_text('{"services": [], "services": []}')
    with pytest.raises(ValueError, match='Duplicate'):
        load_services(path)


@pytest.mark.parametrize('preceding', [[], ['videoedit']])
def test_registration_error_closes_unstarted_common_runtime(tmp_path, monkeypatch, preceding):
    from nndeploy.server import app
    created = []
    def construct(**kwargs):
        instance = Server(**kwargs)
        created.append(instance)
        return instance
    monkeypatch.setattr(app, 'NnDeployServer', construct)
    monkeypatch.setattr(Server, 'run', lambda *args, **kwargs: pytest.fail('Invalid service started'))
    invalid = definition('vsr')
    invalid['options']['execution'] = {'workers': 2}
    manifest = tmp_path / 'services.json'
    manifest.write_text(json.dumps({'services': [definition(name) for name in preceding] + [invalid]}))
    with pytest.raises(ValueError, match='one serial worker'):
        main(['--api-only', '--services', str(manifest), '--resources', str(tmp_path / 'resources')])
    assert len(created) == 1
    assert created[0].shutting_down
    assert created[0].closing_task.done()
    assert not created[0].started
    assert all(not entry['workers'] for entry in created[0].entries.values())

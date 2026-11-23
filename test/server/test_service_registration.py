"""Manifest registration uses the same trusted factories as the Python API."""
import asyncio
import json
import time

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from nndeploy.server import Server, ServiceAdapter
from nndeploy.server.app import cli, main
from nndeploy.server.services import register_services
from test_auth import credentials


TEXT_FACTORY = 'demo.server.text.server:create_entry'


def text_manifest():
    return {'services': [
        {'algorithm': 'text', 'name': 'upper', 'options': {'uppercase': True}},
        {'algorithm': 'text', 'name': 'lower', 'options': {'uppercase': False}},
    ]}


def test_cli_registers_two_instances_with_one_runtime(tmp_path, monkeypatch):
    from nndeploy.server import app
    created = []

    def construct(**kwargs):
        server = Server(**kwargs)
        created.append(server)
        return server

    def run(server, **kwargs):
        assert set(server.entries) == {'upper', 'lower'}
        assert all(not entry['workers'] for entry in server.entries.values())
        with TestClient(server.app) as client:
            assert client.get('/health/ready').status_code == 200
            paths = client.get('/openapi.json').json()['paths']
            for name, expected in [('upper', 'HELLO'), ('lower', 'hello')]:
                assert f'/v1/server/{name}/text' in paths
                headers = {'Idempotency-Key': 'same-key'}
                result = client.post(f'/v1/server/{name}/text',
                    json={'text': 'HeLLo'}, headers=headers)
                assert result.status_code == 200, result.text
                assert result.json()['text'] == expected
                standard = client.post(f'/v1/server/{name}/deploy',
                    json={'inputs': {'text': 'HeLLo'}}, headers=headers)
                assert standard.json()['task_id'] == result.json()['id']
                pending = client.post(f'/v1/server/{name}/text',
                    json={'text': 'HeLLo', 'mode': 'async'})
                assert pending.status_code == 202, pending.text
                for _ in range(300):
                    record = client.get(pending.json()['status_url']).json()
                    if record['state'] in ('succeeded', 'failed'):
                        break
                    time.sleep(.01)
                assert record['state'] == 'succeeded', record
                assert record['outputs']['text'] == expected
        assert all(not entry['workers'] for entry in server.entries.values())

    path = tmp_path / 'services.json'
    path.write_text(json.dumps(text_manifest()))
    monkeypatch.setattr(app, 'NnDeployServer', construct)
    monkeypatch.setattr(Server, 'run', run)
    main(['--api-only', '--services', str(path), '--resources', str(tmp_path / 'runtime'),
          '--register-algorithm', f'text={TEXT_FACTORY}'])
    assert len(created) == 1
    assert len(list(tmp_path.rglob('*.db'))) == 1


def test_manifest_instances_preserve_entry_permissions(tmp_path, credentials):
    auth, headers = credentials
    server = Server(resources=tmp_path, auth_config=auth)
    server.register('text', TEXT_FACTORY)
    manifest = text_manifest()
    manifest['services'][0]['name'] = 'echo'  # Existing test credentials grant Alice this entry.
    manifest['services'][1]['name'] = 'private'
    register_services(server, manifest)
    with TestClient(server.app) as client:
        assert client.post('/v1/server/echo/text', json={'text': 'hi'}).status_code == 401
        assert client.post('/v1/server/private/text', json={'text': 'hi'}, headers=headers['alice']).status_code == 403
        result = client.post('/v1/server/echo/text', json={'text': 'hi'}, headers=headers['alice'])
        assert result.status_code == 200, result.text
        identity = result.json()['id']
        assert client.get('/v1/tasks/' + identity, headers=headers['bob']).status_code == 404


@pytest.mark.parametrize('specifications', [
    ['text'], ['bad/name=module:create'], ['text=os.system'],
    ['text=module:create', 'text=other:create'],
])
def test_invalid_cli_registrations_fail_before_import(specifications):
    args = ['--api-only', '--services', 'unused.json']
    for value in specifications:
        args += ['--register-algorithm', value]
    with pytest.raises(SystemExit):
        cli(args)


@pytest.mark.parametrize('args', [[], ['--api-only', '--workflow', 'graph.json']])
def test_cli_registration_requires_manifest(args):
    with pytest.raises(SystemExit):
        cli([*args, '--register-algorithm', f'text={TEXT_FACTORY}'])


def echo(inputs):
    return inputs


@pytest.mark.parametrize('failure', ['unknown', 'options', 'config', 'route', 'source'])
def test_manifest_preflight_does_not_install_earlier_entries(tmp_path, failure):
    server = Server(resources=tmp_path)
    calls = []

    def factory(**options):
        calls.append(options)
        if options.get('bad_options'):
            raise ValueError('Invalid options')
        if options.get('bad_source'):
            return {'deploy': echo, 'handler_factory': echo}
        router = APIRouter()
        router.add_api_route('/text', echo, methods=['POST'])
        return {'deploy': echo, 'router': router, 'router_prefix': '/common',
                'config': {'execution': {'workers': 0}} if options.get('bad_config') else {}}

    server.register('echo', factory)
    options = {'options': 'bad_options', 'config': 'bad_config', 'source': 'bad_source'}
    second = {'algorithm': 'missing' if failure == 'unknown' else 'echo', 'name': 'second',
              'options': {options[failure]: True} if failure in options else {}}
    before = list(server.app.routes)
    try:
        with pytest.raises(ValueError):
            register_services(server, {'services': [{'algorithm': 'echo', 'name': 'first'}, second]})
        assert not server.entries
        assert server.app.routes == before
        if failure == 'unknown':
            assert not calls
    finally:
        asyncio.run(server.close())


@pytest.mark.parametrize('algorithm', ['vsr', 'videoedit'])
def test_fixed_video_routes_reject_multiple_instances_before_install(tmp_path, algorithm):
    from test_services import definition
    server = Server(resources=tmp_path)
    try:
        with pytest.raises(ValueError, match='already registered'):
            register_services(server, {'services': [definition(algorithm),
                {**definition(algorithm), 'name': 'second'}]})
        assert not server.entries
        assert not (server.data_dir / 'request-secrets').exists()
    finally:
        asyncio.run(server.close())


def test_adapter_factory_works_through_python_and_manifest(tmp_path):
    def factory(**options):
        def install(server, *, name):
            server.add(name=name, deploy=echo)
        return ServiceAdapter(install, exclusive_key='shared-routes')

    server = Server(resources=tmp_path)
    server.register('custom', factory)
    server.add(name='first', algorithm='custom')
    try:
        with pytest.raises(ValueError, match='already registered'):
            register_services(server, {'services': [{'algorithm': 'custom', 'name': 'second'}]})
        assert set(server.entries) == {'first'}
        with pytest.raises(ValueError, match='algorithm options'):
            server.add(name='third', algorithm='custom', workers=2)
    finally:
        asyncio.run(server.close())


def test_cli_closes_runtime_after_adapter_install_failure(tmp_path, monkeypatch):
    from nndeploy.server import app
    created = []

    def construct(**kwargs):
        server = Server(**kwargs)
        def factory():
            def install(server, *, name):
                server.add(name=name, deploy=echo)
                raise ValueError('Adapter installation failed')
            return ServiceAdapter(install)
        server.register('broken', factory)
        created.append(server)
        return server

    path = tmp_path / 'services.json'
    path.write_text(json.dumps({'services': [{'algorithm': 'broken', 'name': 'broken'}]}))
    monkeypatch.setattr(app, 'NnDeployServer', construct)
    monkeypatch.setattr(Server, 'run', lambda *args, **kwargs: pytest.fail('Invalid server started'))
    with pytest.raises(ValueError, match='installation failed'):
        main(['--api-only', '--services', str(path), '--resources', str(tmp_path / 'runtime')])
    assert created[0].shutting_down and created[0].closing_task.done()
    assert all(not entry['workers'] for entry in created[0].entries.values())


def test_manifest_cli_serves_real_http_and_shuts_down(tmp_path):
    import signal
    import subprocess
    import sys
    import requests
    from test_cli import unused_port

    path = tmp_path / 'services.json'
    path.write_text(json.dumps(text_manifest()))
    port = unused_port()
    address = f'http://127.0.0.1:{port}'
    http = requests.Session()
    http.trust_env = False  # Loopback integration traffic must not use ambient proxies.
    log_path = tmp_path / 'server.log'
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'nndeploy.server', '--api-only',
            '--services', str(path), '--register-algorithm', f'text={TEXT_FACTORY}',
            '--resources', str(tmp_path / 'runtime'), '--port', str(port)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text()
                try:
                    if http.get(address + '/health/ready', timeout=.5).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.1)
            else:
                pytest.fail('CLI startup timed out: ' + log_path.read_text())
            for name, expected in [('upper', 'HELLO'), ('lower', 'hello')]:
                result = http.post(address + f'/v1/server/{name}/text', json={'text': 'Hello'}, timeout=5)
                assert result.status_code == 200, result.text
                assert result.json()['text'] == expected
            process.terminate()
            process.wait(timeout=20)
            # Uvicorn may re-raise the termination signal after graceful cleanup.
            assert process.returncode in (0, -signal.SIGTERM), log_path.read_text()
            assert 'Application shutdown complete.' in log_path.read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            http.close()


@pytest.mark.parametrize('kind', ['missing-entry', 'async-result'])
def test_invalid_adapter_installer_contract_is_rejected(tmp_path, kind):
    async def coroutine():
        pass

    def install(server, *, name):
        return coroutine() if kind == 'async-result' else None

    server = Server(resources=tmp_path)
    server.register('broken', lambda: ServiceAdapter(install))
    try:
        with pytest.raises(ValueError, match='synchronously|did not register'):
            server.add(name='broken', algorithm='broken')
        assert not server.entries
    finally:
        asyncio.run(server.close())


def test_unused_registration_is_not_imported_and_options_are_isolated(tmp_path):
    server = Server(resources=tmp_path)
    calls = []
    def factory(settings):
        calls.append(settings['value'])
        settings['value'] = 'mutated by factory'
        return {'deploy': echo}
    server.register('echo', factory)
    server.register('optional', 'nndeploy_missing_optional_package:create')
    options = {'settings': {'value': 'original'}}
    manifest = {'services': [{'algorithm': 'echo', 'name': name, 'options': options}
                             for name in ('one', 'two')]}
    try:
        register_services(server, manifest)
        assert calls == ['original', 'original']
        assert options == {'settings': {'value': 'original'}}
        assert set(server.entries) == {'one', 'two'}
    finally:
        asyncio.run(server.close())


@pytest.mark.parametrize('adapter_first', [True, False])
def test_adapter_install_cannot_shadow_prepared_routes(tmp_path, adapter_first):
    def factory():
        router = APIRouter()
        router.add_api_route('/text', echo, methods=['POST'])
        return {'deploy': echo, 'router': router, 'router_prefix': '/common'}

    def adapter_factory():
        def install(server, *, name):
            server.add(name=name, **factory())
        return ServiceAdapter(install)

    server = Server(resources=tmp_path)
    server.register('plain', factory)
    server.register('adapter', adapter_factory)
    names = ['adapter', 'plain'] if adapter_first else ['plain', 'adapter']
    try:
        with pytest.raises(ValueError, match='conflicts'):
            register_services(server, {'services': [{'algorithm': name, 'name': name} for name in names]})
        assert set(server.entries) == {names[0]}
        assert not any(entry['workers'] for entry in server.entries.values())
    finally:
        asyncio.run(server.close())

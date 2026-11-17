"""Optional local configuration is loaded before environment references resolve."""
import os

import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from nndeploy.server.environment import load_server_env


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Track loader mutations as well as explicit test changes for teardown.
    monkeypatch.setattr(os, 'environ', os.environ.copy())


def test_missing_file_does_not_search_parent_or_generic_dotenv(tmp_path):
    (tmp_path / 'server.env').write_text('TEST_SERVER_ENV_PARENT=unexpected\n')
    child = tmp_path / 'child'
    child.mkdir()
    (child / '.env').write_text('TEST_SERVER_ENV_GENERIC=unexpected\n')
    os.chdir(child)
    before = dict(os.environ)
    assert load_server_env() is False
    assert dict(os.environ) == before


def test_dotenv_syntax_preserves_literal_secrets(tmp_path):
    (tmp_path / 'server.env').write_text(
        '\ufeff# local configuration\n'
        'export TEST_ENV_SINGLE=\'secret $HOME ${HOME} # literal\'\n'
        'TEST_ENV_DOUBLE="quoted value" # comment\n'
        'TEST_ENV_EMPTY=\n'
        'TEST_ENV_BARE\n'
        'TEST_ENV_COMMAND=$(touch should-not-exist)\n', encoding='utf-8')
    assert load_server_env() is True
    assert os.environ['TEST_ENV_SINGLE'] == 'secret $HOME ${HOME} # literal'
    assert os.environ['TEST_ENV_DOUBLE'] == 'quoted value'
    assert os.environ['TEST_ENV_EMPTY'] == ''
    assert 'TEST_ENV_BARE' not in os.environ
    assert os.environ['TEST_ENV_COMMAND'] == '$(touch should-not-exist)'
    assert not (tmp_path / 'should-not-exist').exists()


def test_existing_environment_including_empty_values_wins(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_ENV_EXISTING', 'process-value')
    monkeypatch.setenv('TEST_ENV_EMPTY', '')
    path = tmp_path / 'server.env'
    path.write_text('TEST_ENV_EXISTING=file-value\nTEST_ENV_EMPTY=file-value\nTEST_ENV_NEW=first\n')
    load_server_env()
    path.write_text('TEST_ENV_NEW=second\n')
    load_server_env()
    assert os.environ['TEST_ENV_EXISTING'] == 'process-value'
    assert os.environ['TEST_ENV_EMPTY'] == ''
    assert os.environ['TEST_ENV_NEW'] == 'first'


def test_server_resolves_auth_from_local_file(tmp_path):
    (tmp_path / 'server.env').write_text('TEST_ENV_AUTH_TOKEN=local-test-token-with-at-least-32-characters\n')
    server = Server(data_dir=tmp_path / 'data', auth_config={
        'keys': [{'subject': 'local-user', 'token_env': 'TEST_ENV_AUTH_TOKEN'}]})
    with TestClient(server.app) as client:
        assert client.get('/v1/models').status_code == 401
        assert client.get('/v1/models', headers={
            'Authorization': 'Bearer local-test-token-with-at-least-32-characters'}).status_code == 200


@pytest.mark.parametrize('with_file', [False, True])
def test_cli_loads_before_resolving_environment_options(tmp_path, monkeypatch, with_file):
    from nndeploy.server import app

    captured = {}

    class FakeServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            captured['run'] = kwargs

    monkeypatch.setattr(app, 'NnDeployServer', FakeServer)
    monkeypatch.setattr(app.mp, 'set_start_method', lambda *a, **kw: None)
    arguments = ['--api-only', '--workflow', 'graph.json']
    if with_file:
        (tmp_path / 'server.env').write_text(
            'TEST_ENV_ADMIN=local-admin\nTEST_ENV_DB=sqlite:///local.sqlite\n')
        arguments += ['--admin-token-env', 'TEST_ENV_ADMIN', '--database-url-env', 'TEST_ENV_DB']
    app.main(arguments)
    assert captured['admin_token'] == ('local-admin' if with_file else None)
    assert captured['database_url'] == ('sqlite:///local.sqlite' if with_file else None)
    assert 'run' in captured

"""Regression coverage for repository launchers and maintenance commands."""
from pathlib import Path
import subprocess
import sys

import pytest

from nndeploy.server.app import cli
from nndeploy.server import clean
from nndeploy.dag import run_json
import nndeploy.base


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('script', ['app.py', 'server.py', 'clean.py', 'run_json.py'])
def test_repository_launcher_help(script):
    result = subprocess.run([sys.executable, str(ROOT / script), '--help'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


@pytest.mark.parametrize('arguments', [
    ['--port', '0'], ['--port', '65536'],
    ['--editor-history-retention-seconds', 'nan'],
    ['--editor-history-retention-seconds', '59'],
    ['--api-only', '--workflow', 'a.json', '--json_file', 'b.json'],
])
def test_server_rejects_invalid_options(arguments):
    with pytest.raises(SystemExit) as error:
        cli(arguments)
    assert error.value.code == 2


def test_cleanup_custom_resources_preview_and_explicit_deploy(tmp_path):
    resources = tmp_path / 'resources'
    for name in ('db', 'server'):
        (resources / name).mkdir(parents=True)
        (resources / name / 'data').write_text('preserve')
    assert clean.main(['--resources', str(resources), '--db', '--dry-run']) == 0
    assert (resources / 'db/data').exists()
    assert clean.main(['--resources', str(resources), '--db']) == 0
    assert not (resources / 'db').exists()
    assert (resources / 'server/data').exists()
    assert clean.main(['--resources', str(resources), '--server']) == 0
    assert not (resources / 'server').exists()


def test_cleanup_unlinks_symlink_without_deleting_target(tmp_path):
    target = tmp_path / 'target'
    target.mkdir()
    (target / 'keep').write_text('keep')
    link = tmp_path / 'logs'
    link.symlink_to(target, target_is_directory=True)
    assert clean.main(['--logs', '--logs-dir', str(link)]) == 0
    assert not link.is_symlink()
    assert (target / 'keep').exists()


def test_cleanup_failure_status_and_conflicting_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert clean.main(['--logs', '--logs-dir', '.']) == 1
    with pytest.raises(SystemExit):
        clean.main(['--db', '--keep', 'logs'])


def test_run_json_missing_input_returns_failure(tmp_path):
    assert run_json.main(['--json_file', str(tmp_path / 'missing.json')]) == 1


@pytest.mark.parametrize('success', [True, False])
def test_run_json_existing_resources_executes_and_checks_status(tmp_path, monkeypatch, success):
    monkeypatch.chdir(tmp_path)
    resources = tmp_path / 'resources'
    resources.mkdir()
    workflow = tmp_path / 'graph.json'
    workflow.write_text('{}')
    calls = []

    class Runner:
        def run(self, graph, name, task_id, args):
            calls.append(args.parallel_type)
            return {}, {}, nndeploy.base.StatusCode.Ok if success else object(), 'test status'

    monkeypatch.setattr(run_json, 'GraphRunner', Runner)
    result = run_json.main(['--json_file', str(workflow), '--resources', str(resources),
                            '--parallel_type', 'sequential'])
    assert result == (0 if success else 1)
    assert calls == ['Sequential']


def test_run_json_repeated_parse_replaces_lists():
    args = run_json.GraphRunnerArgs()
    args.parse_args(['--json_file', 'a.json', '--plugin', 'one.py'])
    args.parse_args(['--json_file', 'b.json', '--plugin', 'two.py'])
    assert args.plugin == ['two.py']


def test_parallel_override_reaches_graph_before_initialization():
    import json
    from test_server_api import text_graph
    from nndeploy.dag.graph_runner import GraphRunner

    args = run_json.GraphRunnerArgs()
    args.parallel_type = 'Task'
    observed = []

    class Checked(RuntimeError):
        pass

    def check(graph):
        observed.append(graph.get_parallel_type())
        raise Checked()

    with pytest.raises(Checked):
        GraphRunner().run(json.dumps(text_graph()), 'test', '', args, before_init=check)
    assert observed == [nndeploy.base.ParallelType.Task]


def test_run_json_real_workflow_process(tmp_path):
    import json
    from test_server_api import text_graph

    workflow = tmp_path / 'echo.json'
    workflow.write_text(json.dumps(text_graph()))
    result = subprocess.run([sys.executable, str(ROOT / 'run_json.py'),
                             '--json_file', str(workflow), '--parallel_type', 'sequential'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Graph execution completed successfully' in result.stderr

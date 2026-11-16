"""Real HTTP smoke test for two independently launched Server processes."""
import json
import socket
import subprocess
import sys
import time

import requests

from test_server_api import text_graph


def unused_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def test_q06_independent_cli_servers_do_not_share_tasks_or_shutdown(tmp_path):
    workflow = tmp_path / 'echo.json'
    workflow.write_text(json.dumps(text_graph()))
    processes, logs, addresses = [], [], []
    try:
        for index in range(2):
            port = unused_port()
            log = (tmp_path / f'server-{index}.log').open('w')
            logs.append(log)
            process = subprocess.Popen([sys.executable, '-m', 'nndeploy.server', '--api-only',
                '--workflow', str(workflow), '--port', str(port), '--resources', str(tmp_path / f'resources-{index}'),
                '--max-sync-connections', '2'],
                stdout=log, stderr=log)
            processes.append(process)
            addresses.append(f'http://127.0.0.1:{port}')
            for _ in range(200):
                if process.poll() is not None:
                    raise AssertionError((tmp_path / f'server-{index}.log').read_text())
                try:
                    if requests.get(addresses[-1] + '/health/ready', timeout=.2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.05)
            else:
                raise AssertionError('CLI Server did not become ready')
        first = requests.post(addresses[0] + '/v1/server/echo/deploy', json={'inputs': {'source': 'first'}}, timeout=5)
        assert first.status_code == 200, first.text
        assert first.json()['outputs'] == {'sink': 'first'}
        assert requests.get(addresses[1] + '/v1/tasks/' + first.json()['task_id'], timeout=5).status_code == 404
        processes[0].terminate()
        processes[0].wait(timeout=10)
        second = requests.post(addresses[1] + '/v1/server/echo/deploy', json={'inputs': {'source': 'still-running'}}, timeout=5)
        assert second.status_code == 200, second.text
        assert second.json()['outputs'] == {'sink': 'still-running'}
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def test_editor_history_retention_cli_option():
    from nndeploy.server.app import cli
    assert cli([]).editor_history_retention_seconds is None
    assert cli(['--editor-history-retention-seconds', '86400']).editor_history_retention_seconds == 86400

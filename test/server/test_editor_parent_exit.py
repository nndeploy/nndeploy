"""The editor's children must observe loss of their owning HTTP process."""
import json
import os
import signal
import subprocess
import sys

import pytest
import requests

from test_cli import unused_port
from test_editor_queue_recovery import LAUNCH, live, wait
from test_editor_task_control import HOLD_PLUGIN, held_graph
from test_server_api import text_graph


@pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX process signals')
@pytest.mark.parametrize('running', [False, True])
def test_editor_children_exit_after_main_process_is_killed(tmp_path, running):
    resources = tmp_path / 'resources'
    plugin = resources / 'plugin'
    plugin.mkdir(parents=True)
    (plugin / 'editor_hold.py').write_text(HOLD_PLUGIN)
    address = f'http://127.0.0.1:{unused_port()}'
    console = tmp_path / 'console.log'
    records = dict(managers=[], workers=[])
    with console.open('w') as log:
        command = [sys.executable, '-c', LAUNCH, '--host', '127.0.0.1',
            '--port', address.rsplit(':', 1)[1], '--debug', '--resources', str(resources),
            '--log', str(tmp_path / 'editor.log')]
        environment = dict(os.environ, NNDEPLOY_EDITOR_RECOVERY_TEST=str(tmp_path))
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        def ready():
            assert process.poll() is None, console.read_text()
            try:
                return requests.get(address + '/health/ready', timeout=.5).status_code == 200
            except requests.RequestException:
                return False
        try:
            wait(ready, 30)
            records = json.loads((tmp_path / 'pids.json').read_text())
            graph = text_graph()
            graph['node_repository_'][0]['str_'] = 'Persisted result 中文'
            response = requests.post(address + '/api/queue', json=graph, timeout=3)
            assert response.status_code == 202, response.text
            identity = response.json()['result']['task_id']
            def historical():
                response = requests.get(address + '/api/queue/history', timeout=3)
                assert response.status_code == 200, response.text
                return next((record for record in response.json()['result']['items']
                             if record['task_id'] == identity), None)
            completed = wait(historical)
            assert completed['state'] == 'SUCCEEDED'
            output = requests.get(address + '/api/queue/task/' + identity, timeout=3).json()['result']['outputs']
            assert output == {'sink': 'Persisted result 中文'}
            unfinished = None
            if running:
                response = requests.post(address + '/api/queue', json=held_graph(tmp_path / 'control'), timeout=3)
                assert response.status_code == 202, response.text
                unfinished = response.json()['result']['task_id']
                wait(lambda: (tmp_path / 'control' / 'entered').exists())
            process.kill()
            process.wait(timeout=5)
            wait(lambda: not any(live(pid) for group in records.values() for pid in group), 6)
            if running:
                assert (tmp_path / 'control' / 'entered').read_text() == 'run\n'
                assert not (tmp_path / 'control' / 'exited').exists(), 'The abandoned Graph must not finish later'
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
            wait(ready, 30)
            for kind, group in json.loads((tmp_path / 'pids.json').read_text()).items():
                records[kind].extend(group)
            assert historical() == completed
            restored = requests.get(address + '/api/queue/task/' + identity, timeout=3)
            assert restored.json()['result']['outputs'] == output
            if unfinished is not None:
                records_now = requests.get(address + '/api/queue/history', timeout=3).json()['result']['items']
                recovered = next(record for record in records_now if record['task_id'] == unfinished)
                assert recovered['state'] == 'FAILED'
                assert (tmp_path / 'control' / 'entered').read_text() == 'run\n'
            process.terminate()
            process.wait(timeout=15)
            wait(lambda: not any(live(pid) for group in records.values() for pid in group))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for group in records.values():
                for pid in group:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

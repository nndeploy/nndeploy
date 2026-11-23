"""Isolated Workers may own compute subprocesses, and must stop the full group."""
import multiprocessing
from pathlib import Path
import time

import psutil
from fastapi.testclient import TestClient
from nndeploy.server import Server


def _sleep_forever():
    while True:
        time.sleep(1)


class ChildHandler:
    def init(self):
        self.child = multiprocessing.get_context('spawn').Process(target=_sleep_forever)
        self.child.start()

    def deploy(self, inputs):
        return {'pid': self.child.pid}

    def close(self):
        # Intentionally leave cleanup to the Worker group supervisor.
        pass


def test_worker_closes_descendants(tmp_path):
    server = Server(handler_factory=ChildHandler, resources=tmp_path, data_dir=tmp_path,
                    config={'execution': {'isolation': 'process', 'allow_child_processes': True}})
    with TestClient(server.app) as client:
        response = client.post('/v1/server/default/deploy', json={})
        assert response.status_code == 200, response.text
        pid = response.json()['outputs']['pid']
        assert psutil.pid_exists(pid)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


class BlockingChildHandler(ChildHandler):
    def deploy(self, inputs):
        Path(inputs['started']).write_text(str(self.child.pid))
        if inputs.get('block', True):
            while True:
                time.sleep(.1)
        return {'pid': self.child.pid}


def test_cancel_stops_compute_group_and_next_request_restarts(tmp_path):
    started = tmp_path/'started'
    server = Server(handler_factory=BlockingChildHandler, resources=tmp_path/'server', data_dir=tmp_path/'server',
                    config={'execution': {'isolation': 'process', 'allow_child_processes': True},
                            'inputs': {'started': {'type': 'string'},
                                       'block': {'type': 'boolean', 'required': False, 'default': True}}})
    with TestClient(server.app) as client:
        response = client.post('/v1/server/default/deploy', json={
            'mode': 'async', 'inputs': {'started': str(started)}})
        assert response.status_code == 202, response.text
        identity = response.json()['task_id']
        deadline = time.monotonic()+30
        while not started.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert started.exists()
        child_pid = int(started.read_text())
        response = client.post(f'/v1/tasks/{identity}/cancel')
        assert response.status_code == 200, response.text
        while time.monotonic()<deadline:
            record = client.get(f'/v1/tasks/{identity}').json()
            if record.get('state') == 'cancelled': break
            time.sleep(.05)
        assert record['state'] == 'cancelled', record
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
        response = client.post('/v1/server/default/deploy', json={
            'inputs': {'started': str(started), 'block': False}})
        assert response.status_code == 200, response.text
        assert response.json()['outputs']['pid'] != child_pid


class SlowCloseHandler:
    def deploy(self, inputs):
        import os
        self.marker = Path(inputs['marker'])
        return {'pid': os.getpid()}

    def deinit(self):
        time.sleep(.5)
        self.marker.write_text('closed')


def test_process_shutdown_has_a_configurable_grace_period(tmp_path):
    import pytest
    from nndeploy.server.errors import ServerError
    for timeout in (.05, 3.):
        marker = tmp_path/str(timeout)/'closed'
        server = Server(handler_factory=SlowCloseHandler, resources=marker.parent, data_dir=marker.parent,
                        config={'execution': {'isolation': 'process',
                                              'shutdown_timeout_seconds': timeout},
                                'inputs': {'marker': {'type': 'string'}}})
        def execute():
            with TestClient(server.app) as client:
                response = client.post('/v1/server/default/deploy', json={'inputs': {'marker': str(marker)}})
                assert response.status_code == 200, response.text
                return response.json()['outputs']['pid']
        if timeout < .5:
            with pytest.raises(ServerError, match='deadline'):
                execute()
            assert not marker.exists()
        else:
            pid = execute()
            assert marker.read_text() == 'closed'
            assert not psutil.pid_exists(pid)

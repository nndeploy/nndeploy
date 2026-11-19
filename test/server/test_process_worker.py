import asyncio
import os
import time
from pathlib import Path

import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError


def controlled_deploy(values):
    if 'started' in values:
        Path(values['started']).write_text(str(os.getpid()))
    if values.get('crash'):
        os._exit(17)
    if values.get('block'):
        time.sleep(30)
    return {'pid': os.getpid(), 'text': values.get('text')}


async def wait_file(path):
    for _ in range(500):
        if path.is_file():
            return
        await asyncio.sleep(.01)
    raise AssertionError('Worker did not start')


def test_process_deadline_reclaims_worker_before_next_task(tmp_path):
    async def scenario():
        config = {'execution': {'isolation': 'process', 'execution_timeout_seconds': .2},
                  'inputs': {'block': {'type': 'boolean', 'required': False, 'default': False}}}
        server = Server(deploy=controlled_deploy, config=config, resources=tmp_path, data_dir=tmp_path)
        await server.start()
        try:
            first_pid = server.entries['default']['workers'][0].pid
            assert first_pid != os.getpid()
            with pytest.raises(ServerError) as failure:
                await server.deploy('default', {'block': True})
            assert failure.value.code == 'execution_timeout'
            assert server.db.get_task(failure.value.task_id)['state'] == 'expired'
            with pytest.raises(ProcessLookupError):
                os.kill(first_pid, 0)
            result = await server.deploy('default', {})
            assert result['outputs']['pid'] not in (first_pid, os.getpid())
        finally:
            await server.close()
    asyncio.run(scenario())


def test_q07_process_crash_does_not_stop_other_entry_and_restarts(tmp_path):
    async def scenario():
        config = {'execution': {'isolation': 'process'},
                  'inputs': {'crash': {'type': 'boolean', 'required': False, 'default': False}}}
        server = Server(name='a', deploy=controlled_deploy, config=config, resources=tmp_path, data_dir=tmp_path)
        server.add(name='b', deploy=lambda values: values)
        await server.start()
        try:
            with pytest.raises(ServerError) as failure:
                await server.deploy('a', {'crash': True})
            assert failure.value.code == 'worker_crashed'
            assert (await server.deploy('b', {'ready': True}))['outputs']['ready']
            assert (await server.deploy('a', {}))['outputs']['pid'] != os.getpid()
        finally:
            await server.close()
    asyncio.run(scenario())


def test_running_cancel_keeps_slot_until_process_has_exited(tmp_path):
    async def scenario():
        config = {'execution': {'isolation': 'process'},
                  'inputs': {'block': {'type': 'boolean'}, 'started': {'type': 'string'}}}
        server = Server(deploy=controlled_deploy, config=config, resources=tmp_path, data_dir=tmp_path)
        await server.start()
        try:
            path = tmp_path / 'started'
            identity, future = await server.submit('default', {'block': True, 'started': str(path)})
            await wait_file(path)
            pid = int(path.read_text())
            record = await server.cancel_task(identity)
            assert record['cancel_requested']
            with pytest.raises(ServerError) as cancelled:
                await future
            assert cancelled.value.code == 'task_cancelled'
            assert server.db.get_task(identity)['state'] == 'cancelled'
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert server.inflight == 0
        finally:
            await server.close()
    asyncio.run(scenario())


def test_native_graph_json_runs_in_isolated_process(tmp_path):
    from fastapi.testclient import TestClient
    from nndeploy.server.config import suggest_config
    from test_server_api import text_graph
    workflow = text_graph()
    config = suggest_config(workflow)
    config['execution'] = {'isolation': 'process', 'workers': 2}
    server = Server(workflow=workflow, config=config, resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        pids = {worker.pid for worker in server.entries['echo']['workers']}
        assert len(pids) == 2 and os.getpid() not in pids
        for text in ('first', 'second', 'third'):
            result = client.post('/v1/server/echo/deploy', json={'inputs': {'source': text}})
            assert result.status_code == 200, result.text
            assert result.json()['outputs']['sink'] == text


def test_native_image_process_output_uses_same_file_records(tmp_path):
    import io
    from fastapi.testclient import TestClient
    from PIL import Image
    from nndeploy.server.config import suggest_config
    from test_server_api import image_graph
    workflow = image_graph()
    config = suggest_config(workflow)
    config['execution'] = {'isolation': 'process'}
    with TestClient(Server(workflow=workflow, config=config, resources=tmp_path, data_dir=tmp_path).app) as client:
        data = io.BytesIO()
        Image.new('RGB', (4, 6), (12, 23, 34)).save(data, format='PNG')
        file = client.post('/v1/files', files={'file': ('image.png', data.getvalue(), 'image/png')}).json()
        result = client.post('/v1/server/image/deploy', json={'inputs': {'decode': {'file_id': file['file_id']}}})
        assert result.status_code == 200, result.text
        identity = result.json()['outputs']['encode']['file_id']
        content = client.get('/v1/files/' + identity + '/content').content
        with Image.open(io.BytesIO(content)) as image:
            assert image.size == (4, 6) and image.getpixel((0, 0)) == (12, 23, 34)


def cooperative_deploy(values):
    from nndeploy.server.request_context import current_request
    context = current_request()
    if values.get('started'):
        Path(values['started']).write_text(str(os.getpid()))
    try:
        if values.get('mode') == 'stubborn':
            time.sleep(30)
        elif values.get('mode') == 'cooperative':
            while True:
                context.check()
                time.sleep(.01)
        elif values.get('mode') == 'error':
            raise ServerError(422, 'bad_video', 'Invalid video')
        return {'pid': os.getpid()}
    finally:
        if values.get('cleanup'):
            time.sleep(.2)
            Path(values['cleanup']).write_text('clean')


def cooperative_config(execution):
    return {'execution': execution, 'inputs': {
        key: {'type': 'string', 'required': False, 'default': ''}
        for key in ('mode', 'started', 'cleanup')}}


@pytest.mark.parametrize('operation', ['cancel', 'timeout'])
@pytest.mark.parametrize('mode', ['cooperative', 'stubborn'])
def test_cooperative_process_stop_and_bounded_fallback(tmp_path, operation, mode):
    async def scenario():
        execution = {'isolation': 'process', 'cooperative_cancel': True,
                     'cancellation_grace_seconds': .5}
        if operation == 'timeout':
            execution['execution_timeout_seconds'] = .3
        server = Server(deploy=cooperative_deploy, config=cooperative_config(execution), resources=tmp_path, data_dir=tmp_path)
        await server.start()
        try:
            worker = server.entries['default']['workers'][0]
            first_pid = worker.pid
            for iteration in range(2):
                started = tmp_path / f'started-{iteration}'
                cleanup = tmp_path / f'cleanup-{iteration}'
                identity, future = await server.submit('default', {
                    'mode': mode, 'started': str(started), 'cleanup': str(cleanup)})
                await wait_file(started)
                if operation == 'cancel':
                    await server.cancel_task(identity)
                await asyncio.sleep(.05)
                assert server.inflight == 1
                assert not future.done()
                before = time.monotonic()
                with pytest.raises(ServerError) as stopped:
                    await asyncio.wait_for(asyncio.shield(future), 3)
                assert stopped.value.code == ('task_cancelled' if operation == 'cancel' else 'execution_timeout')
                assert server.db.get_task(identity)['state'] == ('cancelled' if operation == 'cancel' else 'expired')
                assert server.inflight == 0
                if mode == 'cooperative':
                    assert cleanup.read_text() == 'clean'
                    assert worker.pid == first_pid
                else:
                    assert time.monotonic() - before >= .35
                    assert not cleanup.exists()
                    assert worker.pid is None
                    with pytest.raises(ProcessLookupError):
                        os.kill(int(started.read_text()), 0)
                result = await server.deploy('default', {})
                assert (result['outputs']['pid'] == int(started.read_text())) == (mode == 'cooperative')
        finally:
            await server.close()
    asyncio.run(scenario())


def test_completed_request_error_keeps_resident_process(tmp_path):
    async def scenario():
        server = Server(deploy=cooperative_deploy, config=cooperative_config({'isolation': 'process'}), resources=tmp_path, data_dir=tmp_path)
        await server.start()
        try:
            pid = server.entries['default']['workers'][0].pid
            with pytest.raises(ServerError, match='Invalid video'):
                await server.deploy('default', {'mode': 'error'})
            assert (await server.deploy('default', {}))['outputs']['pid'] == pid
        finally:
            await server.close()
    asyncio.run(scenario())

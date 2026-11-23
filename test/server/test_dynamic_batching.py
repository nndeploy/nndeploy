"""Real Server tasks grouped into bounded, capability-declared algorithm calls."""
import asyncio
from pathlib import Path
import threading
import time
import uuid

import pytest
from pydantic import BaseModel

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.logging_taskid import get_task_id
from test_engine import until


class BatchOutput(BaseModel):
    total: int
    batch_size: int
    task_id: str


class VectorBatch:
    def __init__(self):
        self.thread = threading.get_ident()
        self.initialized = self.closed = 0
        self.calls = []

    def init(self):
        assert threading.get_ident() == self.thread
        self.initialized += 1

    def batch_key(self, inputs, **parameters):
        assert threading.get_ident() == self.thread
        if inputs['failure'] == 'key':
            raise ValueError('Unsupported input shape')
        return {'length': len(inputs['values']), 'release': inputs['release']}

    def deploy_batch(self, items):
        assert threading.get_ident() == self.thread and get_task_id() is None
        self.calls.append(items)
        for name in {item['inputs']['marker'] for item in items if item['inputs']['marker']}:
            marker = Path(name)
            temporary = marker.with_name(f'.{marker.name}.{uuid.uuid4().hex}.tmp')
            try:
                # Existence is the test's readiness signal. Publish complete
                # contents atomically, once per distinct marker in a batch.
                temporary.write_text(str(len(items)))
                temporary.replace(marker)
            finally:
                temporary.unlink(missing_ok=True)
        release = items[0]['inputs']['release']
        deadline = time.monotonic() + 10
        while release and not Path(release).exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Test release timed out')
            time.sleep(.005)
        if any(item['inputs']['failure'] == 'batch' for item in items):
            raise RuntimeError('Whole batch failed')
        if any(item['inputs']['failure'] == 'count' for item in items):
            return []
        result = []
        for item in items:
            if item['inputs']['failure'] == 'item':
                result.append(ServerError(503, 'algorithm_unavailable', 'Controlled item failure', retryable=True))
            elif item['inputs']['failure'] == 'schema':
                result.append({'total': 'wrong type'})
            else:
                result.append({'total': sum(item['inputs']['values']) * item['parameters']['scale'],
                               'batch_size': len(items), 'task_id': item['task_id']})
            item['inputs']['values'].append(999)  # Only the algorithm's copy.
        return result

    def deinit(self):
        assert threading.get_ident() == self.thread
        self.closed += 1


def batch_config(size=4, wait=100, queue_size=8, timeout=5):
    return {'execution': {'batching': {'enabled': True, 'max_size': size, 'max_wait_ms': wait},
                          'max_queue_size': queue_size, 'queue_timeout_seconds': timeout},
            'inputs': {'values': {'type': 'json'},
                       'release': {'type': 'string', 'required': False, 'default': ''},
                       'marker': {'type': 'string', 'required': False, 'default': ''},
                       'failure': {'type': 'string', 'required': False, 'default': ''}},
            'parameters': {'scale': {'type': 'integer', 'required': False, 'default': 1}}}


def make_server(tmp_path, **options):
    return Server(name='vectors', handler_factory=VectorBatch, config=batch_config(**options),
                  output_model=BatchOutput, resources=tmp_path, data_dir=tmp_path)


def test_requests_share_one_native_call_and_preserve_task_ids_and_defaults(tmp_path):
    server = make_server(tmp_path)
    async def scenario():
        await server.start()
        handler = server.entries['vectors']['workers'][0].executor.handler
        try:
            results = await asyncio.gather(*(server.deploy('vectors', {'values': [i, i + 1]}) for i in range(3)))
            assert len(handler.calls) == 1 and handler.initialized == 1
            assert [result['outputs']['total'] for result in results] == [1, 3, 5]
            assert all(result['outputs']['task_id'] == result['task_id'] and result['outputs']['batch_size'] == 3 for result in results)
            assert [server.db.get_task(result['task_id'])['inputs']['values'] for result in results] == [[0, 1], [1, 2], [2, 3]]
            started = time.monotonic()
            singleton = await server.deploy('vectors', {'values': [3]})
            assert singleton['outputs']['batch_size'] == 1
            assert time.monotonic() - started < 1
            assert server.describe('vectors')['batching']['enabled'] is True
        finally:
            await server.close()
        assert handler.closed == 1
    asyncio.run(scenario())


def test_groups_never_mix_shapes_parameters_or_owners(tmp_path):
    server = make_server(tmp_path, size=8, wait=200)
    async def scenario():
        await server.start()
        try:
            requests = [([1, 2], 1, 'alice'), ([3, 4], 1, 'alice'), ([5], 1, 'alice'),
                        ([6, 7], 2, 'alice'), ([8, 9], 1, 'bob')]
            submitted = await asyncio.gather(*(server.submit('vectors', {'values': values}, {'scale': scale}, owner=owner)
                                               for values, scale, owner in requests))
            results = await asyncio.gather(*(future for _, future in submitted))
            assert [result['outputs']['batch_size'] for result in results] == [2, 2, 1, 1, 1]
            assert [result['outputs']['total'] for result in results] == [3, 7, 5, 26, 17]
            assert len(server.entries['vectors']['workers'][0].executor.handler.calls) == 4
        finally:
            await server.close()
    asyncio.run(scenario())


def test_cancellation_during_collection_skips_compute_and_deadlines_include_collection(tmp_path):
    server = make_server(tmp_path / 'cancel', size=4, wait=1000)
    async def cancelled():
        await server.start()
        try:
            worker = server.entries['vectors']['workers'][0]
            identity, result = await server.submit('vectors', {'values': [1]})
            await until(lambda: identity in worker.jobs and worker.jobs[identity]['ready'])
            assert server.db.get_task(identity)['state'] == 'pending'
            await server.cancel_task(identity)
            with pytest.raises(ServerError, match='cancelled'):
                await asyncio.wait_for(result, .5)
            assert not worker.executor.handler.calls
        finally:
            await server.close()
    asyncio.run(cancelled())

    server = make_server(tmp_path / 'expiry', wait=150, timeout=.04)
    async def expired():
        await server.start()
        try:
            with pytest.raises(ServerError) as error:
                await server.deploy('vectors', {'values': [1]})
            assert error.value.code == 'queue_timeout'
            assert server.db.get_task(error.value.task_id)['state'] == 'expired'
            assert not server.entries['vectors']['workers'][0].executor.handler.calls
        finally:
            await server.close()
    asyncio.run(expired())


def test_running_cancel_keeps_slots_and_other_batch_items_alive(tmp_path):
    server = make_server(tmp_path, size=2, queue_size=0)
    release, marker = tmp_path / 'release', tmp_path / 'marker'
    async def scenario():
        await server.start()
        try:
            accepted = await asyncio.gather(*(server.submit('vectors', {'values': [i], 'release': str(release), 'marker': str(marker)}) for i in (1, 2)))
            await until(marker.exists)
            assert marker.read_text() == '2'
            await server.cancel_task(accepted[0][0])
            with pytest.raises(ServerError) as error:
                await server.deploy('vectors', {'values': [3]})
            assert error.value.status_code == 429
            assert all(server.db.get_task(identity)['state'] == 'running' for identity, _ in accepted)
            release.touch()
            results = await asyncio.gather(*(future for _, future in accepted), return_exceptions=True)
            assert isinstance(results[0], ServerError) and results[0].code == 'task_cancelled'
            assert results[1]['outputs']['total'] == 2
            assert (await server.deploy('vectors', {'values': [3]}))['outputs']['total'] == 3
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['item', 'schema', 'batch', 'count', 'key'])
def test_batch_failures_do_not_corrupt_other_results_or_next_execution(tmp_path, failure):
    server = make_server(tmp_path, size=2)
    async def scenario():
        await server.start()
        try:
            results = await asyncio.gather(server.deploy('vectors', {'values': [1], 'failure': failure}),
                                           server.deploy('vectors', {'values': [2]}), return_exceptions=True)
            assert isinstance(results[0], ServerError)
            if failure in ('batch', 'count'):
                assert isinstance(results[1], ServerError)
                assert results[0].task_id != results[1].task_id
            else:
                assert results[1]['outputs']['total'] == 2
            if failure == 'item':
                assert results[0].retryable and results[0].status_code == 503
            if failure == 'count':
                assert results[0].code == 'batch_result_mismatch'
            assert (await server.deploy('vectors', {'values': [3]}))['outputs']['total'] == 3
        finally:
            await server.close()
    asyncio.run(scenario())


def test_rejects_unsupported_batch_execution_combinations(tmp_path):
    from nndeploy.server.config import ServerConfig
    from test_server_api import text_graph
    for change in ({'scheduling': 'engine'}, {'scheduling': 'concurrent'}, {'isolation': 'process'}):
        config = batch_config()
        config['execution'].update(change)
        with pytest.raises(ValueError):
            ServerConfig.model_validate(config)
    config = batch_config()
    config['streaming'] = {'enabled': True}
    with pytest.raises(ValueError):
        ServerConfig.model_validate(config)
    async def scenario():
        for index, source in enumerate(({'workflow': text_graph()}, {'deploy': lambda values: values})):
            server = Server(config=batch_config(), resources=tmp_path / str(index), data_dir=tmp_path / str(index), **source)
            with pytest.raises(ValueError, match='Dynamic batching'):
                await server.start()
            assert server.describe('default')['status'] == 'failed'
            await server.close()
    asyncio.run(scenario())


def test_expiry_is_rechecked_after_slow_start_recording(tmp_path, monkeypatch):
    server = make_server(tmp_path, size=2, wait=20, timeout=.08)
    start = server.db.start_task
    calls = []
    def delayed(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            time.sleep(.12)
        return start(*args, **kwargs)
    monkeypatch.setattr(server.db, 'start_task', delayed)
    async def scenario():
        await server.start()
        try:
            result = await asyncio.gather(*(server.deploy('vectors', {'values': [i]}) for i in (1, 2)), return_exceptions=True)
            assert all(isinstance(error, ServerError) and error.code == 'queue_timeout' for error in result)
            assert not server.entries['vectors']['workers'][0].executor.handler.calls
        finally:
            await server.close()
    asyncio.run(scenario())


def test_reload_drains_old_batch_and_new_requests_use_new_handler(tmp_path):
    server = make_server(tmp_path, size=2, wait=30)
    release, marker = tmp_path / 'release', tmp_path / 'marker'
    async def scenario():
        await server.start()
        old = server.entries['vectors']
        old_handler = old['workers'][0].executor.handler
        try:
            identities = await asyncio.gather(*(server.submit('vectors', {'values': [i], 'release': str(release), 'marker': str(marker)}) for i in (1, 2)))
            await until(marker.exists)
            reloading = asyncio.create_task(server.reload_entry('vectors'))
            await until(lambda: server.entries['vectors'] is not old)
            current = server.entries['vectors']['workers'][0].executor.handler
            assert current is not old_handler and not old_handler.closed and not reloading.done()
            fresh = await asyncio.gather(*(server.deploy('vectors', {'values': [i]}) for i in (3, 4)))
            assert all(item['outputs']['batch_size'] == 2 for item in fresh)
            assert len(old_handler.calls) == len(current.calls) == 1
            release.touch()
            await asyncio.gather(*(future for _, future in identities))
            await reloading
            assert old_handler.closed == 1 and current.closed == 0
        finally:
            release.touch()
            await server.close()
        assert current.closed == 1
    asyncio.run(scenario())


def test_oldest_shape_is_not_starved_by_later_full_group(tmp_path):
    server = make_server(tmp_path, size=4, wait=100)
    async def scenario():
        await server.start()
        try:
            first_id, first = await server.submit('vectors', {'values': [1]})
            worker = server.entries['vectors']['workers'][0]
            await until(lambda: first_id in worker.jobs and worker.jobs[first_id]['ready'])
            later = await asyncio.gather(*(server.submit('vectors', {'values': [i, i]}) for i in (2, 3, 4)))
            await asyncio.gather(first, *(future for _, future in later))
            assert worker.executor.handler.calls[0][0]['task_id'] == first_id
            assert len(worker.executor.handler.calls[1]) == 3
        finally:
            await server.close()
    asyncio.run(scenario())

"""Shutdown must join startup and release resources after background failures."""
import asyncio
import threading

import pytest

from nndeploy.server import Server


def test_close_during_start_joins_created_background_tasks(tmp_path, monkeypatch):
    server = Server(data_dir=tmp_path)
    entered, release = threading.Event(), threading.Event()
    recover = server.storage.recover_pending

    def held_recover():
        entered.set()
        assert release.wait(5)
        return recover()

    monkeypatch.setattr(server.storage, 'recover_pending', held_recover)

    async def scenario():
        starting = asyncio.create_task(server.start())
        assert await asyncio.to_thread(entered.wait, 3)
        closing = asyncio.create_task(server.close())
        await asyncio.sleep(.02)
        release.set()
        await starting
        await closing
        assert server.task_maintenance.done()
        assert server.callbacks.task.done()
        assert not server.started

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize('component', ['maintenance', 'callbacks'])
def test_background_failure_still_closes_workers_and_dependencies(tmp_path, monkeypatch, component):
    server = Server(name='echo', deploy=lambda inputs: inputs, data_dir=tmp_path)
    closed = []
    for name in ('controls', 'db', 'storage'):
        dependency = getattr(server, name)
        original = dependency.close

        def close(original=original, name=name):
            closed.append(name)
            return original()

        monkeypatch.setattr(dependency, 'close', close)

    async def failed():
        raise RuntimeError('Injected background failure')

    async def scenario():
        await server.start()
        worker = server.entries['echo']['workers'][0]
        if component == 'maintenance':
            server.shutting_down = True
            server.maintenance_stop.set()
            await server.task_maintenance
            server.shutting_down = False
            server.task_maintenance = asyncio.create_task(failed())
            await asyncio.gather(server.task_maintenance, return_exceptions=True)
        else:
            await server.callbacks.close()
            monkeypatch.setattr(server.callbacks, 'close', failed)
        try:
            with pytest.raises(RuntimeError, match='Injected background failure'):
                await server.close()
            assert worker.closed
            assert closed == ['controls', 'db', 'storage']
            assert not server.started
            with pytest.raises(RuntimeError, match='Injected background failure'):
                await server.close()
            assert closed == ['controls', 'db', 'storage']
        finally:
            # Keep a failing baseline from leaking native worker threads.
            await worker.close()
            if getattr(server, 'callbacks', None):
                server.callbacks.stopping.set()
                await asyncio.gather(server.callbacks.task, return_exceptions=True)
            for name in ('controls', 'db', 'storage'):
                if name not in closed:
                    getattr(server, name).close()

    asyncio.run(scenario())


def test_cancelled_close_waiter_does_not_cancel_cleanup(tmp_path):
    entered, release = threading.Event(), threading.Event()
    closed = []

    class Handler:
        def deploy(self, inputs):
            return inputs

        def deinit(self):
            entered.set()
            assert release.wait(5)
            closed.append(True)

    server = Server(name='echo', handler_factory=Handler, data_dir=tmp_path)

    async def scenario():
        await server.start()
        waiter = asyncio.create_task(server.close())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert not server.closing_task.cancelled()
        finally:
            release.set()
            await server.close()
        assert closed == [True]
        assert not server.started

    asyncio.run(scenario())

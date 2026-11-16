"""Private Graph copies outlive readers and saved configurations, not guesses."""
import asyncio
import functools
import io
import json
import multiprocessing
from pathlib import Path
import threading
import uuid

import pytest
from sqlalchemy import update

from nndeploy.server import Server
from nndeploy.server.assets import AssetCleanup, AssetLease
from nndeploy.server.config import ServerConfig
from nndeploy.server.db import DB, server_configs
from nndeploy.server.resources import prepare_workflow
from test_db import db
from test_declared_resources import graph_with_resources, settings, assets


def copied(directory, db, source):
    lease = AssetLease(directory, db)
    try:
        result = prepare_workflow({'io_type_': 'Model', 'io_params_': ['model'], 'model': str(source)},
                                  ServerConfig(), directory, source.parent, lease=lease)
        return Path(result['model']), lease
    except BaseException:
        lease.close()
        raise


def test_readers_saved_configs_and_legacy_paths_block_cleanup(db, tmp_path):
    source = tmp_path / 'weights.bin'
    source.write_bytes(b'weights')
    directory = tmp_path / 'runtime'
    cleaner = AssetCleanup(directory, db)
    path, first = copied(directory, db, source)
    second = AssetLease(directory, db)
    scope = uuid.uuid4().hex
    try:
        second.hold(path)
        assert cleaner.cleanup()['deleted'] == 0
        first.close()
        assert cleaner.cleanup()['deleted'] == 0
        db.save_server_config(scope, 'graph', {'workflow': {'model': str(path)}, 'config': {}})
        second.close()
        assert cleaner.cleanup()['deleted'] == 0 and path.read_bytes() == b'weights'
        db.save_server_config(scope, 'graph', {'workflow': {}, 'config': {}})
        legacy = directory / 'entries' / 'graph.json'
        legacy.parent.mkdir()
        legacy.write_text(json.dumps({'workflow': {'model': str(path)}}))
        assert cleaner.cleanup()['deleted'] == 0
        legacy.unlink()
        assert cleaner.cleanup()['deleted'] == 1
        assert not path.exists()
        assert cleaner.cleanup()['scanned'] == 0
        # Publication after collection must create a new lock inode safely.
        again, lease = copied(directory, db, source)
        lease.close()
        assert again.exists() and cleaner.cleanup()['deleted'] == 1
    finally:
        first.close()
        second.close()


def test_cleanup_never_adopts_legacy_or_other_database_copies(db, tmp_path):
    directory = tmp_path / 'runtime'
    source = tmp_path / 'model.bin'
    source.write_bytes(b'legacy')
    prepared = prepare_workflow({'model_path_': str(source)}, ServerConfig(), directory, tmp_path)
    legacy = Path(prepared['model_path_'])
    lease = AssetLease(directory, db)
    lease.hold(legacy)
    lease.close()
    assert AssetCleanup(directory, db).cleanup()['deleted'] == 0
    source.write_bytes(b'new')
    path, lease = copied(directory, db, source)
    other = DB(tmp_path / 'other.db').init_schema()
    foreign = AssetLease(directory, other)
    try:
        foreign.hold(path)
        foreign.close()
        lease.close()
        assert AssetCleanup(directory, db).cleanup()['deleted'] == 0
        assert AssetCleanup(directory, other).cleanup()['deleted'] == 0
        assert path.exists() and legacy.exists()
    finally:
        foreign.close()
        lease.close()
        other.close()


def test_cleanup_database_failure_keeps_bytes_and_retries(db, tmp_path, monkeypatch):
    source = tmp_path / 'model'
    source.write_bytes(b'weights')
    path, lease = copied(tmp_path / 'runtime', db, source)
    lease.close()
    cleaner = AssetCleanup(tmp_path / 'runtime', db)
    scope = uuid.uuid4().hex
    db.save_server_config(scope, 'corrupt', {'workflow': {}, 'config': {}})
    with db.engine.begin() as connection:
        connection.execute(update(server_configs).where(server_configs.c.scope == scope).values(data='broken'))
    try:
        with pytest.raises(json.JSONDecodeError):
            cleaner.cleanup()
        assert path.exists()
    finally:
        with db.engine.begin() as connection:
            connection.execute(update(server_configs).where(server_configs.c.scope == scope).values(data='{}'))
    import nndeploy.server.assets as module
    with monkeypatch.context() as patch:
        patch.setattr(module.shutil, 'rmtree', lambda path: (_ for _ in ()).throw(OSError('busy')))
        assert cleaner.cleanup()['failed'] == 1
        assert path.exists()
    assert cleaner.cleanup()['deleted'] == 1


def test_unknown_configuration_write_retains_copies_until_confirmed(db, tmp_path):
    source = tmp_path / 'weights'
    source.write_bytes(b'weights')
    path, lease = copied(tmp_path / 'runtime', db, source)
    token = lease.begin_publication()
    lease.close()  # OS locks alone cannot establish whether a DB commit finished.
    cleaner = AssetCleanup(tmp_path / 'runtime', db)
    assert cleaner.cleanup()['deleted'] == 0 and path.exists()
    confirmed = AssetLease(tmp_path / 'runtime', db)
    try:
        confirmed.hold(path)
        confirmed.end_publication(token)
    finally:
        confirmed.close()
    assert cleaner.cleanup()['deleted'] == 1


def test_editor_worker_holds_materialized_input_until_graph_release(tmp_path):
    from nndeploy.server.editor_inputs import prepare_worker_inputs
    server = Server(data_dir=tmp_path / 'runtime')
    record = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
    options = dict(root=str(server.storage.root), database_url=server.db.url.render_as_string(hide_password=False),
                   timeout_seconds=10, max_bytes=server.storage.max_bytes, config=server.storage.config.model_dump())
    task = {'graph_json': json.loads(graph_with_resources([{'file_id': record['file_id']}]).serialize())}
    held = []
    try:
        prepared = prepare_worker_inputs(task, options, tmp_path, retain=held.append)
        path = Path(prepared['node_repository_'][0]['artifacts'][0])
        assert path.read_bytes() == b'weights'
        assert task['graph_json']['node_repository_'][0]['artifacts'] == [{'file_id': record['file_id']}]
        assert server._asset_cleanup.cleanup()['deleted'] == 0
        held[0].close()
        assert server._asset_cleanup.cleanup()['deleted'] == 1 and not path.exists()
        assert server.storage.metadata(record['file_id'])['state'] == 'ready'
    finally:
        for lease in held:
            lease.close()
        asyncio.run(server.close())


def child_copy(directory, url, source, channel, temporary):
    database = DB(url)
    lease = AssetLease(directory, database)
    if temporary:
        with lease.temporary('.copy-') as pending:
            path = Path(pending) / 'unfinished.bin'
            path.write_bytes(b'partial')
            channel.send(str(path))
            channel.recv()
    else:
        path, held = copied(directory, database, Path(source))
        channel.send(str(path))
        channel.recv()
        held.close()
    lease.close()
    database.close()


@pytest.mark.parametrize('temporary', [False, True])
def test_process_crash_reclaims_only_after_physical_exit(db, tmp_path, temporary):
    source = tmp_path / 'weights.bin'
    source.write_bytes(b'weights')
    directory = tmp_path / 'runtime'
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=child_copy, args=(directory, db.url.render_as_string(hide_password=False),
                                                       source, child, temporary))
    process.start()
    child.close()
    try:
        assert parent.poll(20)
        path = Path(parent.recv())
        cleaner = AssetCleanup(directory, db)
        assert cleaner.cleanup()['deleted'] == 0 and path.exists()
        process.kill()
        process.join(10)
        assert not process.is_alive()
        assert cleaner.cleanup()['deleted'] == 1
        assert not path.exists()
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        process.close()
        parent.close()


@pytest.mark.parametrize('source,isolation', [('graph', 'thread'), ('factory', 'thread'), ('factory', 'process'),
                                              ('workflow', 'thread'), ('workflow', 'process')])
def test_stop_start_and_server_restart_keep_required_copies(db, tmp_path, source, isolation):
    weights, _ = assets(tmp_path / 'original')
    arguments = ({'graph': graph_with_resources([str(weights)])} if source == 'graph' else
                 {'graph_factory': functools.partial(graph_with_resources, [str(weights)])} if source == 'factory' else
                 {'workflow': json.loads(graph_with_resources([str(weights)]).serialize())})
    directory = tmp_path / 'runtime'
    options = dict(data_dir=directory, database_url=db.url.render_as_string(hide_password=False))
    server = Server(**options, **arguments, config=settings(isolation))
    plugin = directory / 'plugin'
    plugin.mkdir()
    (plugin / 'resources.py').write_text('import test_declared_resources\n')
    async def scenario():
        try:
            await server.start()
            weights.unlink()
            assert (await server.cleanup_files())['assets']['deleted'] == 0
            await server.stop_entry('assets')
            assert (await server.cleanup_files())['assets']['deleted'] == 0
            await server.start_entry('assets')
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights'}
        finally:
            await server.close()
        cleaner = AssetCleanup(directory, db)
        assert cleaner.cleanup()['deleted'] == (1 if source == 'graph' else 0)
        if source != 'graph':
            restored = Server(**options, **(dict(**arguments, config=settings(isolation)) if source == 'factory' else {}))
            try:
                await restored.start()
                assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'weights'}
            finally:
                await restored.close()
    asyncio.run(scenario())


def test_reload_discards_previous_and_failed_candidate_copies(tmp_path, monkeypatch):
    from test_declared_resources import ResourceNode
    source = tmp_path / 'model.bin'
    source.write_text('first')
    workflow = json.loads(graph_with_resources([str(source)]).serialize())
    server = Server(workflow=workflow, config=settings(), data_dir=tmp_path / 'runtime')
    def paths():
        return [path for path in (server.data_dir / 'assets').glob('*/model.bin')]
    async def scenario():
        try:
            await server.start()
            old, = paths()
            source.write_text('second')
            await server.reload_entry('assets', workflow=workflow)
            assert (await server.cleanup_files())['assets']['deleted'] == 1
            assert not old.exists()
            source.write_text('broken')
            with monkeypatch.context() as patch:
                def fail(node):
                    raise RuntimeError('controlled init failure')
                patch.setattr(ResourceNode, 'init', fail)
                with pytest.raises(RuntimeError, match='controlled'):
                    await server.reload_entry('assets', workflow=workflow)
            assert (await server.cleanup_files())['assets']['deleted'] == 1
            assert [path.read_text() for path in paths()] == ['second']
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'second'}
        finally:
            await server.close()
    asyncio.run(scenario())


def test_failed_native_deinit_never_releases_asset_lock(db, tmp_path):
    from nndeploy.server.executor import Executor
    source = tmp_path / 'model'
    source.write_bytes(b'weights')
    path, lease = copied(tmp_path / 'runtime', db, source)
    executor = Executor({'graph': object()}, ServerConfig())
    calls = []
    class FailedGraph:
        def deinit(self):
            calls.append(True)
            raise RuntimeError('physical cleanup unknown')
    executor.asset_lease = lease
    executor.runner.graph = FailedGraph()
    executor.runner._owner = threading.get_ident()
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='physical cleanup unknown'):
                executor.close()
            assert AssetCleanup(tmp_path / 'runtime', db).cleanup()['deleted'] == 0
            assert path.exists()
        assert calls == [True]
        with pytest.raises(RuntimeError, match='physical cleanup unknown'):
            executor.runner.execute(None, None)
    finally:
        # The fake Graph owns no real threads/resources; explicitly release the
        # test lease after checking failed cleanup cannot claim success.
        lease.close()

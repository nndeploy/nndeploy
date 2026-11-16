"""Portable transfer must retain every Server table across actual backends."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import itertools
import json
import multiprocessing
import os
import stat
import subprocess
import sys

import pytest
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, event, insert, inspect, select, text
from sqlalchemy.exc import IntegrityError

from nndeploy.server.db import DB, server_meta
from nndeploy.server.database_transfer import _tables, _line
from test_db import db, URLS
from test_database_backup import empty_database, seed


def seed_all(database, root):
    database.init_schema()
    values = {}
    with database.engine.begin() as connection:
        for table in _tables():
            row = {}
            for column in table.c:
                kind = column.type
                if isinstance(kind, Boolean):
                    value = True
                elif isinstance(kind, Integer):
                    value = 4000
                elif isinstance(kind, Float):
                    value = 1234.125
                elif isinstance(kind, DateTime):
                    value = datetime(2026, 11, 14, 1, 2, 3, 456789)
                elif isinstance(kind, Text):
                    value = json.dumps({'text': '中文\\ "\n' + ('x' * 70000 if 'mysql' in kind._variant_mapping else '')}, ensure_ascii=False)
                elif isinstance(kind, String):
                    value = column.name[:16]
                else:
                    raise AssertionError(kind)
                if column.name == 'state':
                    value = 'running'  # Unknown live state must never be invented as terminal.
                if column.nullable and column.name.endswith('_at') and not isinstance(kind, DateTime):
                    value = None
                row[column.name] = value
            for column in table.c:
                for foreign in column.foreign_keys:
                    row[column.name] = values[foreign.column.table.name][foreign.column.name]
            connection.execute(insert(table).values(**row))
            values[table.name] = row
    identity = database.database_identity()
    workflow, task, path = seed(database, root)
    return identity, workflow, task, path, values


def snapshot(database):
    with database.engine.connect() as connection:
        return {table.name: [dict(row) for row in connection.execute(select(table).order_by(*table.primary_key.columns)).mappings()]
                for table in _tables()}


@pytest.mark.parametrize('source_url,target_url', list(itertools.product(URLS, repeat=2)),
                         ids=lambda url: url.split(':')[0])
def test_portable_database_all_backend_pairs(source_url, target_url, tmp_path):
    source_parent = DB(tmp_path / 'source-parent.db' if source_url == 'sqlite' else source_url)
    target_parent = DB(tmp_path / 'target-parent.db' if target_url == 'sqlite' else target_url)
    try:
        with empty_database(source_parent, tmp_path, 'source') as source, empty_database(target_parent, tmp_path, 'target') as target:
            identity, workflow, task, path, values = seed_all(source, tmp_path)
            original = snapshot(source)
            archive = tmp_path / 'database.jsonl'
            assert source.export_data(archive) == archive
            if os.name != 'nt':
                assert stat.S_IMODE(archive.stat().st_mode) == 0o600
            assert target.import_data(archive) == archive
            assert snapshot(target) == original
            assert target.database_identity() == identity
            assert target.get_workflow_path(workflow) == path
            assert target.get_task(task) == source.get_task(task)
            assert target.read_task_events(task) == source.read_task_events(task)
            assert target.create_task('different', 'backup', 1, source.get_task(task)['inputs'], {}, idempotency_key='same') == task
            target.export_data(tmp_path / 'again.jsonl')
            assert (tmp_path / 'again.jsonl').read_bytes() == archive.read_bytes()
            # Explicit imported IDs must not leave PostgreSQL's sequence at 1.
            from nndeploy.server.audit import records
            audit = dict(values['server_audit'], request_id='next-audit')
            audit.pop('id')
            with target.engine.begin() as connection:
                result = connection.execute(insert(records).values(**audit))
                assert result.inserted_primary_key[0] == 4001
            with pytest.raises(IntegrityError):
                with target.engine.begin() as connection:
                    connection.execute(text('DELETE FROM server_files'))
            with pytest.raises(ValueError, match='empty'):
                target.import_data(archive)
            assert target.get_task(task)['state'] == 'succeeded'
    finally:
        source_parent.close()
        target_parent.close()


def rewrite(archive, change):
    records = [json.loads(line) for line in archive.read_bytes().splitlines()]
    change(records)
    raw = b''.join(_line(record) for record in records[:-1])
    records[-1]['sha256'] = hashlib.sha256(raw).hexdigest()
    archive.write_bytes(raw + _line(records[-1]))


@pytest.mark.parametrize('damage', ['truncated', 'checksum', 'extra', 'version', 'duplicate', 'type', 'counts', 'timestamp'])
def test_corrupt_transfer_rejected_before_destination_schema(damage, tmp_path):
    source = DB(tmp_path / 'source.db').init_schema()
    target = DB(tmp_path / 'target.db')
    try:
        seed(source, tmp_path)
        archive = tmp_path / 'data.jsonl'
        source.export_data(archive)
        if damage == 'truncated':
            archive.write_bytes(archive.read_bytes()[:-15])
        elif damage == 'checksum':
            archive.write_bytes(archive.read_bytes().replace(b'backup', b'backuq'))
        elif damage == 'extra':
            archive.write_bytes(archive.read_bytes() + b'{}\n')
        elif damage == 'version':
            rewrite(archive, lambda records: records[0].update(version=2))
        elif damage == 'duplicate':
            archive.write_bytes(archive.read_bytes().replace(b'"version":1', b'"version":1,"version":1', 1))
        elif damage == 'type':
            def change(records):
                index = records.index({'table': 'server_tasks'})
                column = records[0]['tables']['server_tasks'].index('cancel_requested')
                records[index + 1][column] = 1
            rewrite(archive, change)
        elif damage == 'counts':
            rewrite(archive, lambda records: records[-1].update(rows=999999))
        else:
            def change(records):
                index = records.index({'table': 'workflows'})
                column = records[0]['tables']['workflows'].index('created_at')
                records[index + 1][column] = '2026-11-14T01:02:03.1234567'
            rewrite(archive, change)
        with pytest.raises(ValueError):
            target.import_data(archive)
        with target.engine.connect() as connection:
            assert inspect(connection).get_table_names() == []
    finally:
        source.close()
        target.close()


def test_import_transaction_failure_keeps_empty_current_schema_and_can_retry(db, tmp_path):
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed_all(source, tmp_path)
        archive = tmp_path / 'data.jsonl'
        source.export_data(archive)
        original = archive.read_bytes()
        # Valid checksum/type but dangling reference, deliberately late in the
        # import after other tables have been written. FKs stay enabled.
        def change(records):
            index = records.index({'table': 'server_editor_file_refs'})
            column = records[0]['tables']['server_editor_file_refs'].index('file_id')
            records[index + 1][column] = 'file_missing'
        rewrite(archive, change)
        with pytest.raises(IntegrityError):
            target.import_data(archive)
        assert all(not rows for rows in snapshot(target).values())
        archive.write_bytes(original)
        target.import_data(archive)
        assert snapshot(target) == snapshot(source)


def test_export_refuses_schema_drift_and_never_overwrites(db, tmp_path):
    with empty_database(db, tmp_path, 'source') as source:
        seed(source, tmp_path)
        archive = tmp_path / 'data.jsonl'
        source.export_data(archive)
        original = archive.read_bytes()
        with pytest.raises(FileExistsError):
            source.export_data(archive)
        assert archive.read_bytes() == original
        with source.engine.begin() as connection:
            connection.exec_driver_sql('CREATE TABLE custom_data (id INTEGER PRIMARY KEY)')
        with pytest.raises(ValueError, match='exactly'):
            source.export_data(tmp_path / 'drift.jsonl')
        assert not (tmp_path / 'drift.jsonl').exists()
        assert not list(tmp_path.glob('.database-export-*'))


def test_transfer_cli_uses_database_environment_and_exits(tmp_path):
    source = DB(tmp_path / 'source.db').init_schema()
    target = DB(tmp_path / 'target.db')
    try:
        _, task, _ = seed(source, tmp_path)
        archive = tmp_path / 'data.jsonl'
        for database, action in ((source, '--database-export'), (target, '--database-import')):
            environment = dict(os.environ, NNDEPLOY_TRANSFER_URL=database.url.render_as_string(hide_password=False))
            result = subprocess.run([sys.executable, '-m', 'nndeploy.server', '--database-url-env',
                'NNDEPLOY_TRANSFER_URL', action, str(archive)], env=environment, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            assert 'Database operation completed:' in result.stdout
            assert 'Uvicorn running' not in result.stderr
        assert target.get_task(task) == source.get_task(task)
    finally:
        source.close()
        target.close()


def test_transfer_deadline_rolls_back_or_removes_partial_file(db, tmp_path, monkeypatch):
    import nndeploy.server.database_transfer as transfer
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed_all(source, tmp_path)
        archive = tmp_path / 'data.jsonl'
        source.export_data(archive)
        deadline = transfer._deadline
        def forced(_):
            def check():
                raise TimeoutError('test deadline')
            return check
        monkeypatch.setattr(transfer, '_deadline', forced)
        with pytest.raises(TimeoutError):
            source.export_data(tmp_path / 'partial.jsonl')
        assert not (tmp_path / 'partial.jsonl').exists()
        assert not list(tmp_path.glob('.database-export-*'))
        monkeypatch.setattr(transfer, '_deadline', deadline)
        original = transfer._read
        def fail_late(archive, tables, check, consume=None):
            if consume:
                count = 0
                def write(table, row):
                    nonlocal count
                    consume(table, row)
                    count += 1
                    if count == 10:
                        raise TimeoutError('test deadline')
                return original(archive, tables, check, write)
            return original(archive, tables, check)
        monkeypatch.setattr(transfer, '_read', fail_late)
        with pytest.raises(TimeoutError):
            target.import_data(archive)
        assert all(not rows for rows in snapshot(target).values())


def import_in_process(url, archive):
    database = DB(url)
    try:
        database.import_data(archive)
        return 'imported'
    except ValueError as error:
        if 'empty' in str(error):
            return 'occupied'
        raise
    finally:
        database.close()


def test_two_process_imports_only_one_can_populate_destination(db, tmp_path):
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed_all(source, tmp_path)
        archive = tmp_path / 'data.jsonl'
        source.export_data(archive)
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = [pool.submit(import_in_process, target.url.render_as_string(hide_password=False), str(archive)) for _ in range(2)]
            assert sorted(future.result(timeout=40) for future in results) == ['imported', 'occupied']
        assert snapshot(target) == snapshot(source)


def test_export_uses_one_snapshot_across_tables(db, tmp_path):
    if db.engine.dialect.name == 'sqlite':
        pytest.skip('SQLite read transactions already prevent concurrent commit; exercise MVCC backends')
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed_all(source, tmp_path)
        original = snapshot(source)
        changed = False
        def concurrent_write(connection, cursor, statement, parameters, context, many):
            nonlocal changed
            if not changed and statement.startswith('SELECT server_task_events.'):
                changed = True
                source.create_task('task_concurrent', 'concurrent', 1, {'text': 'after snapshot'}, {})
                source.start_task('task_concurrent', 2)
                source.finish_task('task_concurrent', 'succeeded', 3, {'text': 'after snapshot'})
        event.listen(source.engine, 'before_cursor_execute', concurrent_write)
        try:
            source.export_data(tmp_path / 'data.jsonl')
        finally:
            event.remove(source.engine, 'before_cursor_execute', concurrent_write)
        assert changed
        assert source.get_task('task_concurrent')['state'] == 'succeeded'
        target.import_data(tmp_path / 'data.jsonl')
        assert snapshot(target) == original


def test_import_detects_silent_target_timestamp_rounding(db, tmp_path):
    if db.engine.dialect.name != 'mysql':
        pytest.skip('MySQL DATETIME(0) silently rounds microseconds even in strict SQL mode')
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed_all(source, tmp_path)
        source.export_data(tmp_path / 'data.jsonl')
        target.init_schema()
        with target.engine.begin() as connection:
            connection.exec_driver_sql('ALTER TABLE workflows MODIFY created_at DATETIME(0) DEFAULT CURRENT_TIMESTAMP')
        with pytest.raises(ValueError, match='differ'):
            target.import_data(tmp_path / 'data.jsonl')
        assert all(not rows for rows in snapshot(target).values())


def test_writer_committing_before_import_lock_is_not_overwritten(db, tmp_path, monkeypatch):
    if db.engine.dialect.name != 'postgresql':
        pytest.skip('PostgreSQL table lock must precede the empty-table read snapshot')
    import nndeploy.server.database_transfer as transfer
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed(source, tmp_path)
        source.export_data(tmp_path / 'data.jsonl')
        target.init_schema()
        original = transfer._empty
        def write_then_check(connection, tables):
            with target.engine.begin() as other:
                other.execute(insert(server_meta).values(key='concurrent', value='keep'))
            original(connection, tables)
        monkeypatch.setattr(transfer, '_empty', write_then_check)
        with pytest.raises(ValueError, match='empty'):
            target.import_data(tmp_path / 'data.jsonl')
        state = snapshot(target)
        assert state.pop('server_meta') == [{'key': 'concurrent', 'value': 'keep'}]
        assert all(not rows for rows in state.values())


def test_materialized_view_is_not_silently_omitted(db, tmp_path):
    if db.engine.dialect.name != 'postgresql':
        pytest.skip('PostgreSQL-specific schema object')
    with empty_database(db, tmp_path, 'source') as source:
        seed(source, tmp_path)
        with source.engine.begin() as connection:
            connection.exec_driver_sql('CREATE MATERIALIZED VIEW extra_records AS SELECT id FROM workflows')
        with pytest.raises(ValueError, match='materialized'):
            source.export_data(tmp_path / 'data.jsonl')
        assert not (tmp_path / 'data.jsonl').exists()


def test_nontransactional_mysql_target_is_rejected_before_writing(db, tmp_path):
    if db.engine.dialect.name != 'mysql':
        pytest.skip('MySQL-specific storage engine')
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed_all(source, tmp_path)
        source.export_data(tmp_path / 'data.jsonl')
        target.init_schema()
        with target.engine.begin() as connection:
            connection.exec_driver_sql('ALTER TABLE server_meta ENGINE=MyISAM')
        with pytest.raises(ValueError, match='InnoDB'):
            target.import_data(tmp_path / 'data.jsonl')
        assert all(not rows for rows in snapshot(target).values())

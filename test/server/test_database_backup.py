"""Native backups must actually restore records and protect schema upgrades."""
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import uuid

import pytest
from sqlalchemy import inspect, text

from nndeploy.server.db import DB
from nndeploy.server.database_backup import SUFFIXES
from test_db import db


@contextmanager
def empty_database(parent, root, tag):
    if parent.engine.dialect.name == 'sqlite':
        database = DB(root / (tag + '.sqlite3'), backup_dir=root / 'backups')
    else:
        name = 'nndeploy_backup_' + uuid.uuid4().hex
        quote = parent.engine.dialect.identifier_preparer.quote(name)
        with parent.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as connection:
            connection.exec_driver_sql('CREATE DATABASE ' + quote +
                (' CHARACTER SET utf8mb4 COLLATE utf8mb4_bin' if parent.engine.dialect.name == 'mysql' else ''))
        database = DB(parent.url.set(database=name).render_as_string(hide_password=False), backup_dir=root / 'backups')
    try:
        yield database
    finally:
        database.close()
        if parent.engine.dialect.name != 'sqlite':
            with parent.engine.connect().execution_options(isolation_level='AUTOCOMMIT') as connection:
                connection.exec_driver_sql('DROP DATABASE ' + quote)


def seed(database, tmp_path):
    database.init_schema()
    path = tmp_path / 'original.json'
    path.write_text('{}')
    workflow = database.upsert_workflow_by_path(path, '中文 workflow')
    task = 'task_' + uuid.uuid4().hex
    database.create_task(task, 'backup', time.time(), {'text': 'keep \\ quotes " 中文', 'integer': 2**63-1}, {}, idempotency_key='same')
    database.start_task(task, time.time())
    database.finish_task(task, 'succeeded', time.time(), {'text': 'x' * 70000})
    return workflow, task, path


def test_native_database_backup_restores_ids_payloads_events_and_schema(db, tmp_path):
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        workflow, task, path = seed(source, tmp_path)
        destination = tmp_path / ('backup' + SUFFIXES[source.engine.dialect.name])
        assert source.backup(destination) == destination
        assert destination.stat().st_size > 0
        if os.name != 'nt':
            assert stat.S_IMODE(destination.stat().st_mode) == 0o600
        assert not list(tmp_path.glob('.database-backup-*'))
        target.restore(destination)
        target.init_schema()
        assert target.get_workflow_path(workflow) == path
        assert target.get_workflow_name(workflow) == '中文 workflow'
        assert target.get_task(task)['inputs'] == source.get_task(task)['inputs']
        assert target.get_task(task)['outputs'] == {'text': 'x' * 70000}
        assert target.read_task_events(task) == source.read_task_events(task)
        assert target.create_task('different', 'backup', time.time(), source.get_task(task)['inputs'], {}, idempotency_key='same') == task
        with pytest.raises(ValueError, match='empty'):
            target.restore(destination)
        assert target.get_task(task)['state'] == 'succeeded'


def test_backup_destination_never_overwritten(db, tmp_path):
    destination = tmp_path / 'keep'
    destination.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        db.backup(destination)
    assert destination.read_bytes() == b'original'


def test_existing_database_gets_backup_before_upgrade_and_only_once(db, tmp_path):
    from alembic import command
    from alembic.config import Config
    from alembic.migration import MigrationContext
    with empty_database(db, tmp_path, 'legacy') as source, empty_database(db, tmp_path, 'restored') as target:
        config = Config()
        config.set_main_option('script_location', str(Path(__file__).resolve().parents[2] / 'python/nndeploy/server/migrations'))
        with source.engine.begin() as connection:
            config.attributes['connection'] = connection
            command.upgrade(config, '0029_editor_output_files')
        # Older migration code imports the current table descriptors. Remove
        # the 0030-only column if a fresh earlier schema already included it.
        with source.engine.begin() as connection:
            if 'input_state' in {column['name'] for column in inspect(connection).get_columns('server_tasks')}:
                connection.exec_driver_sql('ALTER TABLE server_tasks DROP COLUMN input_state')
        original = source.upsert_workflow_by_path(tmp_path / 'preserved.json', 'old workflow')
        source.init_schema()
        backups = list((tmp_path / 'backups').glob('*')) if source.engine.dialect.name != 'sqlite' else list(tmp_path.glob('legacy.sqlite3.backup-*.sqlite3'))
        assert len(backups) == 1
        target.restore(backups[0])
        with target.engine.connect() as connection:
            assert MigrationContext.configure(connection).get_current_revision() == '0029_editor_output_files'
            assert 'input_state' not in {column['name'] for column in inspect(connection).get_columns('server_tasks')}
        assert target.get_workflow_name(original) == 'old workflow'
        source._migrated = False
        source.init_schema()
        assert source.get_workflow_name(original) == 'old workflow'
        assert (list((tmp_path / 'backups').glob('*')) if source.engine.dialect.name != 'sqlite' else list(tmp_path.glob('legacy.sqlite3.backup-*.sqlite3'))) == backups


def test_backup_failure_stops_remote_migration_before_ddl(db, tmp_path, monkeypatch):
    if db.engine.dialect.name == 'sqlite':
        pytest.skip('SQLite uses its existing native backup path')
    from nndeploy.server import database_backup
    with empty_database(db, tmp_path, 'source') as source:
        with source.engine.begin() as connection:
            connection.exec_driver_sql('CREATE TABLE preserved (id INTEGER PRIMARY KEY)')
            connection.exec_driver_sql('INSERT INTO preserved VALUES (42)')
        def fail(*args, **kwargs):
            kwargs['stdout'].write(b'partial dump')
            raise RuntimeError('Controlled backup failure')
        monkeypatch.setattr(database_backup, '_run', fail)
        with pytest.raises(RuntimeError, match='Controlled backup failure'):
            source.init_schema()
        assert not list((tmp_path / 'backups').iterdir())
        with source.engine.connect() as connection:
            assert inspect(connection).get_table_names() == ['preserved']
            assert connection.exec_driver_sql('SELECT id FROM preserved').scalar() == 42


def test_database_backup_and_restore_cli_starts_no_server(tmp_path):
    source, target = tmp_path / 'source.db', tmp_path / 'target.db'
    database = DB(source)
    workflow, task, path = seed(database, tmp_path)
    database.close()
    destination = tmp_path / 'snapshot.sqlite3'
    environment = dict(os.environ, NNDEPLOY_BACKUP_TEST_URL='sqlite:///' + str(source))
    base = [sys.executable, '-m', 'nndeploy.server', '--database-url-env', 'NNDEPLOY_BACKUP_TEST_URL']
    result = subprocess.run(base + ['--database-backup', str(destination)], env=environment, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()
    assert 'Uvicorn running' not in result.stdout.decode() + result.stderr.decode()
    environment['NNDEPLOY_BACKUP_TEST_URL'] = 'sqlite:///' + str(target)
    result = subprocess.run(base + ['--database-restore', str(destination)], env=environment, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()
    restored = DB(target)
    try:
        assert restored.get_workflow_path(workflow) == path
        assert restored.get_task(task)['state'] == 'succeeded'
    finally:
        restored.close()


def test_failed_migration_keeps_restorable_backup_and_can_retry(db, tmp_path, monkeypatch):
    from alembic import command
    from alembic.migration import MigrationContext
    from sqlalchemy import update, MetaData, Table
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        workflow, task, path = seed(source, tmp_path)
        with source.engine.begin() as connection:
            version = Table('alembic_version', MetaData(), autoload_with=connection)
            connection.execute(update(version).values(version_num='0029_editor_output_files'))
            connection.exec_driver_sql('DROP TABLE server_task_inputs')
            connection.exec_driver_sql('ALTER TABLE server_tasks DROP COLUMN input_state')
        upgrade = command.upgrade
        def interrupted(config, revision):
            upgrade(config, revision)
            raise RuntimeError('Controlled failure after migration DDL')
        source._migrated = False
        monkeypatch.setattr(command, 'upgrade', interrupted)
        with pytest.raises(RuntimeError, match='Controlled failure'):
            source.init_schema()
        backups = list((tmp_path / 'backups').glob('*')) if source.engine.dialect.name != 'sqlite' else list(tmp_path.glob('source.sqlite3.backup-*.sqlite3'))
        assert len(backups) == 1
        target.restore(backups[0])
        assert target.get_workflow_path(workflow) == path
        with target.engine.connect() as connection:
            assert MigrationContext.configure(connection).get_current_revision() == '0029_editor_output_files'
            assert 'input_state' not in {column['name'] for column in inspect(connection).get_columns('server_tasks')}
        monkeypatch.setattr(command, 'upgrade', upgrade)
        source.init_schema()
        assert source.get_task(task)['outputs'] == {'text': 'x' * 70000}
        assert source.get_workflow_path(workflow) == path
        assert not list(tmp_path.rglob('.database-backup-*'))


def test_native_backup_deadline_stops_client_and_discards_partial_file(db, tmp_path, monkeypatch):
    if db.engine.dialect.name == 'sqlite':
        pytest.skip('Native command supervision applies to MySQL/PostgreSQL')
    from nndeploy.server import database_backup
    client = tmp_path / 'slow-client'
    marker = tmp_path / 'pid'
    finished = tmp_path / 'finished'
    client.write_text('#!' + sys.executable + '\nimport os,time,sys\nfrom pathlib import Path\n' +
        f'Path({str(marker)!r}).write_text(str(os.getpid()))\n' +
        'sys.stdout.buffer.write(b"partial");sys.stdout.flush()\ntime.sleep(30)\n' +
        f'Path({str(finished)!r}).touch()\n')
    client.chmod(0o700)
    monkeypatch.setattr(database_backup, '_program', lambda _: str(client))
    with pytest.raises(TimeoutError):
        db.backup(tmp_path / 'never-published', timeout_seconds=3)
    assert marker.exists() and not finished.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    assert not (tmp_path / 'never-published').exists()
    assert not list(tmp_path.glob('.database-backup-*'))


@pytest.mark.parametrize('backend', ['postgresql', 'mysql'])
def test_native_client_credentials_stay_private(tmp_path, backend):
    from nndeploy.server.database_backup import _client
    from sqlalchemy.engine import URL
    password = 'test-credential:quotes\'"\\#'
    database = DB(URL.create(backend, username='test', password=password, host='127.0.0.1', database='test_db').render_as_string(hide_password=False))
    try:
        with _client(database, 'pg_dump' if backend == 'postgresql' else 'mysqldump', tmp_path) as (command, environment):
            assert all(password not in part for part in command)
            assert password not in str(environment)
            for path in tmp_path.iterdir():
                if os.name != 'nt':
                    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        database.close()


def test_sqlite_restore_rejects_same_empty_database_file(tmp_path):
    database = DB(tmp_path / 'empty.db')
    try:
        database.ping()
        with pytest.raises(ValueError, match='different files'):
            database.restore(database.db_path)
    finally:
        database.close()


def test_incomplete_mysql_dump_is_rejected_before_creating_tables(db, tmp_path):
    if db.engine.dialect.name != 'mysql':
        pytest.skip('MySQL SQL dumps need an explicit completion check')
    with empty_database(db, tmp_path, 'source') as source, empty_database(db, tmp_path, 'target') as target:
        seed(source, tmp_path)
        snapshot = source.backup(tmp_path / 'snapshot.sql')
        data = snapshot.read_bytes()
        offset = data.rfind(b'-- Dump completed on ')
        assert offset > 0
        snapshot.write_bytes(data[:offset])
        with pytest.raises(ValueError, match='incomplete'):
            target.restore(snapshot)
        with target.engine.connect() as connection:
            assert inspect(connection).get_table_names() == []


def migrate_with_backup(options):
    location, directory = options
    database = DB(location, backup_dir=directory)
    try:
        database.init_schema()
    finally:
        database.close()


def test_concurrent_process_upgrades_create_one_backup(db, tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from sqlalchemy import MetaData, Table, update
    with empty_database(db, tmp_path, 'source') as source:
        workflow, task, path = seed(source, tmp_path)
        with source.engine.begin() as connection:
            version = Table('alembic_version', MetaData(), autoload_with=connection)
            connection.execute(update(version).values(version_num='0029_editor_output_files'))
            connection.exec_driver_sql('DROP TABLE server_task_inputs')
            connection.exec_driver_sql('ALTER TABLE server_tasks DROP COLUMN input_state')
        options = (source.url.render_as_string(hide_password=False), str(tmp_path / 'backups'))
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            assert list(pool.map(migrate_with_backup, [options] * 3)) == [None] * 3
        backups = list((tmp_path / 'backups').glob('*')) if source.engine.dialect.name != 'sqlite' else list(tmp_path.glob('source.sqlite3.backup-*.sqlite3'))
        assert len(backups) == 1
        assert source.get_workflow_path(workflow) == path
        assert source.get_task(task)['state'] == 'succeeded'


def test_native_clients_authenticate_with_escaped_passwords(db, tmp_path):
    if db.engine.dialect.name == 'sqlite':
        pytest.skip('SQLite has no database login')
    from sqlalchemy.exc import SQLAlchemyError
    postgres = db.engine.dialect.name == 'postgresql'
    username = 'backup_' + uuid.uuid4().hex[:16]
    password = 'test:' + uuid.uuid4().hex + '\'"\\#'
    hba = original_hba = None
    created = False
    if postgres:
        with db.engine.connect() as connection:
            hba = Path(connection.exec_driver_sql('SHOW hba_file').scalar())
        if 'nndeploy-database-integration-' not in str(hba):
            pytest.skip('Authentication fault test only edits the disposable integration cluster')
        original_hba = hba.read_text()
    try:
        with empty_database(db, tmp_path, 'source') as raw_source, empty_database(db, tmp_path, 'target') as raw_target:
            with db.engine.begin() as connection:
                if postgres:
                    from psycopg import sql
                    with connection.connection.driver_connection.cursor() as cursor:
                        cursor.execute(sql.SQL('CREATE ROLE {} LOGIN PASSWORD {}').format(sql.Identifier(username), sql.Literal(password)))
                else:
                    connection.execute(text(f"CREATE USER '{username}'@'%' IDENTIFIED BY :password"), {'password': password})
                created = True
                for database in (raw_source, raw_target):
                    quote = db.engine.dialect.identifier_preparer.quote(database.url.database)
                    if postgres:
                        connection.exec_driver_sql('ALTER DATABASE ' + quote + ' OWNER TO ' + username)
                    else:
                        connection.execute(text(f"GRANT ALL ON {quote}.* TO '{username}'@'%'"))
            if postgres:
                hba.write_text(f'host all {username} 127.0.0.1/32 scram-sha-256\n' + original_hba)
                with db.engine.begin() as connection:
                    assert connection.exec_driver_sql('SELECT pg_reload_conf()').scalar()
            location = raw_source.url.set(username=username, password=password)
            wrong = DB(location.set(password='incorrect').render_as_string(hide_password=False))
            try:
                deadline = time.monotonic() + 5
                while True:
                    try:
                        wrong.ping()
                    except SQLAlchemyError:
                        break
                    wrong.engine.dispose()
                    assert time.monotonic() < deadline, 'Disposable database did not require the configured password'
                    time.sleep(.05)
            finally:
                wrong.close()
            source = DB(location.render_as_string(hide_password=False), backup_dir=tmp_path / 'backups')
            target = DB(raw_target.url.set(username=username, password=password).render_as_string(hide_password=False))
            try:
                workflow, task, path = seed(source, tmp_path)
                snapshot = source.backup(tmp_path / ('authenticated' + SUFFIXES[db.engine.dialect.name]))
                target.restore(snapshot)
                assert target.get_workflow_path(workflow) == path
                assert target.get_task(task)['state'] == 'succeeded'
            finally:
                source.close()
                target.close()
    finally:
        if hba is not None:
            hba.write_text(original_hba)
            with db.engine.begin() as connection:
                connection.exec_driver_sql('SELECT pg_reload_conf()')
        if created:
            with db.engine.begin() as connection:
                if postgres:
                    connection.exec_driver_sql('DROP ROLE ' + username)
                else:
                    connection.execute(text(f"DROP USER '{username}'@'%'"))

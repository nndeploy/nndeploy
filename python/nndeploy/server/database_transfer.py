"""Portable Server records, using the existing schema and no executable SQL.

Run during maintenance with all Server instances stopped. Files, Redis and
physical runtime exit evidence remain in their existing storage locations.
"""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Boolean, DateTime, Integer, BigInteger, Float, String, inspect, select, insert, text

MAX_LINE_BYTES = 128 * 1024 * 1024


def _tables():
    # These modules register their existing tables, without constructing Server.
    from . import audit, control, devices, shared
    from .db import metadata
    return metadata.sorted_tables


def _header(tables):
    config = Config()
    config.set_main_option('script_location', str(Path(__file__).parent / 'migrations'))
    return {'format': 'nndeploy-server-database', 'version': 1,
            'revisions': sorted(ScriptDirectory.from_config(config).get_heads()),
            'tables': {table.name: list(table.c.keys()) for table in tables}}


def _deadline(seconds):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 86400:
        raise ValueError('Database transfer timeout must be finite, positive and at most 86400 seconds')
    end = time.monotonic() + seconds
    def check():
        if time.monotonic() >= end:
            raise TimeoutError('Database transfer deadline exceeded')
    return check


def _schema(connection, tables):
    expected = _header(tables)
    if sorted(MigrationContext.configure(connection).get_current_heads()) != expected['revisions']:
        raise ValueError('Database transfer requires the current Server schema; upgrade it first')
    inspector = inspect(connection)
    if set(inspector.get_table_names()) != set(expected['tables']) | {'alembic_version'} or inspector.get_view_names():
        raise ValueError('Database transfer requires exactly the Server tables, without additional tables or views')
    if connection.dialect.name == 'postgresql' and (inspector.get_materialized_view_names() or inspector.get_foreign_table_names()):
        raise ValueError('Database transfer does not support materialized views or foreign tables')
    for table in tables:
        if {column['name'] for column in inspector.get_columns(table.name)} != set(table.c.keys()):
            raise ValueError('Database columns do not match the current Server schema: ' + table.name)
    # Triggers can silently change imported values or write outside these tables.
    backend = connection.dialect.name
    if backend == 'mysql':
        nontransactional = connection.exec_driver_sql(
            "SELECT 1 FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() "
            "AND UPPER(ENGINE)<>'INNODB' LIMIT 1").first()
        if nontransactional or connection.exec_driver_sql('SELECT @@FOREIGN_KEY_CHECKS').scalar() != 1:
            raise ValueError('Database transfer requires InnoDB tables with foreign key checks enabled')
    query = {
        'sqlite': "SELECT 1 FROM sqlite_master WHERE type='trigger' LIMIT 1",
        'postgresql': "SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid "
                      "JOIN pg_namespace n ON n.oid=c.relnamespace "
                      "WHERE NOT t.tgisinternal AND n.nspname=current_schema() LIMIT 1",
        'mysql': 'SELECT 1 FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA=DATABASE() LIMIT 1',
    }[backend]
    if connection.exec_driver_sql(query).first():
        raise ValueError('Database transfer does not support custom triggers')
    return expected


def _transaction(connection, *, writing=False):
    backend = connection.dialect.name
    if backend == 'sqlite':
        connection.exec_driver_sql('BEGIN IMMEDIATE' if writing else 'BEGIN')
    else:
        # schema_lock acquired and committed its advisory lock before yielding
        # this new transaction. Establish isolation before the first read.
        isolation = ('READ COMMITTED' if backend == 'postgresql' else 'SERIALIZABLE') if writing else 'REPEATABLE READ'
        # PostgreSQL must see commits preceding _empty's table lock, including
        # a writer that committed after our schema checks established a snapshot.
        connection.exec_driver_sql('SET TRANSACTION ISOLATION LEVEL ' + isolation)


def _empty(connection, tables):
    if connection.dialect.name == 'postgresql':
        quote = connection.dialect.identifier_preparer.quote
        connection.exec_driver_sql('LOCK TABLE ' + ', '.join(quote(t.name) for t in tables) + ' IN EXCLUSIVE MODE')
    for table in tables:
        if connection.execute(select(*table.primary_key.columns).limit(1).with_for_update()).first():
            raise ValueError('Database import requires an empty destination; existing records are never overwritten')


def _value(column, value, *, reading):
    if value is None:
        if not column.nullable:
            raise ValueError('Database archive contains null for a required column')
        return None
    kind = column.type
    if isinstance(kind, Boolean):
        valid = type(value) is bool
    elif isinstance(kind, Integer):
        bits = 64 if isinstance(kind, BigInteger) else 32
        valid = type(value) is int and -(2 ** (bits - 1)) <= value < 2 ** (bits - 1)
    elif isinstance(kind, Float):
        valid = type(value) in (int, float) and math.isfinite(value)
    elif isinstance(kind, DateTime):
        encoded = value
        if reading:
            try:
                value = datetime.fromisoformat(value) if isinstance(value, str) else None
            except ValueError:
                value = None
        valid = isinstance(value, datetime) and value.tzinfo is None
        if valid and reading:
            valid = encoded == value.isoformat(timespec='microseconds')
        if valid and not reading:
            return value.isoformat(timespec='microseconds')
    elif isinstance(kind, String):
        valid = isinstance(value, str) and (kind.length is None or len(value) <= kind.length)
    else:
        raise ValueError('Unsupported database column type: ' + column.name)
    if not valid:
        raise ValueError('Invalid database archive value for ' + column.table.name + '.' + column.name)
    return value


def _line(value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
    if len(raw) > MAX_LINE_BYTES:
        raise ValueError('Database archive row exceeds 128 MiB')
    return raw


def _records(connection, tables, write):
    write(_header(tables))
    total = 0
    for table in tables:
        write({'table': table.name})
        count = 0
        with connection.execute(select(table).order_by(*table.primary_key.columns)
                                .execution_options(yield_per=1)) as rows:
            for row in rows:
                write([_value(column, value, reading=False) for column, value in zip(table.c, row)])
                count += 1
        write({'rows': count})
        total += count
    return total


def export_data(db, destination, *, timeout_seconds=300):
    from .migrations.lock import schema_lock
    check = _deadline(timeout_seconds)
    tables = _tables()
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Database export destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.database-export-', dir=destination.parent) as directory:
        temporary = Path(directory) / 'records.jsonl'
        with temporary.open('xb') as output:
            temporary.chmod(0o600)
            digest = hashlib.sha256()
            def write(value):
                check()
                raw = _line(value)
                output.write(raw)
                digest.update(raw)
            with schema_lock(db) as connection:
                _transaction(connection)
                _schema(connection, tables)
                total = _records(connection, tables, write)
                output.write(_line({'sha256': digest.hexdigest(), 'rows': total}))
            output.flush()
            os.fsync(output.fileno())
        check()
        # Publish a complete archive without overwriting a concurrent creator.
        os.link(temporary, destination)
        if os.name != 'nt':
            fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    return destination


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('Duplicate key in database archive')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('Non-finite number in database archive')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Invalid database archive JSON') from error


def _read(archive, tables, check, consume=None):
    digest = hashlib.sha256()
    def read(*, hashed=True):
        check()
        raw = archive.readline(MAX_LINE_BYTES + 1)
        if not raw or not raw.endswith(b'\n') or len(raw) > MAX_LINE_BYTES:
            raise ValueError('Truncated or oversized database archive record')
        value = _json(raw)
        if hashed:
            digest.update(raw)
        return value
    if read() != _header(tables):
        raise ValueError('Database archive format or schema does not match this Server version')
    total = 0
    for table in tables:
        if read() != {'table': table.name}:
            raise ValueError('Unexpected database archive table')
        count = 0
        while True:
            values = read()
            if isinstance(values, dict):
                if set(values) != {'rows'} or type(values['rows']) is not int or values['rows'] != count:
                    raise ValueError('Database archive row count mismatch')
                break
            if not isinstance(values, list) or len(values) != len(table.c):
                raise ValueError('Invalid database archive row')
            values = {column.name: _value(column, value, reading=True) for column, value in zip(table.c, values)}
            if consume:
                consume(table, values)
            count += 1
        total += count
    footer = read(hashed=False)
    if (not isinstance(footer, dict) or set(footer) != {'sha256', 'rows'} or
            type(footer['rows']) is not int or footer != {'sha256': digest.hexdigest(), 'rows': total}):
        raise ValueError('Database archive checksum or total count mismatch')
    if archive.read(1):
        raise ValueError('Unexpected data after database archive footer')
    return digest.hexdigest()


def import_data(db, source, *, timeout_seconds=300):
    from .migrations.lock import schema_lock
    check = _deadline(timeout_seconds)
    tables = _tables()
    source = Path(source)
    # A private immutable copy avoids validating one file then importing a
    # replacement. Validation finishes before even creating the target schema.
    with tempfile.TemporaryFile('w+b') as archive, source.open('rb') as incoming:
        while True:
            check()
            chunk = incoming.read(1024 * 1024)
            if not chunk:
                break
            archive.write(chunk)
        archive.seek(0)
        _read(archive, tables, check)
        with schema_lock(db) as connection:
            inspector = inspect(connection)
            fresh = not inspector.get_table_names() and not inspector.get_view_names()
            if not fresh:
                _schema(connection, tables)
        if fresh:
            db.init_schema()
        with schema_lock(db) as connection:
            _transaction(connection, writing=True)
            _schema(connection, tables)
            _empty(connection, tables)
            archive.seek(0)
            _read(archive, tables, check, lambda table, row: connection.execute(insert(table).values(**row)))
            # Detect silent target coercion (e.g. a non-strict MySQL column or
            # a schema with altered precision) before committing any records.
            def verify(table, row):
                # Compare by the existing primary key, independent of backend
                # collation and row order. Memory stays bounded by one row.
                actual = connection.execute(select(table).where(*(
                    column == row[column.name] for column in table.primary_key.columns))).mappings().one()
                expected = [_value(column, row[column.name], reading=False) for column in table.c]
                found = [_value(column, actual[column.name], reading=False) for column in table.c]
                if _line(found) != _line(expected):
                    raise ValueError('Imported database values differ from the archive; transaction rolled back')
            archive.seek(0)
            _read(archive, tables, check, verify)
            if connection.dialect.name == 'postgresql':
                # Explicit IDs do not advance PostgreSQL's serial sequence.
                # Restart the existing sequence transactionally so rollback
                # also undoes this change (setval would not).
                for table in tables:
                    for column in table.c:
                        if column.autoincrement is True:
                            sequence = connection.execute(text('SELECT pg_get_serial_sequence(:table, :column)'),
                                                          {'table': table.name, 'column': column.name}).scalar_one()
                            if sequence:
                                from sqlalchemy import func
                                maximum = connection.execute(select(func.max(column))).scalar()
                                # pg_get_serial_sequence returns a SQL-qualified
                                # identifier; resolve and quote each actual name.
                                names = connection.execute(text('SELECT n.nspname, c.relname FROM pg_class c '
                                    'JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=CAST(:seq AS regclass)'),
                                    {'seq': sequence}).one()
                                quote = connection.dialect.identifier_preparer.quote
                                connection.exec_driver_sql('ALTER SEQUENCE ' + '.'.join(map(quote, names)) +
                                                           ' RESTART WITH ' + str((maximum or 0) + 1))
            check()
    return source

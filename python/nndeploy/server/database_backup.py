"""Native database backups; never publish an incomplete backup or overwrite data."""
from contextlib import closing, contextmanager
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time

from sqlalchemy import inspect


SUFFIXES = {'sqlite': '.sqlite3', 'postgresql': '.dump', 'mysql': '.sql'}


def _timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 86400:
        raise ValueError('Database backup/restore timeout must be finite, positive and at most 86400 seconds')
    return float(value)


def _program(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f'Database backup/restore requires {name} on PATH')
    return path


@contextmanager
def _client(db, program, directory):
    """Use the configured endpoint; credentials never appear in argv or errors."""
    url = db.url
    environment = dict(os.environ)
    if db.engine.dialect.name == 'postgresql':
        import psycopg.conninfo
        # libpq accepts connection parameters via a private service file. Build
        # the conninfo with Psycopg, matching the SQLAlchemy dialect options.
        _, options = db.engine.dialect.create_connect_args(url)
        options.pop('context', None)  # SQLAlchemy's Psycopg adapter map, not a libpq option.
        options['connect_timeout'] = max(2, math.ceil(db.timeout_seconds))
        conninfo = psycopg.conninfo.make_conninfo(**options)
        # PGSERVICEFILE uses literal values rather than conninfo quoting. A
        # private passfile handles passwords (including ':' and backslashes).
        parameters = psycopg.conninfo.conninfo_to_dict(conninfo)
        password = parameters.pop('password', None)
        service = Path(directory) / 'postgres.service'
        if any('\n' in str(value) or '\r' in str(value) or str(value).strip() != str(value) for value in parameters.values()):
            raise ValueError('Native PostgreSQL service parameters cannot contain newlines or surrounding whitespace')
        service.write_text('[nndeploy_backup]\n' + ''.join(f'{key}={value}\n' for key, value in parameters.items()))
        service.chmod(0o600)
        environment.update(PGSERVICEFILE=str(service), PGSERVICE='nndeploy_backup')
        if password is not None:
            if '\n' in password or '\r' in password:
                raise ValueError('Database backup credentials cannot contain newlines')
            escaped = password.replace('\\', '\\\\').replace(':', '\\:')
            passfile = Path(directory) / 'postgres.pass'
            passfile.write_text('*:*:*:*:' + escaped + '\n')
            passfile.chmod(0o600)
            environment.pop('PGPASSWORD', None)
            environment['PGPASSFILE'] = str(passfile)
        yield [_program(program), '--no-password'], environment
        return
    _, options = db.engine.dialect.create_connect_args(url)
    mapping = {'host': 'host', 'port': 'port', 'user': 'user', 'password': 'password',
               'passwd': 'password', 'db': 'database', 'database': 'database', 'unix_socket': 'socket',
               'charset': 'default-character-set', 'connect_timeout': 'connect-timeout'}
    ignored = {'read_timeout', 'write_timeout'}
    settings = {}
    for key, value in options.items():
        if key == 'ssl':
            ssl_mapping = {'ca': 'ssl-ca', 'capath': 'ssl-capath', 'cert': 'ssl-cert', 'key': 'ssl-key', 'cipher': 'ssl-cipher'}
            if not isinstance(value, dict) or set(value) - set(ssl_mapping) - {'check_hostname', 'verify_mode'}:
                raise ValueError('Unsupported MySQL TLS parameters for native backup')
            settings.update({ssl_mapping[name]: item for name, item in value.items() if name in ssl_mapping})
            settings['ssl-mode'] = 'VERIFY_IDENTITY' if value.get('check_hostname') else 'VERIFY_CA' if value.get('ca') else 'REQUIRED'
        elif key == 'client_flag' and value == 2:
            # SQLAlchemy requests FOUND_ROWS for update rowcounts; dump clients
            # do not perform application updates.
            continue
        elif key in mapping:
            settings[mapping[key]] = value
        elif key not in ignored:
            raise ValueError(f'Unsupported MySQL connection option for native backup: {key}')
    if program == 'mysql':
        settings['connect-timeout'] = max(1, math.ceil(db.timeout_seconds))
    else:
        # mysqldump does not accept mysql's connect-timeout option. The
        # supervised client process still has an overall execution deadline.
        settings.pop('connect-timeout', None)
    settings.setdefault('default-character-set', 'utf8mb4')
    # mysqldump takes the database as a positional argument, not a client option.
    database = settings.pop('database', None)
    if not database or str(database).startswith('-'):
        raise ValueError('Native MySQL backup requires an explicit database name')
    def quote(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r') + '"'
    defaults = Path(directory) / 'mysql.cnf'
    defaults.write_text('[client]\n' + ''.join(f'{key}={quote(value)}\n' for key, value in settings.items()))
    defaults.chmod(0o600)
    environment.pop('MYSQL_PWD', None)
    yield [_program(program), '--defaults-file=' + str(defaults), '--no-login-paths', '--database=' + str(database)] if program == 'mysql' else [
        _program(program), '--defaults-file=' + str(defaults), '--no-login-paths', str(database)], environment


def _run(command, environment, timeout, *, stdin=None, stdout=None):
    # Client diagnostics may contain SQL or connection details. Do not relay
    # them through Server logs, command lines or exceptions.
    try:
        result = subprocess.run(command, env=environment, stdin=stdin or subprocess.DEVNULL,
                                stdout=stdout or subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise TimeoutError('Database backup/restore client exceeded its execution deadline') from None
    if result.returncode:
        raise RuntimeError(f'Database backup/restore client failed (exit {result.returncode}); verify access, client version and connection settings')


def backup(db, connection, destination, *, timeout_seconds=300):
    timeout = _timeout(timeout_seconds)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Database backup destination already exists')
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='.database-backup-', dir=destination.parent) as directory:
        path = Path(directory) / 'backup'
        with path.open('xb') as stream:
            path.chmod(0o600)
            backend = db.engine.dialect.name
            if backend == 'sqlite':
                deadline = time.monotonic() + timeout
                def progress(*args):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('SQLite backup exceeded its execution deadline')
                with closing(sqlite3.connect(path)) as target:
                    connection.connection.driver_connection.backup(target, pages=256, progress=progress, sleep=.05)
            elif backend == 'postgresql':
                with _client(db, 'pg_dump', directory) as (command, environment):
                    _run(command + ['--format=custom', '--no-owner', '--no-privileges'], environment, timeout, stdout=stream)
            else:
                with _client(db, 'mysqldump', directory) as (command, environment):
                    _run(command[:-1] + ['--single-transaction', '--no-tablespaces', '--set-gtid-purged=OFF',
                        '--column-statistics=0', '--hex-blob', '--routines', '--events', '--triggers', command[-1]],
                        environment, timeout, stdout=stream)
            stream.flush()
            os.fsync(stream.fileno())
        if path.stat().st_size == 0:
            raise RuntimeError('Database backup client produced an empty file')
        # Hard-link publication is atomic and refuses a concurrent destination;
        # replace() would silently destroy another completed backup.
        os.link(path, destination)
        if os.name != 'nt':
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return destination


def restore(db, connection, source, *, timeout_seconds=300):
    timeout = _timeout(timeout_seconds)
    source = Path(source).resolve(strict=True)
    if db.db_path is not None and db.db_path.exists() and source.samefile(db.db_path):
        raise ValueError('Database restore source and target must be different files')
    inspector = inspect(connection)
    backend = db.engine.dialect.name
    occupied = inspector.get_table_names() or inspector.get_view_names()
    if backend == 'postgresql':
        occupied = occupied or connection.exec_driver_sql("""SELECT EXISTS (
            SELECT 1 FROM pg_namespace WHERE nspname NOT IN ('public', 'information_schema') AND left(nspname, 3) <> 'pg_'
            UNION ALL SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public'
            UNION ALL SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public')""").scalar()
    elif backend == 'mysql':
        occupied = occupied or connection.exec_driver_sql('SELECT COUNT(*) FROM information_schema.ROUTINES WHERE ROUTINE_SCHEMA=DATABASE()').scalar()
        occupied = occupied or connection.exec_driver_sql('SELECT COUNT(*) FROM information_schema.EVENTS WHERE EVENT_SCHEMA=DATABASE()').scalar()
    if occupied:
        raise ValueError('Database restore requires an empty target; existing data is never overwritten')
    # Native clients inherit the descriptor's kernel offset. Buffered header
    # inspection may read ahead even after seek(0) inside Python's buffer.
    with source.open('rb', buffering=0) as stream, tempfile.TemporaryDirectory(prefix='nndeploy-database-client-') as directory:
        magic = stream.read(16)
        stream.seek(0)
        if backend == 'sqlite':
            if magic != b'SQLite format 3\x00':
                raise ValueError('Expected a native SQLite backup')
            deadline = time.monotonic() + timeout
            def progress(*args):
                if time.monotonic() >= deadline:
                    raise TimeoutError('SQLite restore exceeded its execution deadline')
            with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as snapshot:
                snapshot.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10000)
                if snapshot.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                    raise ValueError('SQLite backup integrity check failed')
                snapshot.backup(connection.connection.driver_connection, pages=256, progress=progress, sleep=.05)
        elif backend == 'postgresql':
            if not magic.startswith(b'PGDMP'):
                raise ValueError('Expected a PostgreSQL custom-format backup')
            with _client(db, 'pg_restore', directory) as (command, environment):
                _run(command + ['--dbname=service=nndeploy_backup', '--single-transaction', '--exit-on-error',
                                '--no-owner', '--no-privileges'], environment, timeout, stdin=stream)
        else:
            if not magic.startswith(b'-- MySQL dump'):
                raise ValueError('Expected a native MySQL SQL dump')
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 4096))
            tail = stream.read().rstrip().splitlines()
            if not tail or not tail[-1].startswith(b'-- Dump completed on '):
                raise ValueError('MySQL backup is incomplete or lacks its completion footer')
            stream.seek(0)
            with _client(db, 'mysql', directory) as (command, environment):
                _run(command + ['--binary-mode'], environment, timeout, stdin=stream)
    return source

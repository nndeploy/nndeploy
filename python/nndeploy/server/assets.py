"""Local Graph resource lifetime; callers keep a lease while paths are usable.

Only copies recorded by this implementation are eligible for collection. OS
locks protect readers across threads/processes; saved configurations protect
restart paths after those readers exit. Untracked legacy copies stay intact.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid

from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from .control_guard import _lock


_held = set()  # Losing a Python reference is not proof that a Graph closed.
_name = re.compile(r'(?:[0-9a-f]{64}|\.(?:copy|materialize)-[A-Za-z0-9_-]+)\Z')


def _database_id(db):
    from .db import server_meta
    key = 'asset-database-id'
    with db.engine.connect() as connection:
        value = connection.execute(select(server_meta.c.value).where(server_meta.c.key == key)).scalar_one_or_none()
    if value is None:
        try:
            with db.engine.begin() as connection:
                connection.execute(insert(server_meta).values(key=key, value=uuid.uuid4().hex))
        except IntegrityError:
            pass
        with db.engine.connect() as connection:
            value = connection.execute(select(server_meta.c.value).where(server_meta.c.key == key)).scalar_one()
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('Invalid asset database identity')
    return value


def _directories(directory):
    directory = Path(directory).resolve()
    assets, locks = directory / 'assets', directory / 'asset-locks'
    for path in (assets, locks):
        if path.is_symlink():
            raise ValueError('Server resource directories cannot be symbolic links')
        path.mkdir(parents=True, exist_ok=True)
    return assets, locks


def _open(path):
    if path.is_symlink():
        raise ValueError('Graph resource locks cannot be symbolic links')
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError('Graph resource locks must be regular files')
    return os.fdopen(descriptor, 'r+b', buffering=0)


@contextmanager
def _exclusive(directory):
    # Serializing lock-file creation/removal prevents waiters on a deleted
    # inode. The per-asset locks remain held after this short critical section.
    with _open(directory / '.lock') as file:
        deadline = time.monotonic() + 30
        while True:
            try:
                _lock(file, shared=False)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Graph resource coordination is busy') from None
                time.sleep(.01)
        yield


def _read(file):
    # Windows locks the first byte. Metadata is coordinated by the directory
    # lock and must stay outside that range when other readers hold shared locks.
    file.seek(1)
    data = file.read(16385)
    value = json.loads(data) if data else None
    if value is not None and (len(data) > 16384 or not isinstance(value, dict) or
            set(value) not in ({'version', 'managed', 'databases'}, {'version', 'managed', 'databases', 'pending'}) or value['version'] != 1 or
            type(value['managed']) is not bool or not isinstance(value['databases'], list) or
            not isinstance(value.get('pending', []), list) or
            any(not isinstance(item, str) or not re.fullmatch('[0-9a-f]{32}', item)
                for item in value['databases'] + value.get('pending', []))):
        raise ValueError('Invalid Graph resource ownership record')
    return value


def _write(file, value):
    data = json.dumps(value, sort_keys=True).encode()
    if len(data) > 16384:
        raise ValueError('Too many databases share this resource directory')
    file.seek(1)
    file.write(data)
    file.truncate()
    os.fsync(file.fileno())


class AssetLease:
    def __init__(self, directory, db):
        self.assets, self.locks = _directories(directory)
        self.database = _database_id(db)
        self.files = {}
        self.mutex = threading.RLock()
        self.closed = False
        _held.add(self)

    def hold(self, path, *, created=False):
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(self.assets):
            raise ValueError('Graph resource must remain in Server assets')
        parts = path.relative_to(self.assets).parts
        if not parts or not _name.fullmatch(parts[0]):
            raise ValueError('Invalid managed Graph resource path')
        name = parts[0]
        with self.mutex:
            if self.closed:
                raise RuntimeError('Graph resource lease is closed')
            if name in self.files:
                return
            with _exclusive(self.locks):
                file = _open(self.locks / name)
                try:
                    _lock(file, shared=True)
                    value = _read(file)
                    if value is None:
                        # Existing copies have no reliable evidence about old
                        # readers. Do not adopt them merely because we read one.
                        value = dict(version=1, managed=created, databases=[])
                    value['databases'] = sorted(set(value['databases']) | {self.database})
                    _write(file, value)
                    self.files[name] = file
                except BaseException:
                    file.close()
                    raise

    def hold_values(self, value):
        if isinstance(value, dict):
            for item in value.values():
                self.hold_values(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.hold_values(item)
        elif isinstance(value, str) and Path(value).is_absolute():
            path = Path(os.path.abspath(value))
            if path.is_relative_to(self.assets) and path != self.assets:
                self.hold(path)

    @property
    def paths(self):
        with self.mutex:
            return [str(self.assets / name) for name in self.files]

    def begin_publication(self):
        token = uuid.uuid4().hex
        self._publication(token, add=True)
        return token

    def end_publication(self, token):
        self._publication(token, add=False)

    def _publication(self, token, *, add):
        # A database commit can outlive a crashed client. Record publication
        # intent before sending it, and clear only after a confirmed outcome.
        # Unknown writes retain their copies for explicit reconciliation.
        with self.mutex, _exclusive(self.locks):
            for file in self.files.values():
                value = _read(file)
                pending = set(value.get('pending', []))
                pending.add(token) if add else pending.discard(token)
                value['pending'] = sorted(pending)
                _write(file, value)

    @contextmanager
    def temporary(self, prefix):
        # Register before any bytes are copied. A crash can leave an empty
        # unregistered directory, but never an unprotected partial model.
        with tempfile.TemporaryDirectory(prefix=prefix, dir=self.assets) as directory:
            lease = object.__new__(AssetLease)
            lease.assets, lease.locks, lease.database = self.assets, self.locks, self.database
            lease.files, lease.mutex, lease.closed = {}, threading.RLock(), False
            _held.add(lease)
            try:
                lease.hold(directory, created=True)
                yield directory
            finally:
                # TemporaryDirectory removes the bytes before releasing the
                # lock; cleanup may remove the remaining empty record later.
                try:
                    shutil.rmtree(directory)
                finally:
                    lease.close()

    def close(self):
        with self.mutex:
            for file in self.files.values():
                file.close()
            self.files.clear()
            self.closed = True
            _held.discard(self)


def _referenced(db, directory, candidates):
    from .db import server_configs
    retained = set()
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and Path(value).is_absolute():
            path = Path(os.path.abspath(value))
            if path.is_relative_to(directory):
                parts = path.relative_to(directory).parts
                if parts and parts[0] in candidates:
                    retained.add(parts[0])
    # This runs AFTER acquiring exclusive locks on the candidates. Reading
    # before locking could miss a config committed just before its Worker exits.
    with db.engine.connect() as connection:
        for record in connection.execution_options(stream_results=True, yield_per=100).execute(select(server_configs.c.data)).scalars():
            visit(json.loads(record))
    for path in (directory.parent / 'entries').glob('*.json'):
        visit(json.loads(path.read_text()))
    return retained


class AssetCleanup:
    def __init__(self, directory, db):
        self.directory, self.db, self.cursor = Path(directory), db, ''
        self.mutex = threading.Lock()

    def cleanup(self, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Cleanup limit must be between 1 and 1000')
        result = dict(scanned=0, deleted=0, skipped=0, failed=0)
        if not (self.directory / 'asset-locks').exists():
            return result
        assets, locks = _directories(self.directory)
        database = _database_id(self.db)
        with self.mutex, _exclusive(locks):
            names = sorted(path.name for path in locks.iterdir() if _name.fullmatch(path.name) and path.name > self.cursor)[:limit]
            self.cursor = names[-1] if len(names) == limit else ''
            acquired = {}
            try:
                for name in names:
                    result['scanned'] += 1
                    file = None
                    try:
                        file = _open(locks / name)
                        _lock(file, shared=False)
                        value = _read(file)
                        if value is None or not value['managed'] or value['databases'] != [database] or value.get('pending'):
                            result['skipped'] += 1
                            file.close()
                            continue
                        acquired[name] = file
                    except BlockingIOError:
                        if file is not None:
                            file.close()
                        result['skipped'] += 1
                    except Exception:
                        if file is not None:
                            file.close()
                        result['failed'] += 1
                retained = _referenced(self.db, assets, acquired) if acquired else set()
                for name, file in acquired.items():
                    if name in retained:
                        result['skipped'] += 1
                        continue
                    target = assets / name
                    try:
                        if target.is_symlink():
                            raise ValueError('Managed resources cannot redirect outside Server assets')
                        if target.exists():
                            shutil.rmtree(target)
                            result['deleted'] += 1
                        file.close()
                        (locks / name).unlink()
                    except Exception:
                        result['failed'] += 1
            finally:
                for file in acquired.values():
                    file.close()
        return result

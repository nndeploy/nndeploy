"""Atomic managed files with a local cache and optional object storage."""
import hashlib
import json
import logging
import os
import socket
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError

from ..errors import ServerError
from ..metrics import observe_storage
from .media import validate
from .objects import ObjectBackend, StorageConfig


class LocalStorage:
    def __init__(self, root, db, max_bytes=None, config=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.config = config if isinstance(config, StorageConfig) else StorageConfig.model_validate(config or {})
        self.max_bytes = self.config.max_bytes if max_bytes is None else max_bytes
        self.backend = None if self.config.backend == 'local' else ObjectBackend(self.config)
        location = [self.config.backend, self.config.bucket, self.config.region, self.config.endpoint, self.config.prefix]
        self.backend_id = 'local' if self.backend is None else self.config.backend + ':' + hashlib.sha256(repr(location).encode()).hexdigest()
        if self.backend is None:
            from ..resources import server_scope
            self.scope = 'local:' + server_scope(self.root)
        else:
            self.scope = self.backend_id

    def _copy(self, stream, target, limit, reserve=None):
        digest, size = hashlib.sha256(), 0
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ServerError(413, 'file_too_large', 'File exceeds the upload limit')
            if reserve is not None:
                reserve(size)
            digest.update(chunk)
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
        return size, digest.hexdigest()

    @observe_storage
    def save(self, stream, media_type=None, *, owner=None, task_id=None, attempt=1, worker_token=None, editor_task=None):
        pending = destination = remote_key = journal = None
        identity = 'file_' + uuid.uuid4().hex
        quota = self.config.max_files_per_subject is not None or self.config.max_storage_bytes_per_subject is not None
        journal_data = None
        editor_journal = {} if editor_task is None else dict(editor_scope=editor_task[0], editor_task_id=editor_task[1])
        cleanup = True
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, prefix='.upload-', delete=False) as target:
                pending = Path(target.name)
                reserve = None
                if quota:
                    journal = self.root / ('.pending-' + identity + '.json')
                    journal_data = dict(file_id=identity, storage_key=pending.name, pending_key=pending.name,
                                        backend=self.backend_id, host=socket.gethostname(), pid=os.getpid(),
                                        owner_key=self.db.subject_key(owner), **editor_journal)
                    self._journal(journal, journal_data)
                    self.db.reserve_file(dict(file_id=identity, storage_key=pending.name,
                        media_type='application/octet-stream', created_at=time.time(), backend=self.backend_id,
                        owner=owner, storage_scope=self.scope), self.config.max_files_per_subject)
                    reserve = lambda size: self.db.grow_file(identity, owner, size, self.config.max_storage_bytes_per_subject)
                size, checksum = self._copy(stream, target, self.max_bytes, reserve=reserve)
            from .archive import MAX_BYTES
            suffix, media = validate(pending, media_type, max_archive_bytes=max(MAX_BYTES, self.max_bytes),
                max_duration_seconds=self.config.media_max_duration_seconds, max_dimension=self.config.media_max_dimension)
            destination = self.root / (identity + suffix)
            if quota:
                # Record both names before rename, so either side of a crash
                # can be reconciled without guessing whether bytes were moved.
                journal_data['storage_key'] = destination.name
                self._journal(journal, journal_data)
            pending.replace(destination)
            if not quota:
                journal = self.root / ('.pending-' + identity + '.json')
                journal_data = dict(file_id=identity, storage_key=destination.name, backend=self.backend_id,
                                    host=socket.gethostname(), pid=os.getpid(), **editor_journal)
                self._journal(journal, journal_data)
            if self.backend:
                remote_key = destination.name
                self.backend.put_file(remote_key, destination, media)
            record = dict(file_id=identity, storage_key=destination.name, media_type=media, size=size,
                          created_at=time.time(), sha256=checksum, backend=self.backend_id, owner=owner, storage_scope=self.scope)
            options = {}
            if quota:
                options['reserved'] = True
            if task_id is not None:
                options.update(task_id=task_id, attempt=attempt, worker_token=worker_token)
            if editor_task is not None:
                options['editor_task'] = editor_task
            if self.config.retention_seconds is not None:
                options['retention_seconds'] = self.config.retention_seconds
            record = self.db.create_file(record, **options) or record
            if journal:
                journal.unlink(missing_ok=True)
            return self.public(record)
        except BaseException as exc:
            cleanup = True
            if journal is not None:
                try:
                    # A lost DB acknowledgement can mean the row committed.
                    # Preserve content whenever the commit outcome is unknown.
                    if editor_task is not None:
                        record = self.db.reconcile_editor_file(identity, editor_task[0], editor_task[1])
                    else:
                        record = (self.db.reconcile_file_upload(identity, self.db.subject_key(owner)) if quota
                                  else self.db.get_file(identity))
                    cleanup = record is None or (quota and record['state'] == 'uploading')
                    if cleanup and remote_key is not None:
                        self.backend.delete(remote_key)
                    if cleanup:
                        if destination is not None:
                            destination.unlink(missing_ok=True)
                        if pending is not None:
                            pending.unlink(missing_ok=True)
                        if quota:
                            self.db.abort_file_upload(identity)
                    if journal:
                        journal.unlink(missing_ok=True)
                except Exception:
                    cleanup = False
                    if journal:
                        self._journal(journal, dict(journal_data, pid=None))
                    logging.error('File cleanup awaits recovery: %s', identity)
            if destination is not None and cleanup:
                destination.unlink(missing_ok=True)
            if isinstance(exc, SQLAlchemyError):
                raise ServerError(503, 'storage_recording_unavailable',
                                  'File metadata storage is unavailable', retryable=True,
                                  status_url=f'/v1/files/{identity}') from exc
            raise
        finally:
            if pending is not None and (not quota or cleanup):
                pending.unlink(missing_ok=True)

    @staticmethod
    def public(record):
        return {k: v for k, v in record.items() if k not in (
            'storage_key', 'backend', 'owner', 'owner_key', 'storage_scope', 'producer_task_id', 'cleanup_after', 'protected_until')}

    @staticmethod
    def _journal(path, value):
        pending = path.with_suffix('.tmp')
        with pending.open('w', encoding='utf-8') as output:
            json.dump(value, output)
            output.flush()
            os.fsync(output.fileno())
        pending.replace(path)

    @observe_storage
    def recover_pending(self, limit=100):
        """Reconcile interrupted writes without deleting live writers' files."""
        from itertools import islice
        recovered = 0
        for journal in islice(self.root.glob('.pending-*.json'), limit):
            try:
                value = json.loads(journal.read_text())
                if value['backend'] != self.backend_id or value['host'] != socket.gethostname():
                    continue
                pid = value.get('pid')
                if pid is not None:
                    if not isinstance(pid, int) or pid <= 0:
                        continue
                    try:
                        os.kill(pid, 0)
                        continue
                    except ProcessLookupError:
                        pass
                key = value['storage_key']
                if Path(key).name != key or '\\' in key or key in ('.', '..'):
                    continue
                pending_key = value.get('pending_key')
                if pending_key is not None and (Path(pending_key).name != pending_key or '\\' in pending_key
                                                or not pending_key.startswith('.upload-')):
                    continue
                if value.get('editor_scope') and value.get('editor_task_id'):
                    record = self.db.reconcile_editor_file(value['file_id'], value['editor_scope'], value['editor_task_id'])
                elif pending_key:
                    owner_key = value.get('owner_key')
                    if not isinstance(owner_key, str) or len(owner_key) != 64 or any(c not in '0123456789abcdef' for c in owner_key):
                        continue
                    record = self.db.reconcile_file_upload(value['file_id'], owner_key)
                else:
                    record = self.db.get_file(value['file_id'])
                uploading = record is not None and record['state'] == 'uploading'
                if uploading and (record['backend'] != self.backend_id or record['storage_scope'] != self.scope
                                  or record['storage_key'] not in (key, pending_key)):
                    continue
                if record is None or uploading:
                    if self.backend:
                        self.backend.delete(key)
                    (self.root / key).unlink(missing_ok=True)
                    if pending_key:
                        (self.root / pending_key).unlink(missing_ok=True)
                    if uploading:
                        self.db.abort_file_upload(value['file_id'])
                elif record['storage_key'] != key or record['backend'] != self.backend_id:
                    continue
                journal.unlink(missing_ok=True)
                recovered += 1
            except Exception:
                logging.error('Could not recover pending file write %s', journal.name)
        return recovered

    def _record(self, identity, *, deleting=False):
        record = self.db.get_file(identity)
        if record is None:
            raise ServerError(404, 'file_not_found', 'Unknown file')
        if record.get('state', 'ready') not in (('ready', 'deleting') if deleting else ('ready',)):
            raise ServerError(404, 'file_not_found', 'File is being deleted')
        if (record.get('backend') or 'local') != self.backend_id:
            raise ServerError(503, 'storage_mismatch', 'The file belongs to another storage configuration')
        if record.get('storage_scope') not in (None, self.scope):
            raise ServerError(503, 'storage_mismatch', 'The file belongs to another storage location')
        key = record['storage_key']
        if not key or Path(key).name != key or key in ('.', '..') or '\\' in key:
            raise ServerError(500, 'invalid_storage_key', 'Invalid stored file location')
        return record, self.root / key

    @observe_storage
    def resolve(self, identity):
        record, path = self._record(identity)
        if path.is_symlink() or path.resolve().parent != self.root:
            raise ServerError(404, 'file_not_found', 'File content is missing')
        if not path.is_file() and self.backend:
            pending = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.root, prefix='.download-', delete=False) as target:
                    pending = Path(target.name)
                    with self.backend.open(record['storage_key']) as source:
                        size, checksum = self._copy(source, target, min(self.max_bytes, record['size']))
                if size != record['size'] or checksum != record['sha256']:
                    raise ServerError(502, 'content_mismatch', 'Stored file failed integrity validation')
                pending.replace(path)
            finally:
                if pending is not None:
                    pending.unlink(missing_ok=True)
        if not path.is_file():
            raise ServerError(404, 'file_not_found', 'File content is missing')
        return path, record

    def metadata(self, identity):
        return self.public(self._record(identity)[0])

    def resolve_input(self, value, field):
        """Resolve a request input without importing local files into storage."""
        if 'path' not in value:
            return self.resolve(value['file_id'])
        path = Path(value['path'])
        try:
            import stat
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ServerError(422, 'invalid_file', f'Local input must be a regular file: {path}')
            if info.st_size > self.max_bytes:
                raise ServerError(413, 'file_too_large', 'Local file exceeds the file byte limit')
            # Check readability here, before accepting a queued request.
            with path.open('rb'):
                pass
        except OSError as exc:
            raise ServerError(422, 'invalid_file', f'Local file does not exist or cannot be read: {path}') from exc
        media = field.media_type or ('image/*' if field.type == 'image' else 'application/octet-stream')
        _, media = validate(path, media)
        record = {'media_type': media, 'size': info.st_size}
        field.check_file(record)
        return path, record

    @contextmanager
    def open(self, identity):
        """Read managed content while protecting it from concurrent deletion."""
        holder = uuid.uuid4().hex
        try:
            self.db.hold_files(holder, {'file_id': identity})
            path, record = self.resolve(identity)
            with path.open('rb') as stream:
                yield stream
        finally:
            self.db.release_files(holder)

    @observe_storage
    def remove(self, identity):
        record, path = self._record(identity, deleting=True)
        if record.get('storage_scope') is None:
            if self.backend is None:
                if not path.is_file() or path.is_symlink() or path.resolve().parent != self.root:
                    raise ServerError(503, 'storage_mismatch', 'Legacy local file location cannot be verified')
                digest = hashlib.sha256()
                size = 0
                with path.open('rb') as content:
                    while chunk := content.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                if size != record['size'] or (record['sha256'] is not None and digest.hexdigest() != record['sha256']):
                    raise ServerError(503, 'storage_mismatch', 'Legacy local file integrity cannot be verified')
            self.db.bind_file_storage(identity, self.scope)
        self.db.mark_file_deleting(identity)
        self._delete_content(identity, record, path)

    def _delete_content(self, identity, record, path):
        if self.backend:
            self.backend.delete(record['storage_key'])
        path.unlink(missing_ok=True)
        return self.db.delete_file(identity)

    def discard_output(self, identity, task_id, *, attempt=1, worker_token=None):
        if self.db.discard_task_output(task_id, identity, attempt=attempt, worker_token=worker_token):
            self.remove(identity)

    @observe_storage
    def cleanup(self, limit=100):
        """Retry durable deletions and reclaim expired, unreferenced content."""
        # The existing periodic/management cleanup also reconciles dead uploads
        # so their reservations do not require an API process restart.
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('Cleanup limit must be between 1 and 1000')
        self.recover_pending(limit)
        identities = self.db.files_for_cleanup(self.scope, limit)
        result = dict(scanned=len(identities), deleted=0, skipped=0, failed=0)
        for identity in identities:
            try:
                if not self.db.claim_file_cleanup(identity, self.scope):
                    result['skipped'] += 1
                    continue
                record, path = self._record(identity, deleting=True)
                if self._delete_content(identity, record, path):
                    result['deleted'] += 1
                else:
                    result['skipped'] += 1
            except ServerError as exc:
                if exc.code in ('file_in_use', 'file_not_found'):
                    result['skipped'] += 1
                    continue
                result['failed'] += 1
                self._defer_cleanup(identity)
            except Exception:
                result['failed'] += 1
                self._defer_cleanup(identity)
        return result

    def _defer_cleanup(self, identity):
        logging.error('File cleanup awaits retry: %s', identity)
        try:
            self.db.defer_file_cleanup(identity, self.scope)
        except Exception:
            logging.error('Could not record file cleanup retry: %s', identity)

    @observe_storage
    def signed_get(self, identity, expires=300):
        record, _ = self._record(identity)
        if self.backend is None:
            raise ServerError(409, 'signed_url_unavailable', 'Local files use the authorized content endpoint')
        url = self.backend.signed_get(record['storage_key'], expires)
        self.db.protect_download(identity, self.scope, expires)
        return url

    def close(self):
        if self.backend:
            self.backend.close()

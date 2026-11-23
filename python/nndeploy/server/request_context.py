"""Request-scoped control and file artifacts for resident algorithm handlers."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

from .errors import ServerError


@dataclass(frozen=True)
class FileArtifact:
    path: str | Path
    media_type: str = 'application/octet-stream'
    include_local_path: bool = False


@dataclass
class RequestContext:
    task_id: str
    directory: Path
    metadata: dict
    check: object
    publish: object


_current = ContextVar('nndeploy_server_request', default=None)


def current_request():
    value = _current.get()
    if value is None:
        raise RuntimeError('No active Server request')
    return value


@contextmanager
def request_scope(executor, values, emit):
    from nndeploy.dag.execution import ExecutionContext, execution_scope
    metadata = values.get('_service') or {}
    identity = values.get('_task_id')
    checked = 0.0
    cancelled = False
    stream_deadline = (time.monotonic() + executor.config.streaming.max_duration_seconds
                       if executor.config.streaming.enabled else float('inf'))

    def check():
        nonlocal checked, cancelled
        interruption = getattr(executor, 'interruption', None)
        reason = interruption.value if interruption is not None else 0
        if reason == 1:
            raise ServerError(409, 'task_cancelled', 'Request cancelled.')
        if reason == 2 or (values.get('_execution_deadline') is not None and
                           time.monotonic() >= values['_execution_deadline']):
            raise ServerError(504, 'execution_timeout', 'Request exceeded its execution deadline.')
        if metadata.get('deadline') is not None and time.time() >= metadata['deadline']:
            raise ServerError(504, 'execution_timeout', 'Request timed out.')
        now = time.monotonic()
        if now >= stream_deadline:
            raise ServerError(504, 'execution_timeout', 'Stream duration exceeded the configured limit')
        if identity and not values.get('_warmup') and now - checked >= .1:
            task = executor.storage.db.get_task(identity)
            cancelled = task is None or task['cancel_requested']
            checked = now
        if cancelled:
            raise ServerError(409, 'task_cancelled', 'Request cancelled.')

    def publish(value):
        check()
        if emit is not None:
            emit(value)

    def probe():
        check()
        return False

    with tempfile.TemporaryDirectory(prefix='service-', dir=executor.workdir) as directory:
        context = RequestContext(identity, Path(directory), metadata, check, publish)
        token = _current.set(context)
        try:
            with execution_scope(ExecutionContext(cancelled=probe, publish=publish if emit else None)):
                check()
                yield context
                check()
        finally:
            _current.reset(token)


def save_artifacts(value, storage, values):
    """Stream nested file outputs to managed storage; never read whole videos."""
    saved = []
    def convert(item):
        if isinstance(item, FileArtifact):
            current_request().check()
            with Path(item.path).open('rb') as stream:
                class CheckedStream:
                    def read(self, size=-1):
                        current_request().check()
                        return stream.read(size)
                record = storage.save(CheckedStream(), item.media_type, owner=values.get('_owner'),
                    task_id=values['_task_id'], attempt=values.get('_attempt', 1),
                    worker_token=values.get('_worker_token'))
            saved.append(record['file_id'])
            result = {'file_id': record['file_id']}
            if item.include_local_path and storage.backend is None:
                result['local_path'] = str(storage.resolve(record['file_id'])[0])
            return result
        if isinstance(item, dict):
            return {key: convert(data) for key, data in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(data) for data in item]
        return item
    try:
        return convert(value)
    except BaseException:
        for identity in saved:
            storage.discard_output(identity, values['_task_id'], attempt=values.get('_attempt', 1),
                                   worker_token=values.get('_worker_token'))
        raise

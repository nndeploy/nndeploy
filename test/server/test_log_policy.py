"""Bounded diagnostics, credential redaction and saturated logging paths."""
import asyncio
import io
import json
import logging
import pickle
import queue
import subprocess
import sys
import threading
import time

import pytest

from nndeploy.server.app import _EditorLogListener
from nndeploy.server.log_policy import DiagnosticFormatter, MAX_LOG_BYTES, redact
from nndeploy.server.logging_taskid import EditorQueueHandler, _IOToLogger
from nndeploy.server.log_broadcast import LogBroadcaster
from nndeploy.server.metrics import Metrics


def record(message, task='abc123'):
    value = logging.LogRecord('algorithm', logging.INFO, __file__, 1, message, (), None)
    value.task_id = task
    return value


def test_sensitive_contexts_are_removed_before_queue_and_extras_never_cross_ipc():
    q = queue.Queue(1)
    handler = EditorQueueHandler(q)
    try:
        raise ValueError('password=exception-secret')
    except ValueError:
        value = record('Authorization: Bearer auth-secret\nCookie: session=cookie-secret\n'
                       '"api_key": "json-secret"\nhttps://user:url-password@example.com/x?token=query-secret\n'
                       'https://example.com?token=no-path-secret@remaining-secret\n'
                       'redis://user:redis-secret@localhost/0\nBearer standalone-secret')
        value.exc_info = sys.exc_info()
        value.payload = 'unstructured-extra-secret' * 100000
        value.not_picklable = lambda: None
        handler.handle(value)
    transferred = q.get_nowait()
    serialized = pickle.dumps(transferred)
    rendered = DiagnosticFormatter('%(message)s').format(transferred)
    for secret in ('auth-secret', 'cookie-secret', 'json-secret', 'url-password', 'query-secret',
                   'no-path-secret', 'remaining-secret', 'redis-secret', 'standalone-secret',
                   'exception-secret', 'unstructured-extra-secret'):
        assert secret not in rendered and secret.encode() not in serialized
    assert b'not_picklable' not in serialized and transferred.exc_info is None
    assert len(serialized) < MAX_LOG_BYTES + 4096
    assert 'ValueError' in rendered and '[REDACTED]' in rendered


def test_queue_full_drops_without_stderr_recursion_and_reports_loss(monkeypatch):
    class ForbiddenStderr:
        def write(self, value):
            raise AssertionError('Logging failure must not write back to redirected stderr')
        def flush(self):
            pass
    monkeypatch.setattr(sys, 'stderr', ForbiddenStderr())
    q = queue.Queue(1)
    handler = EditorQueueHandler(q)
    handler.handle(record('first'))
    for _ in range(25):
        handler.handle(record('password=must-not-escape'))
    assert handler.dropped_total == 25 and q.qsize() == 1
    q.get_nowait()
    event = record('ignored')
    event.server_event = {'event': 'test', 'task_id': 'task_example', 'status': 200,
                          'body': 'must-not-escape', 'authorization': 'must-not-escape'}
    handler.handle(event)
    received = q.get_nowait()
    value = json.loads(DiagnosticFormatter().format(received))
    assert value['logs_dropped'] == 25 and value['event'] == 'test'
    assert 'body' not in value and 'authorization' not in value
    assert handler.dropped == 0
    metrics = Metrics()
    listener = _EditorLogListener(q)
    listener.metrics = metrics
    listener.handle(received)
    assert metrics.log_dropped.labels('queue')._value.get() == 25


def test_utf8_records_and_unterminated_stdio_are_bounded_without_leaking_suffix():
    assert len(redact('中文' * 100000).encode()) <= MAX_LOG_BYTES
    q = queue.Queue(4)
    logger = logging.getLogger('bounded-stdio-test')
    old = logger.handlers[:], logger.propagate, logger.level
    logger.handlers, logger.propagate, logger.level = [EditorQueueHandler(q)], False, logging.INFO
    output = _IOToLogger(logging.INFO, logger.name)
    try:
        output.write('password=')
        for _ in range(100):
            output.write('continuation-secret' * 1024)
            assert len(output._buf) <= MAX_LOG_BYTES
        assert q.empty()
        output.write('\npublic line\n')
        secret, public = q.get_nowait(), q.get_nowait()
        assert 'continuation-secret' not in secret.getMessage()
        assert public.getMessage() == 'public line' and q.empty()
        output.write('partial public')
        output.flush()
        assert q.get_nowait().getMessage() == 'partial public'
    finally:
        logger.handlers, logger.propagate, logger.level = old


def test_multiline_quoted_credentials_and_private_key_blocks_are_redacted():
    text = 'password="first-secret\\"quoted\nsecond-secret"\npublic line\n'
    text += '-----BEGIN RSA PRIVATE KEY-----\nkey-secret\n-----END RSA PRIVATE KEY-----\n'
    cleaned = redact(text)
    assert all(value not in cleaned for value in ('first-secret', 'second-secret', 'key-secret'))
    assert 'public line' in cleaned
    assert 'suffix-secret' not in redact('token="' + 'suffix-secret\n' * 10000)


def test_captured_stream_redacts_credentials_split_across_writes_and_lines():
    q = queue.Queue(30)
    logger = logging.getLogger('split_credentials')
    old = logger.handlers, logger.propagate, logger.level
    logger.handlers, logger.propagate, logger.level = [EditorQueueHandler(q)], False, logging.INFO
    try:
        stream = _IOToLogger(logging.INFO, logger.name)
        for chunk in ('password="first-secret\\"quoted\n', 'second-secret', '\nlast-secret"\n',
                      'public one\n', '-----BEGIN RSA PRIVATE KEY-----\n',
                      'key-secret\n', 'other-key-secret\n-----END RSA PRIVATE KEY-----\n',
                      'public two\n'):
            stream.write(chunk)
        stream.flush()
        lines = []
        while not q.empty():
            lines.append(q.get_nowait().getMessage())
        output = '\n'.join(lines)
        for secret in ('first-secret', 'second-secret', 'last-secret', 'key-secret'):
            assert secret not in output
        assert 'public one' in output and 'public two' in output
    finally:
        logger.handlers, logger.propagate, logger.level = old


def test_listener_stops_and_drains_when_queue_has_no_room_for_sentinel():
    q = queue.Queue(3)
    entered, release = threading.Event(), threading.Event()
    received = []
    class Sink(logging.Handler):
        def emit(self, record):
            entered.set()
            assert release.wait(timeout=5)
            received.append(record.getMessage())
    listener = _EditorLogListener(q, Sink())
    listener.start()
    q.put(record('first'))
    assert entered.wait(timeout=2)
    for index in range(3):
        q.put_nowait(record(str(index)))
    stopped = threading.Thread(target=listener.stop)
    stopped.start()
    try:
        release.set()
        stopped.join(timeout=3)
        assert not stopped.is_alive() and listener._thread is None
        assert received == ['first', '0', '1', '2']
    finally:
        release.set()
        stopped.join(timeout=5)


def test_broadcast_cache_bounds_task_count_lines_and_total_utf8_bytes():
    broadcaster = LogBroadcaster(lambda: None, lambda: {})
    broadcaster.max_tasks, broadcaster.max_lines, broadcaster.max_bytes = 2, 3, 180
    broadcaster.setFormatter(DiagnosticFormatter('%(message)s'))
    for task in ('task_a', 'task_b', 'task_c'):
        for index in range(10):
            broadcaster.handle(record('中文' * 5 + str(index), task))
            assert len(broadcaster.task_log_map) <= 2
            assert broadcaster.cached_bytes <= 180
            assert all(len(lines) <= 3 for lines in broadcaster.task_log_map.values())
    result = broadcaster.get_logs('task_c')
    assert result and result[-1].endswith('9')
    result.clear()
    assert broadcaster.get_logs('task_c')
    broadcaster.close()
    assert not broadcaster.task_log_map and broadcaster.cached_bytes == 0


def test_slow_websocket_has_bounded_sends_timeout_and_no_recursive_logging(monkeypatch):
    def recurse(*args, **kwargs):
        raise AssertionError('A log sink must not log its own failures')
    monkeypatch.setattr(logging, 'warning', recurse)
    async def scenario():
        class Slow:
            async def send_json(self, value):
                await asyncio.Future()
        ws = Slow()
        metrics = Metrics()
        handler = LogBroadcaster(asyncio.get_running_loop, lambda: {'task_a': {ws}}, metrics=metrics)
        handler.max_pending, handler.send_timeout = 4, .05
        try:
            for _ in range(100):
                handler.handle(record('safe line', 'task_a'))
            assert len(handler.pending) == 4 and handler.dropped_sends == 96
            await asyncio.sleep(.2)
            assert not handler.pending and handler.failed_sends == 4
            assert metrics.log_dropped.labels('websocket')._value.get() == 100
            handler.handle(record('last', 'task_a'))
            handler.close()
            await asyncio.sleep(.02)
            assert not handler.pending
        finally:
            handler.close()
    asyncio.run(scenario())


def test_isolated_python_and_native_output_are_redacted_and_long_lines_bounded(tmp_path):
    script = tmp_path / 'logs.py'
    script.write_text('''import os, logging
from nndeploy.server.logging_taskid import configure_resident_logger, close_resident_logger
handler = configure_resident_logger()
print("password=python-secret")
os.write(1, b"Authorization: Bearer native-secret\\n")
os.write(2, b"token=" + b"native-continuation-secret" * 10000 + b"\\n")
os.write(1, b'password="native-first\\nnative-second"\\n')
os.write(1, b'-----BEGIN PRIVATE KEY-----\\nnative-key-secret\\n-----END PRIVATE KEY-----\\n')
try:
    raise RuntimeError("secret=trace-secret")
except RuntimeError:
    logging.exception("failed operation")
print("public complete")
close_resident_logger(handler)
''')
    result = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    output = result.stdout + result.stderr
    for secret in (b'python-secret', b'native-secret', b'native-continuation-secret', b'trace-secret',
                   b'native-first', b'native-second', b'native-key-secret'):
        assert secret not in output
    assert b'public complete' in output and b'[REDACTED]' in output
    assert len(output) < 2 * MAX_LOG_BYTES


def test_isolated_log_burst_does_not_block_compute_on_an_unread_console(tmp_path):
    script, marker = tmp_path / 'burst.py', tmp_path / 'computed'
    script.write_text('''from pathlib import Path
from nndeploy.server.logging_taskid import configure_resident_logger
configure_resident_logger()
for _ in range(5000):
    print("public " + "x" * 2048)
Path(__file__).with_name("computed").touch()
import time
time.sleep(30)
''')
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Deliberately never consume the pipes: the diagnostic writer stalls,
        # while the bounded queue drops logs and algorithm work can finish.
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline, 'Compute stalled behind diagnostic output'
            time.sleep(.02)
    finally:
        process.kill()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_server_reaps_isolated_worker_when_console_is_not_consumed(tmp_path):
    script = tmp_path / 'server_console.py'
    script.write_text('''import asyncio, os
from pathlib import Path
from nndeploy.server import Server

def deploy(values):
    for _ in range(5000):
        print("public " + "x" * 2048)
    return {"pid": os.getpid()}

async def main():
    root = Path(__file__).parent
    import logging.config
    from nndeploy.server.request import logging_config
    logging.config.dictConfig(logging_config())
    server = Server(deploy=deploy, config={"execution": {"isolation": "process"}}, resources=root, data_dir=root / "data")
    try:
        await server.start()
        (root / "started").touch()
        result = await server.deploy("default", {})
        pid = result["outputs"]["pid"]
        (root / "worker_pid").write_text(str(pid))
    finally:
        await server.close()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        (root / "closed").touch()
    else:
        raise AssertionError("Worker remains alive after Server.close")

if __name__ == "__main__":
    asyncio.run(main())
''')
    process = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # No reader drains the console, including during Worker shutdown.
        process.wait(timeout=15)
        assert process.returncode == 0
        assert (tmp_path / 'closed').exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_blocked_log_file_open_does_not_hold_setup_emit_or_close(tmp_path, monkeypatch):
    import nndeploy.server.log_policy as policy
    monkeypatch.setattr(policy, 'LOG_QUEUE_SIZE', 4)
    entered, release, configured, emitted, closed = (threading.Event() for _ in range(5))
    handlers = []

    class BlockedFile(policy.DiagnosticRotatingFileHandler):
        def _open(self):
            entered.set()
            assert release.wait(10)
            return super()._open()

    def produce():
        handler = BlockedFile(tmp_path / 'server.log', maxBytes=1000, backupCount=2, encoding='utf-8')
        handlers.append(handler)
        configured.set()
        handler.handle(record('first'))
        emitted.set()

    producer = threading.Thread(target=produce)
    closer = None
    producer.start()
    try:
        assert configured.wait(2), 'File open blocked logging setup'
        assert entered.wait(2)
        assert emitted.wait(2), 'File open blocked log emission'
        handler = handlers[0]
        for _ in range(20):
            handler.handle(record('queued'))
        assert handler.queue.qsize() == 4 and handler.dropped_total == 16
        def close():
            handler.close()
            closed.set()
        closer = threading.Thread(target=close)
        closer.start()
        assert closed.wait(2), 'Blocked file kept shutdown waiting'
        assert handler.queue.empty() and handler.dropped_total == 20
    finally:
        release.set()
        producer.join(5)
        if closer is not None:
            closer.join(5)
        for handler in handlers:
            handler.close()
            writer = getattr(handler, 'thread', None)
            if writer is not None:
                writer.join(5)


def test_redirected_regular_file_write_does_not_hold_log_emission(tmp_path):
    from nndeploy.server.log_policy import QueuedDiagnosticStreamHandler
    entered, release, emitted = (threading.Event() for _ in range(3))
    with (tmp_path / 'redirected.log').open('w') as stream:
        handler = QueuedDiagnosticStreamHandler(stream)
    write = handler._write_value
    def blocked(value):
        entered.set()
        assert release.wait(10)
        write(value)
    handler._write_value = blocked
    def produce():
        handler.handle(record('safe output'))
        emitted.set()
    producer = threading.Thread(target=produce)
    producer.start()
    try:
        assert entered.wait(2)
        assert emitted.wait(2), 'Regular file write blocked the caller'
    finally:
        release.set()
        producer.join(5)
        handler.close()
    assert (tmp_path / 'redirected.log').read_text() == 'safe output\n'


@pytest.mark.parametrize('phase', ['write', 'rotate'])
def test_blocked_file_write_and_rotation_do_not_block_other_sinks_or_shutdown(tmp_path, phase):
    from nndeploy.server.log_policy import DiagnosticRotatingFileHandler
    entered, release, stopped = (threading.Event() for _ in range(3))

    class BlockedFile(DiagnosticRotatingFileHandler):
        def _write_bytes(self, stream, data):
            if phase == 'write':
                entered.set()
                assert release.wait(10)
            return super()._write_bytes(stream, data)

        def doRollover(self):
            if phase == 'rotate':
                entered.set()
                assert release.wait(10)
            return super().doRollover()

    handler = BlockedFile(tmp_path / 'server.log', maxBytes=10, backupCount=2)
    other = io.StringIO()
    listener = _EditorLogListener(queue.Queue(4), handler, logging.StreamHandler(other))
    listener.start()
    listener.queue.put(record('first'))
    listener.queue.put(record('second'))
    closer = None
    try:
        assert entered.wait(2)
        def close():
            listener.stop()
            handler.close()
            stopped.set()
        closer = threading.Thread(target=close)
        closer.start()
        assert stopped.wait(2), 'File sink blocked listener shutdown'
        assert other.getvalue() == 'first\nsecond\n'
    finally:
        release.set()
        if closer is not None:
            closer.join(5)
        listener.stop()
        handler.close()
        handler.thread.join(5)
    assert not handler.thread.is_alive()


def test_queued_file_rotation_preserves_order_retention_and_redaction(tmp_path):
    from nndeploy.server.log_policy import DiagnosticRotatingFileHandler
    path = tmp_path / 'server.log'
    handler = DiagnosticRotatingFileHandler(path, maxBytes=20, backupCount=2)
    handler.setFormatter(DiagnosticFormatter('%(message)s'))
    for value in ('oldest', 'first', 'second', 'third', 'password=private-value'):
        handler.handle(record(value))
    handler.close()
    handler.thread.join(5)
    assert not handler.thread.is_alive()
    assert path.read_text() == 'password=[REDACTED]\n'
    assert path.with_suffix('.log.1').read_text() == 'second\nthird\n'
    assert path.with_suffix('.log.2').read_text() == 'oldest\nfirst\n'
    assert len(list(tmp_path.iterdir())) == 3
    assert handler.dropped_total == 0


def test_file_open_failure_is_counted_and_later_writes_recover(tmp_path):
    from nndeploy.server.log_policy import DiagnosticRotatingFileHandler
    attempted = threading.Event()
    class FailingFile(DiagnosticRotatingFileHandler):
        def _open(self):
            if not attempted.is_set():
                attempted.set()
                raise OSError('private-filesystem-error')
            return super()._open()
    path = tmp_path / 'server.log'
    handler = FailingFile(path)
    handler.metrics = Metrics()
    handler.handle(record('first'))
    assert attempted.wait(2)
    handler.handle(record('recovered'))
    handler.close()
    handler.thread.join(5)
    assert path.read_text() == 'recovered\n'
    assert handler.dropped_total == 1
    assert handler.metrics.log_dropped.labels('file')._value.get() == 1


def test_deploy_and_server_shutdown_finish_while_log_filesystem_is_blocked(tmp_path):
    from nndeploy.server import Server
    from nndeploy.server.log_policy import DiagnosticRotatingFileHandler
    entered, release = threading.Event(), threading.Event()
    class BlockedFile(DiagnosticRotatingFileHandler):
        def _open(self):
            entered.set()
            assert release.wait(10)
            return super()._open()
    handler = BlockedFile(tmp_path / 'log' / 'server.log')
    handler.setFormatter(DiagnosticFormatter('%(message)s'))
    logger = logging.getLogger('nndeploy.server.events')
    previous = logger.handlers[:], logger.level, logger.propagate
    logger.handlers, logger.propagate = [handler], False
    logger.setLevel(logging.INFO)

    async def scenario():
        server = Server(deploy=lambda values: values, resources=tmp_path / 'data', data_dir=tmp_path / 'data')
        await server.start()
        try:
            result = await asyncio.wait_for(server.deploy('default', {'text': 'correct'}), timeout=5)
            assert result['outputs'] == {'text': 'correct'}
            assert entered.is_set() and not release.is_set()
        finally:
            await asyncio.wait_for(server.close(), timeout=5)

    try:
        asyncio.run(scenario())
        handler.close()
        assert handler.thread.is_alive() and not release.is_set()
    finally:
        logger.handlers, logger.level, logger.propagate = previous
        release.set()
        handler.close()
        handler.thread.join(5)
    assert not handler.thread.is_alive()

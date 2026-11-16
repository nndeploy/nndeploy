# logging_taskid.py
import logging
from logging.handlers import QueueHandler, QueueListener
import contextvars
import sys
import os
import threading
from contextlib import contextmanager
from typing import Callable, Any
from queue import Empty, Full, Queue
from .log_policy import DiagnosticFormatter, transport_record


class EditorQueueHandler(QueueHandler):
    """A lost IPC process must not recurse through redirected stderr."""

    def __init__(self, queue):
        super().__init__(queue)
        self.dropped = 0
        self.dropped_total = 0

    def prepare(self, record):
        return transport_record(record, DiagnosticFormatter())

    def handleError(self, record):
        # QueueHandler's default writes the failed record to stderr, which is
        # redirected back into this very handler in the editor Worker.
        self.dropped += 1
        self.dropped_total += 1

    def enqueue(self, record):
        try:
            if self.dropped:
                from .log_policy import bounded
                record.msg = bounded(f'[logs dropped={self.dropped}] ' + record.getMessage())
                # Keep structured event JSON valid, with loss in a separate
                # transport attribute interpreted by the sink formatter.
                record.logs_dropped = self.dropped
            super().enqueue(record)
            self.dropped = 0
        except (Full, EOFError, OSError):
            # The parent supervises and replaces this queue. Its fallback
            # logger reports recovery; a dead channel cannot preserve logs.
            self.handleError(record)

class BoundedLogListener(QueueListener):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_requested = threading.Event()
        self.metrics = None

    def handle(self, record):
        if self.metrics is not None and getattr(record, 'logs_dropped', 0):
            self.metrics.log_dropped.labels('queue').inc(record.logs_dropped)
        super().handle(record)

    def _monitor(self):
        from .log_policy import LOG_QUEUE_SIZE
        remaining = LOG_QUEUE_SIZE
        try:
            while True:
                try:
                    if self.stop_requested.is_set():
                        if remaining <= 0:
                            break
                        record = self.queue.get_nowait()
                        remaining -= 1
                    else:
                        record = self.queue.get(True, .25)
                except Empty:
                    if self.stop_requested.is_set():
                        break
                    continue
                if record is self._sentinel:
                    if hasattr(self.queue, 'task_done'):
                        self.queue.task_done()
                    break
                self.handle(record)
                if hasattr(self.queue, 'task_done'):
                    self.queue.task_done()
        except (EOFError, OSError):
            pass  # The queue process exited; its supervisor restores logging.

    def stop(self):
        thread = self._thread
        if thread is None:
            return
        # No sentinel insertion into a full queue. Producers are detached by
        # the runtime first; drain at most its bounded capacity, then exit.
        self.stop_requested.set()
        thread.join()
        self._thread = None


# --- ContextVar ---
_TASK_ID = contextvars.ContextVar("task_id", default="0")
_REQUEST_ID = contextvars.ContextVar("request_id", default=None)


def get_request_id():
    return _REQUEST_ID.get()


def set_request_id(identity):
    return _REQUEST_ID.set(identity)


def reset_request_id(token):
    _REQUEST_ID.reset(token)

_TASK_ID_FALLBACK = "0"

def set_task_id(tid: str):
    return _TASK_ID.set(tid)

def get_task_id() -> str | None:
    """Return the current execution's task ID without using a process fallback."""
    value = _TASK_ID.get()
    return value if value != '0' else None

def set_task_id_fallback(tid: str):
    global _TASK_ID_FALLBACK
    _TASK_ID_FALLBACK = tid
    return _TASK_ID.set(tid)

def reset_task_id(token, *, clear_fallback: bool = False):
    try:
        _TASK_ID.reset(token)
    finally:
        if clear_fallback:
            global _TASK_ID_FALLBACK
            _TASK_ID_FALLBACK = "0"

@contextmanager
def task_log_context(task_id: str, *, use_fallback: bool = False):
    tok = set_task_id_fallback(task_id) if use_fallback else set_task_id(task_id)
    try:
        yield
    finally:
        reset_task_id(tok, clear_fallback=use_fallback)

def install_taskid_logrecord_factory() -> None:
    old = logging.getLogRecordFactory()
    if getattr(old, '_nndeploy_task_context', False):
        return
    def factory(*args, **kwargs):
        record = old(*args, **kwargs)
        try:
            tid = _TASK_ID.get()
        except Exception:
            tid = "0"
        if not tid or tid == "0":
            tid = _TASK_ID_FALLBACK

        record.task_id = tid
        record.request_id = get_request_id()
        if record.name != 'nndeploy.server.events' and tid and tid != "0" and not getattr(record, "_taskid_prefixed", False):
            msg = record.getMessage()
            record.msg = f"[task_id={tid}] {msg}"
            record.args = ()
            record._taskid_prefixed = True
        return record
    factory._nndeploy_task_context = True
    logging.setLogRecordFactory(factory)

def redirect_python_stdio(logger_name: str = "stdio") -> None:
    sys.stdout = _IOToLogger(logging.INFO, logger_name)
    sys.stderr = _IOToLogger(logging.ERROR, logger_name)


def configure_resident_logger():
    """Capture an isolated worker's Python/native output to its original sink."""
    from .log_policy import QueuedDiagnosticStreamHandler
    # Preserve the original descriptor before stdout/stderr become pipes.
    # Writing the reader's logs into either redirected fd would recurse.
    with os.fdopen(os.dup(2), 'wb', buffering=0) as stream:
        handler = QueuedDiagnosticStreamHandler(stream)
    handler.setFormatter(DiagnosticFormatter('%(levelname)s: %(message)s'))
    root = logging.getLogger()
    from .log_policy import LOG_QUEUE_SIZE
    queue = Queue(maxsize=LOG_QUEUE_SIZE)
    root.handlers = [EditorQueueHandler(queue)]
    root.setLevel(logging.INFO)
    listener = BoundedLogListener(queue, handler)
    listener.start()
    install_taskid_logrecord_factory()
    redirect_python_stdio()
    redirect_fd_to_logger_once()
    return listener

_FD_REDIRECTED = False
_FD_READERS = []
def redirect_fd_to_logger_once(logger_name: str = "fd") -> None:
    global _FD_REDIRECTED
    if _FD_REDIRECTED:
        return
    logger = logging.getLogger(logger_name)
    pipes = {}
    for fd, level in ((1, logging.INFO), (2, logging.ERROR)):
        r, w = os.pipe()
        os.dup2(w, fd)
        os.close(w)
        pipes[fd] = (r, level)
    for r, level in pipes.values():
        def reader(rr=r, lvl=level):
            import codecs
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            output = _IOToLogger(lvl, logger_name)
            try:
                while data := os.read(rr, 4096):
                    output.write(decoder.decode(data))
                output.write(decoder.decode(b'', final=True))
                output.flush()
            finally:
                os.close(rr)
        thread = threading.Thread(target=reader, daemon=True, name='ServerLogReader')
        _FD_READERS.append(thread)
        thread.start()
    _FD_REDIRECTED = True


def close_resident_logger(listener):
    sys.stdout.flush()
    sys.stderr.flush()
    for fd in (1, 2):
        try:
            os.close(fd)
        except OSError:
            pass
    for thread in _FD_READERS:
        thread.join(timeout=1)
    listener.stop()
    for handler in listener.handlers:
        handler.close()

def run_func_in_copied_context(func: Callable[..., Any], *args, **kwargs):
    ctx = contextvars.copy_context()
    return ctx.run(func, *args, **kwargs)

class _IOToLogger:
    encoding = 'utf-8'

    def __init__(self, level: int, logger_name: str):
        from .log_policy import StreamRedactor
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self._buf = ""
        self._discarding = False
        self._lock = threading.RLock()
        self._redactor = StreamRedactor()

    def _line(self):
        from .log_policy import TRUNCATED
        line = self._buf.strip()
        if self._discarding:
            line += TRUNCATED
        self._buf, self._discarding = '', False
        if line:
            self.logger.log(self.level, self._redactor.line(line))

    def write(self, msg: str):
        from .log_policy import MAX_LOG_BYTES
        with self._lock:
            # Long unterminated lines keep only their prefix. Do not emit the
            # discarded suffix as a new line: it could contain an unlabeled
            # continuation of a credential already redacted in the prefix.
            for offset in range(0, len(msg), 4096):
                parts = msg[offset:offset + 4096].replace('\r', '\n').split('\n')
                for index, part in enumerate(parts):
                    if not self._discarding:
                        space = MAX_LOG_BYTES - len(self._buf)
                        self._buf += part[:space]
                        self._discarding = len(part) > space
                    if index < len(parts) - 1:
                        self._line()
        return len(msg)

    def flush(self):
        with self._lock:
            self._line()

    def isatty(self):
        return False

# stdout/stderr → logging
__STDIO_LOCK = threading.RLock()
__STDIO_DEPTH = 0
__ORIG_STDOUT = None
__ORIG_STDERR = None

from contextlib import contextmanager

@contextmanager
def scoped_stdio_to_logging(logger_name: str = "stdio"):
    global __STDIO_DEPTH, __ORIG_STDOUT, __ORIG_STDERR
    with __STDIO_LOCK:
        first = __STDIO_DEPTH == 0
        __STDIO_DEPTH += 1
        if first:
            __ORIG_STDOUT, __ORIG_STDERR = sys.stdout, sys.stderr
            sys.stdout = _IOToLogger(logging.INFO, logger_name)
            sys.stderr = _IOToLogger(logging.ERROR, logger_name)
    try:
        yield
    finally:
        with __STDIO_LOCK:
            __STDIO_DEPTH -= 1
            if __STDIO_DEPTH == 0:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    sys.stdout, sys.stderr = __ORIG_STDOUT, __ORIG_STDERR

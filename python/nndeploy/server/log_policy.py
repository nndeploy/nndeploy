"""Bound and sanitize diagnostic text before IPC, storage and display."""
import copy
import json
import logging
import re

MAX_LOG_BYTES = 32 * 1024
LOG_QUEUE_SIZE = 1024
TRUNCATED = ' … [log truncated]'
REDACTED = '[REDACTED]'

# Context-based redaction, not a claim to recognize arbitrary unlabeled data.
# Drop the remainder of a sensitive assignment line, including quoted/JSON
# values: guessing where an escaped credential ends can expose its suffix.
_LABEL = (r'''(?<![\w-])((?:proxy[-_])?authorization|cookie|set-cookie|password|passwd|pwd|'''
    r'''(?:api|access|secret)[-_]?key|(?:client[-_])?secret|(?:access[-_]|refresh[-_]|session[-_])?token|signature)'''
    r'''(["']?\s*[:=]\s*)''')
_ASSIGNMENT = re.compile(_LABEL + r'[^\r\n]*', re.IGNORECASE)
_QUOTED_ASSIGNMENT = re.compile(_LABEL + r'''(["'])(?:\\.|(?!\3)[\s\S])*(?:\3|\Z)''', re.IGNORECASE)
_PRIVATE_KEY = re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)')
_PRIVATE_START = re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')
_PRIVATE_END = re.compile(r'-----END [A-Z ]*PRIVATE KEY-----')
_QUOTED_START = re.compile(_LABEL + r'''(["'])''', re.IGNORECASE)
_BEARER = re.compile(r'(?i)\bBearer\s+[^\s,;"\'<>]+')
_URL = re.compile(r'''(?i)\b(?:https?|s3|mysql(?:\+\w+)?|postgresql(?:\+\w+)?|redis|rediss)://[^\s<>"']+''')
EVENT_FIELDS = frozenset(('event', 'timestamp', 'pid', 'request_id', 'task_id', 'entry', 'worker_pid',
                          'state', 'method', 'route', 'status', 'duration_seconds'))


def bounded(value, limit=MAX_LOG_BYTES):
    # Slice characters first so encoding a caller's huge message cannot double
    # its entire allocation. Keep the final UTF-8 payload within the byte cap.
    prefix = value[:limit]
    data = prefix.encode('utf-8', errors='replace')
    if len(value) > limit or len(data) > limit:
        allowance = max(0, limit - len(TRUNCATED.encode()))
        return data[:allowance].decode('utf-8', errors='ignore') + TRUNCATED
    return data.decode('utf-8')


def redact(value):
    text = bounded(str(value))
    text = _PRIVATE_KEY.sub(REDACTED, text)
    text = _QUOTED_ASSIGNMENT.sub(lambda m: m.group(1) + m.group(2) + REDACTED, text)
    def url(match):
        value = match.group()
        suffix = ('?' + REDACTED if '?' in value else '') + ('#' + REDACTED if '#' in value else '')
        value = re.split(r'[?#]', value, maxsplit=1)[0]
        scheme, rest = value.split('://', 1)
        authority, separator, tail = rest.partition('/')
        # Userinfo can also precede a query without a path.
        if '@' in authority:
            authority = REDACTED + '@' + authority.rsplit('@', 1)[1]
        return scheme + '://' + authority + separator + tail + suffix
    text = _URL.sub(url, text)
    text = _BEARER.sub('Bearer ' + REDACTED, text)
    return bounded(_ASSIGNMENT.sub(lambda m: m.group(1) + m.group(2) + REDACTED, text))


def event_data(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in EVENT_FIELDS:
        if key not in value:
            continue
        item = value.get(key)
        if item is None or type(item) in (bool, int, float):
            result[key] = item
        elif isinstance(item, str):
            result[key] = bounded(redact(item), 1024)
    return result


class StreamRedactor:
    """Remember sensitive multiline values without retaining their contents."""
    def __init__(self):
        self.private_key = False
        self.quote = None

    @staticmethod
    def _closing_quote(line, offset, quote):
        while offset < len(line):
            if line[offset] == '\\':
                offset += 2
            elif line[offset] == quote:
                return offset + 1
            else:
                offset += 1
        return None

    def line(self, value):
        sensitive = self.private_key or self.quote is not None
        start = _PRIVATE_START.search(value)
        if self.private_key or start:
            self.private_key = not bool(_PRIVATE_END.search(value, start.end() if start else 0))
            # Drop the complete continuation line, including any suffix. Only
            # state crosses line boundaries; never retain key material.
            return REDACTED
        offset = 0
        if self.quote is not None:
            offset = self._closing_quote(value, 0, self.quote)
            if offset is None:
                return REDACTED
            self.quote = None
        while match := _QUOTED_START.search(value, offset):
            offset = self._closing_quote(value, match.end(), match.group(3))
            if offset is None:
                self.quote = match.group(3)
                break
        return REDACTED if sensitive else redact(value)


class DiagnosticFormatter(logging.Formatter):
    def format(self, record):
        # Formatting exceptions can cache exc_text on the original record.
        # Do not leave raw diagnostic text for another Server-owned sink.
        record = copy.copy(record)
        if hasattr(record, 'server_event'):
            data = event_data(record.server_event)
            if getattr(record, 'logs_dropped', 0):
                data['logs_dropped'] = record.logs_dropped
            return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        return redact(super().format(record))


class DiagnosticStreamHandler(logging.StreamHandler):
    """A broken diagnostic sink must not recurse through captured stderr."""
    def handleError(self, record):
        self.dropped_total = getattr(self, 'dropped_total', 0) + 1
        metrics = getattr(self, 'metrics', None)
        if metrics is not None:
            metrics.log_dropped.labels(self.log_sink).inc()


class _QueuedDiagnosticHandler(logging.Handler):
    """Bounded, disposable diagnostics; sink I/O belongs only to its writer."""
    log_sink = 'console'

    def __init__(self):
        import queue
        import threading
        super().__init__()
        self.queue = queue.Queue(LOG_QUEUE_SIZE)
        self.stopping = threading.Event()
        self.dropped_total = 0
        self.thread = threading.Thread(target=self._write, name='ServerLogWriter', daemon=True)
        self.thread.start()

    def emit(self, record):
        if self.stopping.is_set():
            self._drop()
            return
        try:
            self.queue.put_nowait(bounded(self.format(record)) + '\n')
        except Exception:
            # Never recurse into redirected stderr on formatting/sink errors.
            self._drop()

    def _drop(self):
        self.dropped_total += 1
        metrics = getattr(self, 'metrics', None)
        if metrics is not None:
            metrics.log_dropped.labels(self.log_sink).inc()

    def _write(self):
        from queue import Empty
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    value = self.queue.get(timeout=.05)
                except Empty:
                    continue
                try:
                    self._write_value(value)
                except Exception:
                    self._drop()
        finally:
            try:
                self._close_sink()
            except Exception:
                self._drop()

    def close(self):
        self.acquire()
        try:
            self.stopping.set()
        finally:
            self.release()
        # A blocked filesystem/console cannot hold Server shutdown open. Only
        # the writer closes its unbuffered stream; never acquire its I/O lock.
        self.thread.join(timeout=.25)
        if self.thread.is_alive():
            from queue import Empty
            while True:
                try:
                    self.queue.get_nowait()
                    self._drop()
                except Empty:
                    break
        super().close()

    @staticmethod
    def _write_bytes(stream, data):
        while data:
            written = stream.write(data)
            if not written:
                raise OSError('Diagnostic write made no progress')
            data = data[written:]


class QueuedDiagnosticStreamHandler(_QueuedDiagnosticHandler):
    """Queue console and redirected file output without buffered stdio locks."""
    def __init__(self, stream=None):
        import os
        import sys
        stream = sys.stderr if stream is None else stream
        self._owned_stream = False
        try:
            self.stream = os.fdopen(os.dup(stream.fileno()), 'wb', buffering=0)
            self._owned_stream = True
        except (AttributeError, OSError, ValueError):
            self.stream = stream
        super().__init__()

    def _write_value(self, value):
        if self._owned_stream:
            self._write_bytes(self.stream, value.encode('utf-8', errors='replace'))
        else:
            self.stream.write(value)
            self.stream.flush()

    def _close_sink(self):
        if self._owned_stream:
            self.stream.close()


class DiagnosticRotatingFileHandler(_QueuedDiagnosticHandler):
    """Keep open/write/rotation/close off both request and log-listener threads.

    Rotation retains the existing filename.N convention. One writer owns the
    file; multiple processes must use different paths. Unbuffered writes avoid
    interpreter shutdown acquiring a blocked daemon's buffered stream lock.
    """
    log_sink = 'file'

    def __init__(self, filename, maxBytes=0, backupCount=0, encoding='utf-8'):
        import os
        self.baseFilename = os.path.abspath(filename)
        self.maxBytes, self.backupCount = maxBytes, backupCount
        self.encoding, self.stream = encoding, None
        super().__init__()

    def _open(self):
        from pathlib import Path
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return open(self.baseFilename, 'ab', buffering=0)

    def _write_value(self, value):
        data = value.encode(self.encoding, errors='replace')
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes > 0 and self.backupCount > 0:
            self.stream.seek(0, 2)
            size = self.stream.tell()
            if size and size + len(data) >= self.maxBytes:
                self.doRollover()
                self.stream = self._open()
        self._write_bytes(self.stream, data)

    def doRollover(self):
        import os
        self.stream.close()
        self.stream = None
        for index in range(self.backupCount - 1, 0, -1):
            source = self.baseFilename + '.' + str(index)
            if os.path.exists(source):
                os.replace(source, self.baseFilename + '.' + str(index + 1))
        os.replace(self.baseFilename, self.baseFilename + '.1')

    def _close_sink(self):
        if self.stream is not None:
            self.stream.close()


def transport_record(record, formatter):
    """Only bounded, sanitized, picklable fields cross the logging queue."""
    value = formatter.format(record)
    result = logging.LogRecord(bounded(redact(record.name), 128), record.levelno,
                               bounded(redact(record.pathname), 512), record.lineno, value, (), None)
    result.created = record.created
    result.processName = bounded(redact(record.processName), 128)
    result.process = record.process
    result.threadName = bounded(redact(record.threadName), 128)
    for key in ('task_id', 'request_id'):
        item = getattr(record, key, None)
        if isinstance(item, str):
            setattr(result, key, bounded(redact(item), 128))
    if hasattr(record, 'server_event'):
        result.server_event = event_data(record.server_event)
    return result

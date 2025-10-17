# logging_taskid.py
import logging
import contextvars
import sys
import os
import threading
from contextlib import contextmanager
from typing import Callable, Any

# --- ContextVar ---
_TASK_ID = contextvars.ContextVar("task_id", default="0")

_TASK_ID_FALLBACK = "0"

def set_task_id(tid: str):
    return _TASK_ID.set(tid)

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
    def factory(*args, **kwargs):
        record = old(*args, **kwargs)
        try:
            tid = _TASK_ID.get()
        except Exception:
            tid = "0"
        if not tid or tid == "0":
            tid = _TASK_ID_FALLBACK

        record.task_id = tid
        if tid and tid != "0" and not getattr(record, "_taskid_prefixed", False):
            msg = record.getMessage()
            record.msg = f"[task_id={tid}] {msg}"
            record.args = ()
            record._taskid_prefixed = True
        return record
    logging.setLogRecordFactory(factory)

def redirect_python_stdio(logger_name: str = "stdio") -> None:
    logger = logging.getLogger(logger_name)
    class _Std:
        def __init__(self, level): self.level = level
        def write(self, msg):
            msg = msg.rstrip()
            if msg:
                logger.log(self.level, msg)
        def flush(self): pass
    sys.stdout = _Std(logging.INFO)
    sys.stderr = _Std(logging.ERROR)

_FD_REDIRECTED = False
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
            with os.fdopen(rr, "r", buffering=1) as pr:
                for line in pr:
                    line = line.rstrip()
                    if line:
                        logger.log(lvl, line)
        threading.Thread(target=reader, daemon=True).start()
    _FD_REDIRECTED = True

def run_func_in_copied_context(func: Callable[..., Any], *args, **kwargs):
    ctx = contextvars.copy_context()
    return ctx.run(func, *args, **kwargs)

class _IOToLogger:
    def __init__(self, level: int, logger_name: str):
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self._buf = ""

    def write(self, msg: str):
        if not msg:
            return
        self._buf += msg.replace("\r", "\n")
        lines = self._buf.split("\n")
        self._buf = lines[-1]
        for line in lines[:-1]:
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        if self._buf:
            line = self._buf.strip()
            if line:
                self.logger.log(self.level, line)
            self._buf = ""

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
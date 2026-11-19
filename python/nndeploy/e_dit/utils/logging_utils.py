"""Minimal logging helpers for diffusion inference."""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable

from rich.logging import RichHandler


CYAN = "\033[1;36m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0;0m"

_main_process_check: Callable[[], bool] = lambda: True


def set_main_process_check(check: Callable[[], bool]) -> None:
    """Let the runtime supply rank state without importing it from logging."""
    global _main_process_check
    _main_process_check = check


class _MainProcessOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return get_is_main_process()


class _ServerLogHandler(logging.Handler):
    def emit(self, record):
        # Skip the nndeploy parent logger's console handler: stderr is already
        # redirected by Server, so that route duplicates and reclassifies logs.
        logging.getLogger().handle(record)


def init_logger(name: str) -> logging.Logger:
    """Create a simple logger for local runtime work."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Preserve severity and task IDs when nndeploy owns logging. Routing INFO
    # through redirected stderr would otherwise turn every message into ERROR.
    if getattr(logging.getLogRecordFactory(), '_nndeploy_task_context', False):
        handler = _ServerLogHandler()
        handler.addFilter(_MainProcessOnlyFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    # When stdout is a terminal, use RichHandler for pretty output.
    # When piped/redirected (tmux + file-redirect), RichHandler buffers
    # heavily even when stream=sys.stderr is set.  Fall back to a plain
    # StreamHandler with line_buffering so every log record is written
    # immediately.
    is_tty = sys.stdout.isatty()
    target_stream = sys.stdout if is_tty else sys.stderr
    formatter = logging.Formatter(fmt="%(message)s", datefmt="[%m/%d %H:%M:%S]")

    if is_tty:
        handler = RichHandler(
            rich_tracebacks=False,
            markup=False,
            show_path=False,
            omit_repeated_times=False,
            console=None,
        )
        handler.stream = target_stream
        handler.setFormatter(formatter)
        handler.addFilter(_MainProcessOnlyFilter())
        logger.addHandler(handler)

    plain_handler = logging.StreamHandler(stream=target_stream)
    plain_handler.setFormatter(formatter)
    plain_handler.addFilter(_MainProcessOnlyFilter())
    # Server redirects stderr to its task logger, which is not a TextIOWrapper.
    if not is_tty and hasattr(plain_handler.stream, 'reconfigure'):
        plain_handler.stream.reconfigure(line_buffering=True)
    logger.addHandler(plain_handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_log_level() -> int:
    """Return the effective root logging level used by the local runtime."""
    return logging.getLogger().getEffectiveLevel() or logging.INFO


def _sanitize_for_logging(value: Any, key_hint: str | None = None) -> Any:
    """Keep logging output compact for prompt-like fields."""
    if not isinstance(value, str):
        return value
    if key_hint in {"prompt", "negative_prompt"} and len(value) > 200:
        return value[:197] + "..."
    if len(value) > 200:
        suffix = f" ({key_hint})" if key_hint else ""
        return value[:197] + "..." + suffix
    return value


def get_is_main_process() -> bool:
    try:
        return _main_process_check()
    except Exception:
        return True


def globally_suppress_loggers(*logger_names: str) -> None:
    for logger_name in logger_names:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

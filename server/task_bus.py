# task_bus.py
from contextvars import ContextVar
from typing import Callable, Optional, Dict, Any

_emit_cb: ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = ContextVar("emit_cb", default=None)
_pid: ContextVar[Optional[int]] = ContextVar("pid", default=None)

def install_emitter(pid: int, emit: Callable[[Dict[str, Any]], None]) -> None:
    _pid.set(pid); _emit_cb.set(emit)

def clear_emitter() -> None:
    _pid.set(None); _emit_cb.set(None)

def emit_llm_delta(delta_text: str) -> None:
    cb = _emit_cb.get()
    if not cb:
        return
    payload = {
        "event": "stream",
        "pid": _pid.get(),
        "data": {           # 用 data 字段承载业务负载
            "type": "llm_delta",
            "delta": delta_text
        }
    }
    try:
        cb(payload)
    except Exception:
        pass

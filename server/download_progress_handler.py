# download_progress_handler.py
import logging, re, time, asyncio
from typing import Callable, Optional, Dict

ProgressDict = Dict[str, object]

# Downloading [detect/yolo11s.sim.onnx]:   3%|2         | 1.00M/36.2M [00:01<00:46, 790kB/s]
_RE = re.compile(
    r"Downloading\s*\[(?P<name>.+?)\]\s*:\s*"
    r"(?:(?P<pct>\d+(?:\.\d+)?)%)?.*?\|\s*"
    r"(?P<cur>[\d\.]+)(?P<unit>[kKmMgG]?[bB]?)\s*/\s*"
    r"(?P<tot>[\d\.]+)(?P<tunit>[kKmMgG]?[bB]?)"
)

_UNIT = {"":1, "b":1, "kb":1024, "mb":1024**2, "gb":1024**3}

def _to_bytes(v: float, unit: str) -> int:
    u = unit.lower()
    if u and u[-1] != "b":  # 兼容纯 "M/G/K"
        u += "b"
    return int(float(v) * _UNIT.get(u, 1))

class DownloadProgressHandler(logging.Handler):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        emit_cb: Callable[[ProgressDict], None],
        task_id_field: str = "task_id",
        logger_names=("model-download", "modelscope"),
    ):
        super().__init__(level=logging.INFO)
        self.loop = loop
        self.emit_cb = emit_cb
        self.start_ts = time.time()
        self.logger_names = set(logger_names)
        self.task_id_field = task_id_field

    def emit(self, record: logging.LogRecord) -> None:
        if record.name not in self.logger_names:
            return
        msg = record.getMessage()
        m = _RE.search(msg)
        if not m:
            return

        name = m.group("name")
        pct = m.group("pct")
        cur, tot = m.group("cur"), m.group("tot")
        unit, tunit = m.group("unit") or "", m.group("tunit") or ""
        downloaded = _to_bytes(float(cur), unit)
        total = _to_bytes(float(tot), tunit)
        percent = float(pct) if pct else (downloaded / total * 100.0 if total else None)

        payload: ProgressDict = {
            "filename": name,
            "downloaded": downloaded,
            "total": total,
            "percent": percent,
            "elapsed": time.time() - self.start_ts
        }

        try:
            self.loop.call_soon_threadsafe(self.emit_cb, payload)
        except RuntimeError:
            pass

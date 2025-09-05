# queue.py

import copy
import heapq
import time
import threading
import queue as _queue
from typing import Any, Dict, List, Optional

_MAX_HISTORY = 1000

class ExecutionStatus:
    def __init__(self, ok: bool, msg: str = "", label: str | None = None):
        self.str = label or ("success" if ok else "failed")
        self.completed = ok
        self.messages = [msg] if msg else []

class TaskQueue:
    """thread safe queue"""
    def __init__(self, server: "NnDeployServer", job_mp_q: "mp.Queue"):
        self.server = server
        self._mtx = threading.RLock()
        self._not_empty = threading.Condition(self._mtx)
        self._counter = 0
        self._pq: List[Any] = []
        self._running: Dict[int, Any] = {}
        self._hist: Dict[str, Any] = {}
        self._job_q  = job_mp_q
    
    def put(self, payload, prio: int = 0):
        with self._mtx:
            heapq.heappush(self._pq, (prio, time.time(), payload))
            # self.server.queue_updated()
            self._not_empty.notify()
    
    def get(self, timeout: Optional[float] = None):
        with self._not_empty:
            while not self._pq:
                if not self._not_empty.wait(timeout):
                    return None
            prio, ts, payload = heapq.heappop(self._pq)
            idx = self._counter
            self._running[idx] = copy.deepcopy(payload)
            self._counter += 1

            return idx, payload

    def task_done(self, idx: int, status: ExecutionStatus, time_profile_map: Dict):
        with self._mtx:
            task = self._running.pop(idx)
            if len(self._hist) >= _MAX_HISTORY:
                self._hist.pop(next(iter(self._hist)))
            self._hist[task["id"]] = {
                "task": task,
                "status": status.__dict__
            }
            self.server.notify_task_done(task["id"], status, time_profile_map)

    def get_current_queue(self):
        with self._mtx:
            return copy.deepcopy(self._running), copy.deepcopy(self._pq)
    
    def get_history(self, max_items: int | None = None):
        with self._mtx:
            items = list(self._hist.items())[-max_items:] if max_items else self._hist.items()
            return dict(items)

    def get_task_by_id(self, task_id: str) -> Optional[dict]:
        with self._mtx:
            for task in self._running.values():
                if task.get("id") == task_id:
                    return copy.deepcopy(task)
            record = self._hist.get(task_id)
            return copy.deepcopy(record) if record else None

    def _push_hist_cancelled_unlocked(self, idx: int, payload: dict, reason: str):
        self._running.pop(idx, None)
        if len(self._hist) >= _MAX_HISTORY:
            self._hist.pop(next(iter(self._hist)))
        tid = payload.get("id")
        status = ExecutionStatus(ok=False, msg=reason, label="cancelled")
        self._hist[tid] = {
            "task": payload,
            "status": status.__dict__,
        }

    def clear_pending(self) -> int:
        with self._mtx:
            n = len(self._pq)
            self._pq.clear()
            return n

    def drain_job_q(self) -> int:
        drained = 0
        while True:
            try:
                idx, payload = self._job_q.get_nowait()
            except _queue.Empty:
                break
            except Exception:
                break
            else:
                with self._mtx:
                    drained += 1
                    self._push_hist_cancelled_unlocked(idx, payload, reason="flushed from job_q")
        return drained

    def flush(self) -> dict:
        cleared_pending = self.clear_pending()
        drained_jobq = self.drain_job_q()

        import time as _t
        for _ in range(2):
            _t.sleep(0.02)
            drained_jobq += self.drain_job_q()

        return {"cleared_pending": cleared_pending, "drained_job_q": drained_jobq}
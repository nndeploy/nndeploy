# queue.py

import copy
import heapq
import time
import threading
import queue as _queue
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

_MAX_HISTORY = 1000

class ExecutionStatus:
    def __init__(self, ok: bool, msg: str = "", label: str | None = None):
        self.str = label or ("success" if ok else "failed")
        self.completed = ok
        self.messages = [msg] if msg else []

class TaskState(Enum):
    PENDING = auto()         # in pq_
    DISPATCHED = auto()      # in job_mp_queue, not running
    RUNNING = auto()         # running
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()

@dataclass
class TaskRecord:
    idx: int
    payload: Dict[str, Any]
    state: TaskState = TaskState.PENDING
    ts_submit: float = field(default_factory=time.time)
    ts_dispatch: Optional[float] = None
    ts_start: Optional[float] = None
    ts_finish: Optional[float] = None
    worker_pid: Optional[int] = None

class TaskQueue:
    """thread safe queue"""
    def __init__(self, server: "NnDeployServer", job_mp_q: "mp.Queue"):
        self.server = server
        self._mtx = threading.RLock()
        self._not_empty = threading.Condition(self._mtx)
        self._counter = 0
        self._pq: List[Any] = []
        self._active: Dict[int, TaskRecord] = {}
        self._hist: Dict[str, Any] = {}
        self._job_q  = job_mp_q
    
    def put(self, payload, prio: int = 0):
        with self._mtx:
            heapq.heappush(self._pq, (prio, time.time(), payload))
            self._not_empty.notify()
    
    def get(self, timeout: Optional[float] = None):
        with self._not_empty:
            while not self._pq:
                if not self._not_empty.wait(timeout):
                    return None
            prio, ts, payload = heapq.heappop(self._pq)
            idx = self._counter
            rec = TaskRecord(idx=idx, payload=copy.deepcopy(payload), state=TaskState.PENDING, ts_submit=ts)
            self._active[idx] = rec
            self._counter += 1

            return idx, payload

    def mark_dispatched(self, idx: int):
        with self._mtx:
            rec = self._active.get(idx)
            if not rec:
                return
            rec.state = TaskState.DISPATCHED
            rec.ts_dispatch = time.time()

    def mark_started(self, task_id: str, worker_pid: Optional[int] = None):
        with self._mtx:
            target: Optional[TaskRecord] = None
            for rec in self._active.values():
                if rec.payload.get("id") == task_id:
                    target = rec
                    break
            if not target:
                return
            target.state = TaskState.RUNNING
            target.ts_start = time.time()
            if worker_pid:
                target.worker_pid = worker_pid

    def task_done(self, idx: int, status: ExecutionStatus, results: Dict, time_profile_map: Dict):
        with self._mtx:
            rec = self._active.pop(idx, None)
            if rec is None:
                return
            rec.ts_finish = time.time()
            final_state = TaskState.SUCCEEDED if status.completed else (
                TaskState.CANCELLED if status.str == "cancelled" else TaskState.FAILED
            )
            if len(self._hist) >= _MAX_HISTORY:
                self._hist.pop(next(iter(self._hist)))
            task_id = rec.payload.get("id")
            self._hist[task_id] = {
                "task": rec.payload,
                "status": status.__dict__,
                "state": final_state.name,
                "ts_submit": rec.ts_submit,
                "ts_dispatch": rec.ts_dispatch,
                "ts_start": rec.ts_start,
                "ts_finish": rec.ts_finish,
                "worker_pid": rec.worker_pid,
                "time_profile": time_profile_map,
            }
            self.server.notify_task_done(task_id, status, results, time_profile_map)

    def get_current_queue(self):
        with self._mtx:
            running = []
            dispatched = []
            pending = []
            for rec in self._active.values():
                data = {
                    "idx": rec.idx,
                    "task": copy.deepcopy(rec.payload),
                    "state": rec.state.name,
                    "ts_submit": rec.ts_submit,
                    "ts_dispatch": rec.ts_dispatch,
                    "ts_start": rec.ts_start,
                    "worker_pid": rec.worker_pid,
                }
                if rec.state == TaskState.RUNNING:
                    running.append(data)
                elif rec.state == TaskState.DISPATCHED:
                    dispatched.append(data)
                else:
                    pending.append(data)
            pq_snapshot = [ (p, ts, copy.deepcopy(pl)) for (p, ts, pl) in self._pq ]
            return {
                "RUNNING": running,
                "DISPATCHED": dispatched,
                "PENDING": pq_snapshot
            }

    def get_history(self, max_items: int | None = None):
        with self._mtx:
            items = list(self._hist.items())[-max_items:] if max_items else self._hist.items()
            return dict(items)

    def get_task_by_id(self, task_id: str) -> Optional[dict]:
        with self._mtx:
            for rec in self._active.values():
                if rec.payload.get("id") == task_id:
                    return {
                        "task": copy.deepcopy(rec.payload),
                        "state": rec.state.name,
                        "ts_submit": rec.ts_submit,
                        "ts_dispatch": rec.ts_dispatch,
                        "ts_start": rec.ts_start,
                        "worker_pid": rec.worker_pid,
                    }
            record = self._hist.get(task_id)
            return copy.deepcopy(record) if record else None

    def _push_hist_cancelled_unlocked(self, idx: int, payload: dict, reason: str):
        rec = self._active.pop(idx, None)
        if len(self._hist) >= _MAX_HISTORY:
            self._hist.pop(next(iter(self._hist)))
        tid = payload.get("id")
        status = ExecutionStatus(ok=False, msg=reason, label="cancelled")
        self._hist[tid] = {
            "task": payload,
            "status": status.__dict__,
            "state": TaskState.CANCELLED.name,
            "ts_submit": rec.ts_submit if rec else None,
            "ts_dispatch": rec.ts_dispatch if rec else None,
            "ts_start": rec.ts_start if rec else None,
            "ts_finish": time.time(),
            "worker_pid": rec.worker_pid if rec else None,
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
# queue.py

import copy
import heapq
import time
import threading
from typing import Any, Dict, List, Optional

_MAX_HISTORY = 1000

class ExecutionStatus:
    """keep same with comfyui"""
    def __init__(self, ok: bool, msg: str = ""):
        self.str = "success" if ok else "failed"
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
# queue.py

import copy
import heapq
import time
import threading
import queue as _queue
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp
    from .server import NnDeployServer

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
    cancel_requested: bool = False

class TaskQueue:
    """thread safe queue"""
    def __init__(self, server: "NnDeployServer" = None, job_mp_q: "mp.Queue" = None,
                 *, capacity: int | None = None, on_done=None, history_limit=_MAX_HISTORY, history_store=None):
        if type(history_limit) is not int or history_limit < 0:
            raise ValueError('history_limit must be a nonnegative integer')
        self.server = server
        self.history_limit = history_limit
        self.history_store = history_store
        self.capacity = capacity
        self.on_done = on_done
        self._mtx = threading.RLock()
        self._not_empty = threading.Condition(self._mtx)
        self._counter = 0
        self._enqueue_counter = 0
        self._pq: List[Any] = []
        self._active: Dict[int, TaskRecord] = {}
        self._hist: Dict[str, Any] = {}
        self._pending_cancellations = set()
        self._pending_completions = {}
        self._job_q  = job_mp_q
        self._editor_paused = False
        self._last_recovery = 0.0
        if self.history_store is not None:
            try:
                self.recover_editor()
                while self.history_store.cursor >= 0:
                    self.recover_editor()
            except BaseException:
                self.history_store.close()
                raise

    def recover_editor(self):
        if self.history_store is None:
            return
        with self._mtx:
            rows = self.history_store.recover()
            pending = {payload['id'] for _, _, _, payload in self._pq}
            active = {rec.payload['id']: rec for rec in self._active.values()}
            for row in rows:
                identity = row['task_id']
                if identity not in pending and identity not in active and row['state'] == 'pending':
                    import json
                    heapq.heappush(self._pq, (row['priority'], row['sequence'], row['created_at'], json.loads(row['payload'])))
                    pending.add(identity)
                    self._not_empty.notify_all()
                if row['cancel_requested']:
                    if identity in pending:
                        self._pending_cancellations.add(identity)
                    elif identity in active:
                        active[identity].cancel_requested = True
            self._last_recovery = time.monotonic()

    def pause_editor(self):
        with self._mtx:
            self._editor_paused = True
            self._not_empty.notify_all()

    def resume_editor(self, job_q):
        with self._mtx:
            self._job_q = job_q
            self._editor_paused = False
            self._not_empty.notify_all()

    def fail_editor_active(self):
        """Finalize uncertain deliveries only after their Worker has exited.

        The parent heap is still authoritative and is left intact. Anything
        handed to the lost IPC process must not be automatically executed again.
        """
        with self._mtx:
            records = [(rec.idx, rec.cancel_requested) for rec in self._active.values()]
        for idx, cancelled in records:
            self.task_done(idx, ExecutionStatus(False, 'Editor Worker exited before confirming a result',
                                               'cancelled' if cancelled else 'failed'), {}, {})
    
    def put(self, payload, prio: int = 0, *, idempotent=False, prepare=None):
        with self._mtx:
            rejection = None
            if self._editor_paused:
                from .errors import ServerError
                rejection = ServerError(503, 'editor_recovering', 'Editor Worker is recovering; retry later',
                                        retryable=True, accepted=False)
            elif self.capacity is not None and len(self._pq) + len(self._active) >= self.capacity:
                rejection = _queue.Full("Task queue capacity reached")
            if rejection is not None and not (idempotent and self.history_store is not None):
                raise rejection
            payload = copy.deepcopy(payload)
            submitted, sequence = time.time(), self._enqueue_counter
            if self.history_store is not None:
                options = {'prepare': prepare} if prepare is not None else {}
                if idempotent:
                    # Duplicate lookup and new admission are serialized together
                    # in the workspace DB. A replay bypasses new-task capacity.
                    record = self.history_store.admit(payload, prio, submitted,
                                                     idempotent=True, rejection=rejection, **options)
                    if (record['state'] != 'pending' or record['token'] != self.history_store.token
                            or self._editor_paused):
                        return
                else:
                    record = self.history_store.admit(payload, prio, submitted, **options)
                if any(item[3]['id'] == payload['id'] for item in self._pq) or any(
                        rec.payload['id'] == payload['id'] for rec in self._active.values()):
                    return
                submitted, sequence = record['created_at'], record['sequence']
                import json
                payload = json.loads(record['payload'])
            elif prepare is not None:
                payload = prepare(payload, None)
            heapq.heappush(self._pq, (prio, sequence, submitted, payload))
            self._enqueue_counter += 1
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None):
        with self._not_empty:
            while True:
                eligible = None
                if self._pq and not self._editor_paused:
                    eligible = min((item for item in self._pq if item[3].get('id') not in self._pending_cancellations),
                                   default=None, key=lambda item: item[:3]) if self._pending_cancellations else self._pq[0]
                if eligible is not None:
                    break
                if not self._not_empty.wait(timeout):
                    return None
            if self.history_store is not None:
                # Returning a payload permits legacy callers to publish it
                # directly. Record possible delivery before that can happen.
                self.history_store.dispatch(eligible[3]['id'])
            if eligible is self._pq[0]:
                heapq.heappop(self._pq)
            else:
                self._pq.remove(eligible)
                heapq.heapify(self._pq)
            prio, sequence, ts, payload = eligible
            idx = self._counter
            rec = TaskRecord(idx=idx, payload=copy.deepcopy(payload), state=TaskState.PENDING, ts_submit=ts)
            self._active[idx] = rec
            self._counter += 1
            if self.history_store is not None:
                payload = dict(payload, _editor_guard=str(self.history_store.guard.path),
                               _editor_scope=self.history_store.scope, _editor_token=self.history_store.token,
                               _editor_submitted=ts)
            return idx, payload

    def mark_dispatched(self, idx: int):
        with self._mtx:
            rec = self._active.get(idx)
            if not rec:
                return
            # A fast worker may report started before the scheduler returns
            # from job_q.put(). Do not regress RUNNING back to DISPATCHED.
            if rec.state == TaskState.PENDING:
                rec.state = TaskState.DISPATCHED
            rec.ts_dispatch = time.time()

    def dispatch(self, idx, payload, job_q):
        """Serialize editor cancellation with publishing to its Worker."""
        with self._mtx:
            rec = self._active.get(idx)
            if rec is None or rec.cancel_requested or self._editor_paused:
                return
            if self.history_store is not None:
                self.history_store.dispatch(rec.payload['id'])
            try:
                job_q.put((idx, payload), block=False)
            except _queue.Full:
                if self.history_store is not None:
                    self.history_store.undispatched(rec.payload['id'])
                raise
            self.mark_dispatched(idx)

    def finish_result(self, result):
        self.task_done(*result)
        if hasattr(self._job_q, 'acknowledge'):
            self._job_q.acknowledge(result[0])

    def collect_results(self):
        if self.history_store is not None and time.monotonic() - self._last_recovery >= 1:
            self.recover_editor()
        with self._mtx:
            completions = [(idx, *value[:3]) for idx, value in list(self._pending_completions.items())[:16]]
            cancellations = list(self._pending_cancellations)[:16]
            cancellations.extend([rec.payload['id'] for rec in self._active.values()
                                  if rec.cancel_requested and rec.idx not in self._pending_completions][:16])
        for result in completions:
            self.finish_result(result)
        finished_cancellations = 0
        for task_id in cancellations:
            if self.server is not None and hasattr(self.server, '_cancel_task'):
                record = self.server._cancel_task(task_id)
            else:
                record = self.cancel_editor_task(task_id)
            finished_cancellations += bool(record and record['state'] in ('SUCCEEDED', 'FAILED', 'CANCELLED'))
        if hasattr(self._job_q, 'outcomes'):
            outcomes = self._job_q.outcomes()
            for result in outcomes:
                self.finish_result(result)
            return len(outcomes) + len(completions) + finished_cancellations
        return len(completions) + finished_cancellations

    def cancel_editor_task(self, task_id, *, pending_only=False):
        finish = None
        pending_cancelled = False
        with self._mtx:
            if self.cancel_pending(task_id):
                pending_cancelled = True
            else:
                rec = next((rec for rec in self._active.values() if rec.payload.get('id') == task_id), None)
                if rec is not None:
                    if rec.idx in self._pending_completions:
                        finish = (rec.idx, *self._pending_completions[rec.idx][:3])
                    elif self._editor_paused:
                        # IPC ownership is unknown until the old Worker exits.
                        if not pending_only:
                            rec.cancel_requested = True
                            if self.history_store is not None:
                                self.history_store.cancel(task_id)
                    elif hasattr(self._job_q, 'cancel'):
                        previously_requested = rec.cancel_requested
                        if not pending_only and self.history_store is not None:
                            rec.cancel_requested = True
                            self.history_store.cancel(task_id)
                        state = self._job_q.cancel(rec.idx, only_pending=pending_only)
                        if state not in ('finished', 'running'):
                            rec.cancel_requested = True
                            if self.history_store is not None:
                                self.history_store.cancel(task_id)
                        if state in ('absent', 'cancelled'):
                            finish = (rec.idx, ExecutionStatus(False, 'Cancelled before execution', 'cancelled'), {}, {})
                        elif state == 'finished':
                            rec.cancel_requested = previously_requested
                            finish = self._job_q.outcomes(idx=rec.idx)[0]
                    else:
                        if not pending_only or rec.state == TaskState.PENDING:
                            rec.cancel_requested = True
                            if self.history_store is not None:
                                self.history_store.cancel(task_id)
                        if rec.state == TaskState.PENDING:
                            finish = (rec.idx, ExecutionStatus(False, 'Cancelled before dispatch', 'cancelled'), {}, {})
                elif self.history_store is not None and not pending_only:
                    self.history_store.cancel(task_id)
        if finish is not None:
            self.finish_result(finish)
        if pending_cancelled:
            self._notify_done(task_id, ExecutionStatus(False, 'Cancelled before execution', 'cancelled'), {}, {})
        record = self.get_task_by_id(task_id)
        if record is None:
            return None
        return dict(task_id=task_id, state=record['state'], cancel_requested=record.get('cancel_requested', False))

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
            if self.history_store is not None:
                self.history_store.started(task_id, target.ts_start, target.worker_pid)

    def task_done(self, idx: int, status: ExecutionStatus, results: Dict, time_profile_map: Dict):
        with self._mtx:
            rec = self._active.get(idx)
            if rec is None:
                return
            task_id = rec.payload.get("id")
            pending = self._pending_completions.get(idx)
            if pending is None:
                rec.ts_finish = time.time()
                final_state = TaskState.SUCCEEDED if status.completed else (
                    TaskState.CANCELLED if status.str == "cancelled" else TaskState.FAILED)
                record = {
                    "task": copy.deepcopy(rec.payload),
                    "status": copy.deepcopy(status.__dict__),
                    "state": final_state.name,
                    "ts_submit": rec.ts_submit,
                    "ts_dispatch": rec.ts_dispatch,
                    "ts_start": rec.ts_start,
                    "ts_finish": rec.ts_finish,
                    "worker_pid": rec.worker_pid,
                    "cancel_requested": rec.cancel_requested,
                    "time_profile": copy.deepcopy(time_profile_map),
                    "results": copy.deepcopy(results),
                } if self.history_limit else {}
                if self.history_limit:
                    from .editor_outputs import preview
                    record['outputs'], record['outputs_truncated'] = preview(record['results'])
                    status = copy.deepcopy(status)
                    results, time_profile_map = record['results'], record['time_profile']
                self._pending_completions[idx] = (status, results, time_profile_map, record)
            else:
                # A late cancellation or queue failure must not replace the
                # known computation result while its database receipt is lost.
                status, results, time_profile_map, record = pending
            if self.history_limit:
                self._save_history_unlocked(task_id, record)
            self._active.pop(idx)
            self._pending_completions.pop(idx)
        # Notifications may perform I/O; never keep the queue lock while calling them.
        self._notify_done(task_id, status, results, time_profile_map)

    def _notify_done(self, task_id, status, results, time_profile_map):
        if self.on_done is not None:
            self.on_done(task_id, status, results, time_profile_map)
        elif self.server is not None:
            try:
                self.server.notify_task_done(task_id, status, results, time_profile_map)
            except Exception:
                logging.exception('Editor task notification failed after recording its result')

    def cancel_pending(self, task_id: str) -> bool:
        with self._mtx:
            for index, (_, _, submitted, payload) in enumerate(self._pq):
                if payload.get("id") == task_id:
                    self._pending_cancellations.add(task_id)
                    if self.history_store is not None:
                        self.history_store.cancel(task_id)
                    self._push_hist_cancelled_unlocked(-1, payload, "cancelled before execution", submitted=submitted)
                    self._pq.pop(index)
                    heapq.heapify(self._pq)
                    self._pending_cancellations.discard(task_id)
                    self._not_empty.notify_all()
                    return True
        return False

    def counts(self):
        """Bounded monitoring snapshot without copying request payloads."""
        with self._mtx:
            counts = dict(pending=len(self._pq), dispatched=0, running=0)
            for record in self._active.values():
                counts[record.state.name.lower()] += 1
            return counts

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
                    "cancel_requested": rec.cancel_requested,
                }
                if rec.state == TaskState.RUNNING:
                    running.append(data)
                elif rec.state == TaskState.DISPATCHED:
                    dispatched.append(data)
                else:
                    pending.append(data)
            pq_snapshot = [
                {"task": copy.deepcopy(pl), "state": TaskState.PENDING.name,
                 "ts_submit": ts, "priority": p,
                 "cancel_requested": pl.get('id') in self._pending_cancellations}
                for p, sequence, ts, pl in sorted(self._pq)
            ]
            if self.history_store is not None:
                visible = {rec.payload['id'] for rec in self._active.values()} | {item[3]['id'] for item in self._pq}
                for record in self.history_store.unresolved():
                    if record['task']['id'] not in visible:
                        {'RUNNING': running, 'DISPATCHED': dispatched, 'PENDING': pending}[record['state']].append(record)
            return {
                "RUNNING": running,
                "DISPATCHED": dispatched,
                "PENDING": pending + pq_snapshot
            }

    def cleanup_history(self, retention_seconds, limit=100):
        with self._mtx:
            if self.history_store is None:
                return 0
            removed = self.history_store.cleanup(retention_seconds, limit)
            # Other Server instances can expire this workspace too. Drop
            # native cache entries whose persistent history was removed there.
            retained = self.history_store.retained(list(self._hist))
            for identity in list(self._hist):
                if identity not in retained:
                    self._hist.pop(identity, None)
            return len(removed)

    def get_history(self, max_items: int | None = None, *, before=None, include_results=True):
        with self._mtx:
            if self.history_store is not None and self.history_limit:
                records = self.history_store.latest(min(max_items or self.history_limit, self.history_limit, 1000), before=before)
                if include_results:
                    for identity, record in records.items():
                        if identity in self._hist and 'results' in self._hist[identity]:
                            record['results'] = copy.deepcopy(self._hist[identity]['results'])
                return records
            items = sorted(self._hist.items(), key=lambda item: (item[1].get('ts_finish') or 0, item[0]))
            if before is not None:
                index = next((i for i, (identity, _) in enumerate(items) if identity == before), None)
                if index is None:
                    from .errors import ServerError
                    raise ServerError(404, 'history_cursor_not_found', 'History page is no longer available; refresh the latest tasks')
                items = items[:index]
            if max_items:
                items = items[-max_items:]
            return copy.deepcopy({identity: record if include_results else {
                key: value for key, value in record.items() if key != 'results'} for identity, record in items})

    def get_task_by_id(self, task_id: str, *, include_results=True) -> Optional[dict]:
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
                        "cancel_requested": rec.cancel_requested,
                    }
            for priority, sequence, ts, payload in self._pq:
                if payload.get("id") == task_id:
                    return {"task": copy.deepcopy(payload), "state": TaskState.PENDING.name,
                            "ts_submit": ts, "ts_dispatch": None, "ts_start": None,
                            "worker_pid": None, "cancel_requested": task_id in self._pending_cancellations}
            record = self._hist.get(task_id)
            if record is not None:
                if self.history_store is not None:
                    saved = self.history_store.get(task_id)
                    if saved is None or saved.get('outputs_expired'):
                        self._hist.pop(task_id, None)
                        return saved
                return copy.deepcopy(record if include_results else {
                    key: value for key, value in record.items() if key != 'results'
                })
        if self.history_store is not None:
            return self.history_store.get(task_id) or self.history_store.active(task_id)
        return None

    def owns_active(self, task_id):
        with self._mtx:
            return any(rec.payload.get('id') == task_id for rec in self._active.values())

    def _save_history_unlocked(self, task_id, record):
        if self.history_store is not None:
            saved = self.history_store.save(task_id, record)
            record = dict(saved, **({'results': record['results']} if 'results' in record and not saved.get('outputs_expired') else {}))
        if len(self._hist) >= self.history_limit and task_id not in self._hist:
            self._hist.pop(next(iter(self._hist)))
        self._hist[task_id] = record

    def _push_hist_cancelled_unlocked(self, idx: int, payload: dict, reason: str, *, submitted=None):
        rec = self._active.get(idx)
        if not self.history_limit:
            self._active.pop(idx, None)
            return
        tid = payload.get("id")
        status = ExecutionStatus(ok=False, msg=reason, label="cancelled")
        self._save_history_unlocked(tid, {
            "task": payload,
            "status": status.__dict__,
            "state": TaskState.CANCELLED.name,
            "ts_submit": rec.ts_submit if rec else submitted,
            "ts_dispatch": rec.ts_dispatch if rec else None,
            "ts_start": rec.ts_start if rec else None,
            "ts_finish": time.time(),
            "worker_pid": rec.worker_pid if rec else None,
            "cancel_requested": True,
        })
        self._active.pop(idx, None)

    def clear_pending(self) -> int:
        with self._mtx:
            n = len(self._pq)
            for _, _, _, payload in list(self._pq):
                self.cancel_pending(payload['id'])
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
                    record = self._active.get(idx)
                    if record is not None:
                        record.cancel_requested = True
                self.task_done(idx, ExecutionStatus(False, 'flushed from job_q', 'cancelled'), {}, {})
        return drained

    def flush(self) -> dict:
        if hasattr(self._job_q, 'cancel'):
            with self._mtx:
                pending = [(payload['id'], 'cleared_pending') for _, _, _, payload in self._pq]
                active = [(rec.payload['id'], 'cleared_pending' if rec.state == TaskState.PENDING else 'drained_job_q')
                          for rec in self._active.values()]
            counts = dict(cleared_pending=0, drained_job_q=0)
            for identity, category in pending + active:
                record = self.cancel_editor_task(identity, pending_only=True)
                if record is not None and record['state'] == 'CANCELLED':
                    counts[category] += 1
            return counts
        cleared_pending = self.clear_pending()
        drained_jobq = self.drain_job_q()

        import time as _t
        for _ in range(2):
            _t.sleep(0.02)
            drained_jobq += self.drain_job_q()

        return {"cleared_pending": cleared_pending, "drained_job_q": drained_jobq}


class RedisTaskQueue:
    """DB-backed admission and ownership with one Redis Stream per entry/config.

    Redis messages contain task IDs only. Expired leases are reported for
    reconciliation; they never imply that native computation has stopped.
    """
    _PUBLISH = '''
        local existing = redis.call('HGET', KEYS[2], ARGV[1])
        if existing then
            if #redis.call('XRANGE', KEYS[1], existing, existing, 'COUNT', 1) > 0 then return existing end
            redis.call('HDEL', KEYS[2], ARGV[1])
        end
        if redis.call('HLEN', KEYS[2]) >= tonumber(ARGV[2]) then return false end
        local id = redis.call('XADD', KEYS[1], '*', 'task_id', ARGV[1])
        redis.call('HSET', KEYS[2], ARGV[1], id)
        return id
    '''
    _ACK = '''
        redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
        redis.call('XDEL', KEYS[1], ARGV[2])
        if redis.call('HGET', KEYS[2], ARGV[3]) == ARGV[2] then
            redis.call('HDEL', KEYS[2], ARGV[3])
        end
        return 1
    '''

    def __init__(self, db, config, name, fingerprint, capacity, *, queue_key=None):
        import hashlib
        from .db import encode
        from .redis import RedisConfig
        self.db = db
        self.config = config if isinstance(config, RedisConfig) else RedisConfig.model_validate(config)
        self.name, self.capacity = name, capacity
        if queue_key is not None and (len(queue_key) != 64 or any(c not in '0123456789abcdef' for c in queue_key)):
            raise ValueError('Invalid persisted shared queue key')
        self.key = queue_key or hashlib.sha256(encode([self.config.namespace, name, fingerprint]).encode()).hexdigest()
        self.group_key = hashlib.sha256(encode([self.config.namespace, 'group']).encode()).hexdigest()
        prefix = f'{self.config.namespace}:{{{self.key}}}'
        self.stream, self.published = prefix + ':tasks', prefix + ':published'
        self.group = 'workers'
        self.client = self.config.connect()
        self._reclaim_cursor = '0-0'
        self._terminal_cursor = '-'
        self._database_identity = None

    async def start(self):
        import asyncio
        from .errors import ServerError
        identity = await asyncio.to_thread(self.db.database_identity)
        self._database_identity = identity
        prefix = f'{self.config.namespace}:{{{self.key}}}:db:{identity}'
        self.stream, self.published = prefix + ':tasks', prefix + ':published'
        await self._check_database()
        await asyncio.to_thread(self.db.register_queue, self.group_key, self.config.max_inflight_tasks)
        await asyncio.to_thread(self.db.register_queue, self.key, self.capacity)
        await self._ensure_stream()

    async def _check_database(self):
        from .errors import ServerError
        if self._database_identity is None:
            raise RuntimeError('Start the shared queue before using it')
        marker = f'{self.config.namespace}:database'
        await self.client.set(marker, self._database_identity, nx=True)
        if await self.client.get(marker) != self._database_identity:
            raise ServerError(409, 'queue_database_mismatch', 'Redis namespace is attached to a different database')

    async def _ensure_stream(self):
        from redis.exceptions import ResponseError
        try:
            await self.client.xgroup_create(self.stream, self.group, id='0', mkstream=True)
        except ResponseError as exc:
            if 'BUSYGROUP' not in str(exc):
                raise

    def _idempotency_key(self, value, owner):
        import hashlib
        from .db import encode
        if value is None:
            return None
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ValueError('Idempotency key must contain 1 to 128 characters')
        return hashlib.sha256(encode([self.config.namespace, self.name, owner, value]).encode()).hexdigest()

    async def put(self, inputs, parameters=None, *, queue_timeout=30, idempotency_key=None, owner=None,
                  retention_seconds=None, request_id=None, max_tasks_per_subject=None, input_state=None, callback_url=None, callback_scope=None, service=None):
        import asyncio
        import uuid
        from redis.exceptions import RedisError
        from sqlalchemy.exc import SQLAlchemyError
        parameters = {} if parameters is None else parameters
        key = self._idempotency_key(idempotency_key, owner)
        if key is not None:
            record = await asyncio.to_thread(self.db.find_task, self.name, key, inputs, parameters, owner, callback_url)
            if record:
                return {'task_id': record['task_id'], 'state': record['state'], 'dispatch_pending': record['state'] == 'pending'}
        identity = await asyncio.to_thread(self.db.create_shared_task, 'task_' + uuid.uuid4().hex,
            self.name, self.key, self.group_key, inputs, parameters, queue_timeout, key, owner,
            retention_seconds=retention_seconds, request_id=request_id, max_tasks_per_subject=max_tasks_per_subject,
            input_state=input_state, callback_url=callback_url, callback_scope=callback_scope, service=service)
        pending = True
        try:
            pending = not await self._publish(identity)
        except (RedisError, SQLAlchemyError):
            # The DB outbox is durable. Callers must report accepted/pending,
            # rather than claiming the task was delivered or dropping its ID.
            pass
        return {'task_id': identity, 'state': 'pending', 'dispatch_pending': pending}

    async def _publish(self, identity):
        import asyncio
        await self._check_database()
        message = await self.client.eval(self._PUBLISH, 2, self.stream, self.published, identity, self.capacity)
        if message:
            await asyncio.to_thread(self.db.mark_published, identity)
        return bool(message)

    async def _ack(self, message, identity):
        await self.client.eval(self._ACK, 2, self.stream, self.published, self.group, message, identity)

    async def _claim(self, message, identity, worker_token, holder=None):
        import asyncio
        from .db import TERMINAL
        record = await asyncio.to_thread(self.db.get_task, identity)
        if record is None or record['queue_key'] != self.key or record['state'] in TERMINAL:
            await self._ack(message, identity)
            return None
        now = await asyncio.to_thread(self.db.clock)
        if record['state'] == 'pending' and (record['cancel_requested'] or record['expires_at'] <= now):
            state = 'cancelled' if record['cancel_requested'] else 'expired'
            try:
                await asyncio.to_thread(self.db.finish_task, identity, state, now,
                    error={'code': 'task_cancelled' if state == 'cancelled' else 'queue_timeout', 'message': state}, attempt=record['attempt'])
            except RuntimeError:
                pass  # A concurrent conditional claim/finalization won.
            await self._ack(message, identity)
            return None
        claimed = await asyncio.to_thread(self.db.claim_task, identity, self.key, worker_token, self.config.lease_seconds, holder=holder)
        if claimed is None:
            # A running duplicate is not a new execution. Its DB ownership and
            # capacity remain intact even if Redis transfers the pending message.
            await self._ack(message, identity)
            return None
        claimed['_message_id'] = message
        return claimed

    async def get(self, worker_token, *, holder=None):
        from redis.exceptions import ResponseError
        await self._check_database()
        try:
            reclaimed = await self.client.xautoclaim(self.stream, self.group, worker_token,
                min_idle_time=int(self.config.lease_seconds * 1000), start_id=self._reclaim_cursor, count=1)
            self._reclaim_cursor = reclaimed[0]
            messages = reclaimed[1]
            if not messages:
                response = await self.client.xreadgroup(self.group, worker_token, {self.stream: '>'}, count=1, block=100)
                messages = response[0][1] if response else []
        except ResponseError as exc:
            if 'NOGROUP' not in str(exc):
                raise
            await self.start()
            await self.reconcile()
            return None
        if not messages:
            return None
        message, values = messages[0]
        identity = values.get('task_id')
        if not identity:
            await self._ack(message, '')
            return None
        return await self._claim(message, identity, worker_token, holder)

    async def heartbeat(self, record):
        import asyncio
        alive = await asyncio.to_thread(self.db.heartbeat_task, record['task_id'], record['worker_token'],
            record['attempt'], self.config.lease_seconds)
        if alive:
            await self.client.set(f'{self.config.namespace}:worker:{record["worker_token"]}', self.key,
                                  px=int(self.config.lease_seconds * 1000))
        return alive

    async def finish(self, record, *, outputs=None, error=None, state=None):
        import asyncio
        state = state or ('failed' if error is not None else 'succeeded')
        try:
            now = await asyncio.to_thread(self.db.clock)
            await asyncio.to_thread(self.db.finish_task, record['task_id'], state, now, outputs, error,
                                   record['attempt'], record['worker_token'])
        except RuntimeError:
            existing = await asyncio.to_thread(self.db.get_task, record['task_id'])
            if not (existing and existing['worker_token'] == record['worker_token'] and existing['attempt'] == record['attempt']
                    and existing['state'] == state and existing['outputs'] == outputs and existing['error'] == error):
                raise
        await self._ack(record['_message_id'], record['task_id'])

    async def reconcile(self, limit=100):
        import asyncio
        if not 1 <= limit <= 1000:
            raise ValueError('Reconciliation limit must be between 1 and 1000')
        pending = await asyncio.to_thread(self.db.pending_tasks, self.key, limit)
        published = 0
        for identity in pending:
            published += await self._publish(identity)
        # A recovered dead worker can commit a terminal state without owning
        # the original Redis delivery. Repair that ACK independently of lease
        # reclaim, including when every remaining worker has been stopped.
        from .db import TERMINAL
        messages = await self.client.xrange(self.stream, min=self._terminal_cursor, count=limit)
        self._terminal_cursor = '(' + messages[-1][0] if messages else '-'
        for message, values in messages:
            identity = values.get('task_id', '')
            record = await asyncio.to_thread(self.db.get_task, identity) if identity else None
            if record is None or record['queue_key'] != self.key or record['state'] in TERMINAL:
                await self._ack(message, identity)
        return {'published': published, 'stalled': await asyncio.to_thread(self.db.stalled_tasks, self.key, limit)}

    async def close(self):
        await self.client.aclose()

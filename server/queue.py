# job_queue.py
import asyncio, heapq, time
from collections import deque
from typing import Callable, Awaitable, Dict, List

from job import Job, JobStatus

class JobQueue:
    def __init__(self,
                 run_graph_func: Callable[[Dict[str, Any], Callable[[str, float], None]], Any],
                 broadcaster: Callable[[str, dict], None] | None = None,
                 history_depth: int = 200):
        self._heap: List[tuple[int, float, str]] = []
        self._cv = asyncio.Condition()

        self.run_graph = run_graph_func
        self.broadcast = broadcaster or (lambda *_: None)

        self.jobs: Dict[str, Job] = {}
        self.history: deque[Job] = deque(maxlen=history_depth)

        self._worker_task: asyncio.Task | None = None

    async def submit(self, graph_json: dict, priority: int = 0) -> Job:
        job = Job(graph=graph_json, priority=priority)
        self.jobs[job.id] = job
        async with self._cv:
            heapq.heappush(self._heap, (priority, time.time(), job.id))
            self._cv.notify()
        self.broadcast("status", job.__dict__)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list_history(self, max_items: int | None = None):
        if max_items:
            return list(self.history)[:max_items]
        return list(self.history)

    async def start(self):
        if not self._worker_task:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

    # ---------- inner implementation ----------

    async def _pop(self) -> Job:
        async with self._cv:
            while not self._heap:
                await self._cv.wait()
            _, _, job_id = heapq.heappop(self._heap)
        return self.jobs[job_id]

    async def _worker(self):
        while True:
            job = await self._pop()
            await self._run_job(job)

    async def _run_job(self, job: Job):
        job.status = JobStatus.running
        try:
            graph = nndeploy.Graph.from_dict(job.graph)
            await asyncio.to_thread(graph.run)
            job.status = JobStatus.done
            job.progress = 1.0
        except Exception as e:
            job.status = JobStatus.error
            job.error = str(e)
        self.history.appendleft(job)

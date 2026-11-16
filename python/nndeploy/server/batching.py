"""Bounded dynamic batches inside a single resident Worker."""
import asyncio
import time

from .errors import ServerError
from .worker import Worker


class BatchWorker(Worker):
    """One owning thread for init, compatibility keys, batch execution and close."""
    def __init__(self, *args):
        super().__init__(*args)
        self.jobs = {}
        self.wake = asyncio.Event()
        self.scheduler = None
        self.draining = False

    async def start(self, *, deadline=None):
        await super().start(deadline=deadline)
        self.scheduler = asyncio.create_task(self._run())

    def cancel(self, identity):
        job = self.jobs.get(identity)
        if job is not None:
            job['cancelled'] = True
            self.wake.set()

    def _finish(self, job, result=None, failure=None):
        identity = job['values']['_task_id']
        if job['cancelled']:
            failure = ServerError(409, 'task_cancelled', 'Batch request cancelled')
        self.jobs.pop(identity, None)
        if failure is not None:
            # Errors can be reused by an algorithm, but mutable task IDs cannot.
            error = ServerError(failure.status_code, failure.code, str(failure),
                                task_id=identity, retryable=failure.retryable) if isinstance(failure, ServerError) else (
                ServerError(500, 'batch_execution_failed', 'Batch execution failed', task_id=identity))
            job['future'].set_exception(error)
        else:
            job['future'].set_result(result)
        self.wake.set()

    async def deploy(self, values, emit=None, *, before_run):
        if self.draining or self.closed:
            raise ServerError(503, 'worker_stopping', 'Batch worker is stopping')
        if len(self.jobs) >= self.config.execution.request_slots:
            raise ServerError(429, 'worker_full', 'Batch worker capacity reached', retryable=True)
        identity = values['_task_id']
        if identity in self.jobs:
            raise ServerError(409, 'duplicate_task', 'Task is already owned by this worker')
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        job = dict(values=values, future=future, before_run=before_run, key=None, ready=False,
                   cancelled=False, arrived=time.monotonic(), active=False, deadline=None)
        self.jobs[identity] = job
        try:
            # The executor queue is bounded by request_slots, including keys
            # waiting behind active native work. Never run model methods on HTTP.
            async def prepare():
                return await asyncio.shield(loop.run_in_executor(self.pool, self.executor.batch_key, values))
            # Compatibility checks can decode arrays/Tensors. With a temporary
            # memory declaration they must also acquire one request's budget.
            job['key'] = await self.device_prepare([values], prepare) if hasattr(self, 'device_prepare') else await prepare()
        except Exception as exc:
            self._finish(job, failure=exc)
        else:
            job['ready'] = True
            self.wake.set()
        return await asyncio.shield(future)

    async def _run(self):
        limits = self.config.execution.batching
        while True:
            self.wake.clear()
            ready = [job for job in self.jobs.values() if job['ready'] and not job['active']]
            for job in ready:
                if job['cancelled']:
                    self._finish(job)
            ready = [job for job in ready if not job['cancelled']]
            if not ready:
                if self.draining and not self.jobs:
                    return
                await self.wake.wait()
                continue
            # Oldest compatible group first: later traffic cannot keep extending
            # an earlier request's collection window or starve a rare shape.
            first = min(ready, key=lambda job: job['arrived'])
            group = [job for job in ready if job['key'] == first['key']][:limits.max_size]
            remaining = first['arrived'] + limits.max_wait_ms / 1000 - time.monotonic()
            if len(group) < limits.max_size and remaining > 0 and not self.draining:
                try:
                    await asyncio.wait_for(self.wake.wait(), remaining)
                except asyncio.TimeoutError:
                    pass
                continue
            async def execute():
                active = []
                for job in group:
                    try:
                        # This also checks DB ownership/deadlines immediately before
                        # execution, after time spent collecting compatible requests.
                        job['deadline'] = await job['before_run']()
                    except Exception as exc:
                        self._finish(job, failure=exc)
                    else:
                        if job['cancelled']:
                            self._finish(job)
                        else:
                            job['active'] = True
                            active.append(job)
                if not active:
                    return
                try:
                    results = await asyncio.shield(asyncio.get_running_loop().run_in_executor(
                        self.pool, self._execute_batch, active))
                except Exception as exc:
                    for job in active:
                        self._finish(job, failure=exc)
                else:
                    for job, result in zip(active, results):
                        self._finish(job, failure=result) if isinstance(result, ServerError) else self._finish(job, result=result)
            if hasattr(self, 'device_run'):
                try:
                    await self.device_run([job['values'] for job in group], execute)
                except Exception as exc:
                    for job in group:
                        if job['future'].done():
                            continue
                        failure = exc
                        if isinstance(exc, ServerError) and exc.code in ('task_cancelled', 'queue_timeout', 'execution_ownership_lost'):
                            try:
                                await job['before_run']()
                            except Exception as invalid:
                                failure = invalid
                        self._finish(job, failure=failure)
            else:
                await execute()

    def _execute_batch(self, jobs):
        # The owning thread may have been busy calculating another key. Recheck
        # at the actual algorithm call, not only when submitting work to the pool.
        results, active = [None] * len(jobs), []
        for index, job in enumerate(jobs):
            if job['cancelled']:
                results[index] = ServerError(409, 'task_cancelled', 'Cancelled before batch execution')
            elif job['deadline'] is not None and time.monotonic() >= job['deadline']:
                results[index] = ServerError(504, 'queue_timeout', 'Task expired before batch execution')
            else:
                active.append((index, job))
        if active:
            try:
                values = self.executor.deploy_batch([job['values'] for _, job in active])
            except Exception as exc:
                failure = exc if isinstance(exc, ServerError) else ServerError(500, 'batch_execution_failed', 'Batch execution failed')
                values = [failure] * len(active)
            for (index, _), value in zip(active, values):
                results[index] = value
        return results

    async def close(self):
        self.draining = True
        self.wake.set()
        if self.scheduler is not None:
            await asyncio.shield(self.scheduler)
        await super().close()

"""Per-Server metrics with bounded labels and no task payloads or identifiers."""
import asyncio
from contextlib import contextmanager
from functools import wraps
import time

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily
from starlette.requests import ClientDisconnect


BUCKETS = (.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 300, 900)


def outcome(error):
    if error is None:
        return 'success'
    if isinstance(error, asyncio.CancelledError):
        return 'cancelled'
    code = getattr(error, 'code', '')
    if code in {'task_cancelled', 'queue_timeout', 'execution_timeout', 'queue_full', 'server_full',
                'not_ready', 'entry_changed', 'invalid_request', 'invalid_file', 'recording_unavailable',
                'idempotency_conflict'}:
        return code
    return 'error'


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.stages = Histogram('nndeploy_execution_stage_seconds',
                                'Summed stage time per request (per physical batch for batched compute); nested stages can overlap.',
                                ['entry', 'stage', 'outcome'], buckets=BUCKETS, registry=self.registry)
        self.http = Histogram('nndeploy_http_request_seconds', 'HTTP lifetime through final response body or disconnect.',
                              ['method', 'route', 'entry', 'status'], buckets=BUCKETS, registry=self.registry)
        self.first_byte = Histogram('nndeploy_http_first_byte_seconds', 'Time until the first nonempty response body.',
                                    ['method', 'route', 'entry'], buckets=BUCKETS, registry=self.registry)
        self.operations = Histogram('nndeploy_operation_seconds', 'Server operation duration; requests include worker dispatch and output conversion.',
                                    ['operation', 'entry', 'outcome'], buckets=BUCKETS, registry=self.registry)
        self.queue_wait = Histogram('nndeploy_queue_wait_seconds', 'Wait until recorded worker admission; shared batch collection may follow.',
                                    ['entry'], buckets=BUCKETS, registry=self.registry)
        self.storage = Histogram('nndeploy_storage_operation_seconds', 'Parent-process managed storage calls, including DB and content work.',
                                 ['operation', 'outcome'], buckets=BUCKETS, registry=self.registry)
        self.database = Histogram('nndeploy_database_statement_seconds', 'Parent-process SQL execution time, excluding pool wait and commit.',
                                  ['outcome'], buckets=BUCKETS, registry=self.registry)
        self.redis = Histogram('nndeploy_redis_command_seconds', 'Redis commands including network wait; blocking reads include idle wait.',
                               ['entry', 'outcome'], buckets=BUCKETS, registry=self.registry)
        self.cleanup = Counter('nndeploy_file_cleanup_total', 'Observed file cleanup outcomes in this process.',
                               ['outcome'], registry=self.registry)
        self.terminal = Counter('nndeploy_tasks_terminal_total', 'Task terminal commits acknowledged to this process; not an audit ledger.',
                                ['entry', 'state'], registry=self.registry)
        self.audit_errors = Counter('nndeploy_audit_errors_total', 'Audit storage failures or unconfirmed waits in this process.',
                                    ['operation'], registry=self.registry)
        self.log_dropped = Counter('nndeploy_logs_dropped_total', 'Observed queue loss reports and WebSocket log drops in this process.',
                                   ['sink'], registry=self.registry)
        from prometheus_client import ProcessCollector, PlatformCollector
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)

    @contextmanager
    def measure(self, operation, entry):
        started, error = time.perf_counter(), None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            self.operations.labels(operation, entry, outcome(error)).observe(time.perf_counter() - started)

    def attach_database(self, db):
        from sqlalchemy import event
        db.metrics = self

        def before(conn, cursor, statement, parameters, context, executemany):
            context._nndeploy_metrics_started = time.perf_counter()

        def record(context, result):
            started = getattr(context, '_nndeploy_metrics_started', None)
            if started is not None:
                context._nndeploy_metrics_started = None
                self.database.labels(result).observe(time.perf_counter() - started)

        def after(conn, cursor, statement, parameters, context, executemany):
            record(context, 'success')

        def failed(context):
            record(context.execution_context, 'error')

        event.listen(db.engine, 'before_cursor_execute', before)
        event.listen(db.engine, 'after_cursor_execute', after)
        event.listen(db.engine, 'handle_error', failed)

    def attach_redis(self, client, entry):
        execute = client.execute_command

        @wraps(execute)
        async def measured(*args, **kwargs):
            started, error = time.perf_counter(), None
            try:
                return await execute(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                self.redis.labels(entry, outcome(error)).observe(time.perf_counter() - started)

        client.execute_command = measured

    @staticmethod
    def snapshot(server):
        """Called on the owning loop: no DB calls or request payload copies."""
        families = []

        def gauge(name, help, labels=()):
            family = GaugeMetricFamily('nndeploy_' + name, help, labels=labels)
            families.append(family)
            return family

        gauge('inflight', 'Local accepted tasks, or in-progress admission calls when using shared queues.').add_metric([], server.inflight)
        gauge('inflight_limit', 'Maximum local tasks or concurrent shared admission calls; shared task capacity is enforced in DB.').add_metric([], server.max_inflight_tasks)
        gauge('shared_result_waiters', 'Active shared task result readers, independent of admission capacity.').add_metric([], len(server.waiters))
        gauge('stream_connections', 'Active SSE connections in this process.').add_metric([], server.stream_connections)
        gauge('sync_connections', 'Synchronous HTTP responses held through body delivery.').add_metric([], server.sync_connections)
        gauge('sync_connections_limit', 'Maximum synchronous HTTP responses in this process.').add_metric([], server.max_sync_connections)
        ready = gauge('entry_accepting', 'Whether this entry accepts requests.', ['entry'])
        workers = gauge('workers', 'Model instances held by this process, including instances being drained.', ['entry', 'state'])
        queue = gauge('queue_tasks', 'Local queue counts; excludes shared queues.', ['entry', 'state'])
        capacity = gauge('entry_capacity', 'Configured queue plus worker request slots.', ['entry'])
        shared = gauge('shared_inflight', 'Cached global queue usage; replicas report the same value, do not sum.', ['entry'])
        age = gauge('shared_usage_age_seconds', 'Age of the last successful shared queue usage sample.', ['entry'])
        for name, entry in list(server.entries.items()):
            ready.add_metric([name], int(server.accepting(name, entry)))
            workers.add_metric([name, 'current'], len(entry['workers']))
            retired = server.retiring_entries.get(name, [])
            workers.add_metric([name, 'draining'], sum(len(item['workers']) for item in retired))
            capacity.add_metric([name], entry['config'].execution.capacity)
            if server.redis_config is None:
                counts = dict(pending=0, dispatched=0, running=0)
                for version in [entry, *retired]:
                    if version.get('queue') is not None:
                        for state, value in version['queue'].counts().items():
                            counts[state] += value
                for state, value in counts.items():
                    queue.add_metric([name, state], value)
            elif entry.get('metrics_shared_usage') is not None:
                sampled_at, usage = entry['metrics_shared_usage']
                shared.add_metric([name], usage['inflight'])
                age.add_metric([name], max(0, time.monotonic() - sampled_at))
        return families

    def render(self, families):
        # Each scrape owns its snapshot. Encoding and process collection run off-loop.
        class Snapshot:
            def collect(self):
                return iter(families)
        registry = CollectorRegistry(auto_describe=False)
        registry.register(self.registry)
        registry.register(Snapshot())
        return generate_latest(registry)


def observe_admission(function):
    @wraps(function)
    async def wrapped(self, name, *args, **kwargs):
        label = name if name in self.entries else '__unknown__'
        with self.metrics.measure('admission', label):
            result = await function(self, name, *args, **kwargs)
        from .request import event, get_request_id
        event('task_selected', request_id=get_request_id(), task_id=result[0], entry=label)
        return result
    return wrapped


def observe_storage(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        metrics = getattr(self, 'metrics', None)
        if metrics is None:
            return function(self, *args, **kwargs)
        started, error = time.perf_counter(), None
        try:
            result = function(self, *args, **kwargs)
            if function.__name__ == 'cleanup':
                for state in ('deleted', 'skipped', 'failed'):
                    metrics.cleanup.labels(state).inc(result[state])
            return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            metrics.storage.labels(function.__name__, outcome(error)).observe(time.perf_counter() - started)
    return wrapped


class MetricsMiddleware:
    def __init__(self, app, server):
        self.app, self.server = app, server

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] == '/metrics':
            return await self.app(scope, receive, send)
        started, status, first_byte = time.perf_counter(), 'aborted', None

        async def observe(message):
            nonlocal status, first_byte
            await send(message)
            if message['type'] == 'http.response.start':
                status = str(message['status'])
            elif message['type'] == 'http.response.body' and message.get('body') and first_byte is None:
                first_byte = time.perf_counter() - started

        try:
            await self.app(scope, receive, observe)
        except ClientDisconnect:
            if status == 'aborted':
                status = '499'
            raise
        except Exception:
            if status == 'aborted':
                status = '500'
            raise
        finally:
            from .request import http_labels
            method, route, entry = http_labels(scope, self.server)
            metrics = self.server.metrics
            metrics.http.labels(method, route, entry, status).observe(time.perf_counter() - started)
            if first_byte is not None:
                metrics.first_byte.labels(method, route, entry).observe(first_byte)

"""Typed entry state and explicit generation construction.

Mapping access is retained for existing integrations. Runtime ownership is
never copied when constructing a replacement generation.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Literal, Protocol, TypedDict, TYPE_CHECKING

from .config import ServerConfig

if TYPE_CHECKING:
    from .task_queue import TaskQueue, RedisTaskQueue
    from .control_guard import ControlGuard


class WorkerProtocol(Protocol):
    config: ServerConfig
    shared_holder: str | None

    async def start(self, *, deadline: float | None = None) -> None: ...
    async def deploy(self, values: dict, emit: Callable | None = None) -> Any: ...
    async def close(self) -> None: ...


class EntrySource(TypedDict, total=False):
    base_dir: str
    autodetect: bool
    untyped: bool
    workflow: dict | None
    graph: Any
    graph_factory: Callable | None
    deploy: Callable | None
    handler_factory: Callable | None
    input_model: Any
    output_model: Any
    identity: str
    resource_snapshot: Any
    resource_fallback: Any


class EntryRuntime(TypedDict, total=False):
    queue: TaskQueue | RedisTaskQueue
    wake: asyncio.Event
    admitted: set[str]
    consumer_runs: list[asyncio.Task]
    consumer_failed: bool
    local_stop: bool
    maintenance: asyncio.Task
    file_holder: str
    control_guard: ControlGuard
    device_guard: ControlGuard
    release_pending: bool
    shared_holders: list[str]
    shared_candidate: str
    shared_draining: bool
    shared_start_repair: bool
    shared_recovery_after: float
    metrics_shared_usage: tuple[float, dict]
    saved_token: str | None
    shared_key: str
    workflow_id: str
    restore_enabled: bool


class Entry(EntryRuntime):
    config: ServerConfig
    source: EntrySource
    status: Literal['stopped', 'starting', 'ready', 'reloading', 'stopping', 'failed']
    error: str | None
    schema: dict
    untyped: bool
    workers: list[WorkerProtocol]
    consumers: list[asyncio.Task]
    futures: dict[str, asyncio.Future]
    closing: asyncio.Task | None


def new_entry(config: ServerConfig, source: EntrySource, *, untyped: bool) -> Entry:
    return Entry(config=config, source=source, status='stopped', error=None,
                 workers=[], consumers=[], consumer_runs=[], futures={}, admitted=set(),
                 untyped=untyped, schema=config.public_schema(), closing=None, shared_holders=[])


def replacement(old: Entry, *, source: EntrySource | None = None,
                config: ServerConfig | None = None) -> Entry:
    source = dict(old['source']) if source is None else source
    candidate = new_entry(config or old['config'], source, untyped=source.get('untyped', old['untyped']))
    # Persisted identity belongs to the logical entry, while locks, queues,
    # consumers and leases belong exclusively to one running generation.
    for key in ('saved_token', 'shared_key', 'workflow_id', 'restore_enabled'):
        if key in old:
            candidate[key] = old[key]
    return candidate

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import APIRouter, Request, status, UploadFile, File, Query
from fastapi import Depends
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from typing import Set, Dict, Any, Optional, Literal
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import os
import json
import asyncio
import uuid
import logging
import sqlite3
import shutil
import requests
import contextvars
from contextlib import asynccontextmanager
import unicodedata
import chardet
from queue import Full
import nndeploy.dag

from nndeploy.dag.node import add_global_import_lib, import_global_import_lib
from nndeploy import get_type_enum_json
from .download_progress_handler import DownloadProgressHandler
from .utils import extract_encode_output_paths, _handle_urls, _save_plugin_upload
from .task_queue import TaskQueue
from .task_queue import ExecutionStatus
from .schemas import (
    EnqueueRequest,
    EnqueueResponse,
    QueueStateResult,
    QueueStateResponse,
    HistoryItem,
    HistoryResponse,
    UploadResponse,
    NodeListResponse,
    WorkFlowSaveResponse,
    WorkFlowListResponse,
    WorkFlowLoadResponse,
    WorkFlowDeleteResponse,
    ParamTypeResponse,
    TemplateDirListResponse,
    TemplateJsonListResponse,
    TemplateLoadResponse
)
from .files import router as files_router
from .files import get_workdir
from .logging_taskid import set_task_id, reset_task_id, run_func_in_copied_context, scoped_stdio_to_logging
from .db import DB
from .metrics import Metrics, MetricsMiddleware, observe_admission
from .audit import audited

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise

class Server:
    instance: "NnDeployServer" = None

    def __init__(self, args=None, job_mp_queue=None, plugin_update_q=None, cancel_event_queue=None,
                 *, workflow=None, graph=None, graph_factory=None, deploy=None, handler_factory=None,
                 name=None, config=None, workers=None, input_model=None, output_model=None, router=None,
                 data_dir="./resources/server", max_inflight_tasks=64, admin_token=None, database_url=None, storage_config=None,
                 redis_config=None, auth_config=None, cors_origins=None, max_sync_connections=None, device_limits=None,
                 database_timeout_seconds=None, audit_config=None, editor_history_retention_seconds=None, mcp_config=None):
        from .environment import load_server_env
        load_server_env()
        self.mcp = None
        self.mcp_config = None
        if mcp_config is not None:
            from .mcp.config import MCPConfig
            self.mcp_config = MCPConfig.model_validate(mcp_config)
            if self.mcp_config.enabled:
                try:
                    from .mcp.app import MCP
                except ImportError as exc:
                    raise ImportError('MCP requires the optional nndeploy[server-mcp] dependencies') from exc
        DB._validate_retention(editor_history_retention_seconds)
        self.editor_history_retention_seconds = editor_history_retention_seconds
        self.database_timeout_seconds = database_timeout_seconds
        from .auth import origin_key
        self.cors_origins = list(cors_origins or [])
        limit = max_inflight_tasks if max_sync_connections is None else max_sync_connections
        if type(limit) is not int or limit < 1:
            raise ValueError('max_sync_connections must be a positive integer')
        self.max_sync_connections = limit
        self.sync_connections = 0
        self.sync_connections_by_subject = {}
        for origin in self.cors_origins:
            origin_key(origin)
        if args is None:
            self._init_api(data_dir, max_inflight_tasks, admin_token, database_url=database_url, storage_config=storage_config,
                           redis_config=redis_config, auth_config=auth_config, device_limits=device_limits, audit_config=audit_config)
            if any(value is not None for value in (workflow, graph, graph_factory, deploy, handler_factory)):
                self.add(name=name, workflow=workflow, graph=graph, graph_factory=graph_factory,
                         deploy=deploy, handler_factory=handler_factory, config=config, workers=workers,
                         input_model=input_model, output_model=output_model, router=router)
            return
        from .template import WorkflowTemplateManager
        NnDeployServer.instance = self
        self.loop: Optional[asyncio.AbstractEventLoop] = None # lazy loading
        self.args = args
        self.app = FastAPI(
            title="nndeploy backend",
            version="1.0.0"
        )

        self.workflow_dir = Path(self.args.resources) / "workflow"
        self.db_path = Path(self.args.resources) / "db" / "nndeploy.db"
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure required subdirectories exist
        required_dirs = ["images", "videos", "audios", "models", "others"]
        for sub in required_dirs:
            d = Path(self.args.resources) / sub
            d.mkdir(parents=True, exist_ok=True)

        self.db = DB(database_url or self.db_path, timeout_seconds=self.database_timeout_seconds,
                     backup_dir=self.db_path.parent / 'backups').init_schema()

        self.cancel_event_queue = cancel_event_queue
        self.plugin_update_q = plugin_update_q
        self.sockets: set[WebSocket] = set()
        self.ws_task_map: dict[WebSocket, set[str]] = {}
        self.task_ws_map: dict[str, set[WebSocket]] = {}
        self._init_api(Path(self.args.resources) / 'server', max_inflight_tasks, admin_token, editor=True,
                       storage_config=storage_config, redis_config=redis_config, auth_config=auth_config,
                       device_limits=device_limits, audit_config=audit_config)
        from .db import _EditorHistory
        history = _EditorHistory(self.db, self.config_scope, self.data_dir)
        try:
            self.queue = TaskQueue(self, job_mp_queue, history_store=history)
            self._register_routes(args)
            template_parent = WorkflowTemplateManager.init_templates()
            self.template_path = Path(template_parent) / "nndeploy-workflow"
        except BaseException:
            history.close()
            raise

    def _init_api(self, data_dir, max_inflight_tasks, admin_token, *, editor=False, database_url=None, storage_config=None,
                  redis_config=None, auth_config=None, device_limits=None, audit_config=None):
        from contextlib import asynccontextmanager
        from .api.deploy import register_routes
        from .storage import LocalStorage
        from .dependencies import Authentication
        self.authentication = Authentication(auth_config, admin_token)
        if max_inflight_tasks < 1:
            raise ValueError("max_inflight_tasks must be positive")
        self.entries = {}
        self._asset_leases = {}
        from .registry import Registry
        self._registry = Registry()
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        from .resources import server_scope
        self.config_scope = server_scope(self.data_dir)
        if not editor:
            self.db = DB(database_url or self.data_dir / "server.db", timeout_seconds=self.database_timeout_seconds,
                         backup_dir=self.data_dir / 'db-backups').init_schema()
        self.db.init_tasks()
        self.metrics = Metrics()
        self.metrics.attach_database(self.db)
        self.storage = LocalStorage(self.data_dir / "files", self.db, config=storage_config)
        self.storage.metrics = self.metrics
        from .assets import AssetCleanup
        self._asset_cleanup = AssetCleanup(self.data_dir, self.db)
        self.max_inflight_tasks = max_inflight_tasks
        self.inflight = 0
        self.stream_connections = 0
        self.admin_token = admin_token
        self.started = False
        self.shutting_down = False
        self.control_tasks = set()
        self.entry_controls = {}
        self.retiring_entries = {}
        self.starting_task = None
        self.closing_task = None
        self.task_futures = {}
        self.running_workers = {}
        from .redis import RedisConfig
        self.redis_config = RedisConfig.model_validate(redis_config) if redis_config is not None else None
        self.event_notifications = None
        if self.redis_config is not None:
            from .redis import RedisEvents
            self.event_notifications = RedisEvents(self.redis_config, self.db.database_identity())
        from .rate_limit import RequestLimiter
        self.request_limiter = RequestLimiter(self.authentication.request_rate, self.redis_config)
        from .audit import Audit
        self.audit = Audit(self, audit_config)
        from .shared_runtime import SharedRuntime
        self.shared_runtime = SharedRuntime(self) if self.redis_config is not None else None
        from .control import Controls
        self.controls = Controls(self)
        from .devices import Devices
        self.devices = Devices(self, device_limits) if device_limits is not None else None
        self.waiters = {}
        self.shared_control = None
        self.task_maintenance = None
        self.maintenance_stop = asyncio.Event()

        @asynccontextmanager
        async def lifespan(app):
            try:
                self.loop = asyncio.get_running_loop()
                await self.start()
                if self.mcp is None:
                    yield
                else:
                    async with self.mcp.lifespan():
                        yield
            finally:
                await self.close()

        if editor:
            self.app.router.lifespan_context = lifespan
        else:
            self.app = FastAPI(title="nndeploy Server", lifespan=lifespan)
        self.app.state.server = self
        self.editor_access = None
        if editor:
            from .auth import EditorAccess, EditorAccessMiddleware
            self.editor_access = EditorAccess(self.authentication, self.cors_origins, self.data_dir)
            self.editor_access.register(self.app)
            self.app.add_middleware(EditorAccessMiddleware, access=self.editor_access)
        register_routes(self)
        if self.mcp_config is not None and self.mcp_config.enabled:
            from .mcp.app import MCP
            self.mcp = MCP(self, self.mcp_config)
        self.app.add_middleware(MetricsMiddleware, server=self)
        if self.cors_origins:
            self.app.add_middleware(CORSMiddleware, allow_origins=self.cors_origins,
                                    allow_credentials=True, allow_methods=['*'], allow_headers=['*'],
                                    expose_headers=['X-Request-ID', 'X-Task-ID', 'Retry-After'])
        from .request import RequestMiddleware
        self.app.add_middleware(RequestMiddleware, server=self)

    def register(self, name, factory):
        """Register trusted algorithm setup code without importing its dependencies."""
        self._registry.register(name, factory)
        return self

    def add(self, *, name=None, workflow=None, graph=None, graph_factory=None, deploy=None,
            handler_factory=None, config=None, workers=None, input_model=None, output_model=None,
            router=None, algorithm=None, options=None, mcp=None):
        from .config import ServerConfig, load_json, suggest_config
        import copy
        if algorithm is not None:
            if any(value is not None for value in (workflow, graph, graph_factory, deploy, handler_factory)):
                raise ValueError('Use either a registered algorithm or a direct execution source')
            arguments = self._registry.create(algorithm, options)
            for key, value in dict(config=config, input_model=input_model, output_model=output_model, router=router, mcp=mcp).items():
                if value is not None:
                    arguments[key] = value
            return self.add(name=name, workers=workers, **arguments)
        if options is not None:
            raise ValueError('Options require a registered algorithm')
        if sum(value is not None for value in (workflow, graph, graph_factory, deploy, handler_factory)) != 1:
            raise ValueError("Provide exactly one workflow, graph, graph_factory, deploy or handler_factory")
        export = None
        if mcp is not None:
            from .mcp.config import ToolConfig
            export = ToolConfig.model_validate(mcp)
            if self.mcp is None:
                raise ValueError('Enable mcp_config before exporting an MCP tool')
            if export.name in self.mcp.exports:
                raise ValueError(f'Duplicate MCP tool name: {export.name}')
        supplied_config = config is not None
        base_dir = Path.cwd()
        if isinstance(workflow, (str, Path)):
            path = Path(workflow)
            base_dir = path.resolve().parent
            if config is None and path.with_suffix(".server.json").is_file():
                config = path.with_suffix(".server.json")
                supplied_config = True
            workflow = load_json(path)
        if isinstance(config, (str, Path)):
            config = load_json(config)
        if config is None:
            if workflow is not None:
                config = suggest_config(workflow)
            elif graph is not None:
                config = suggest_config(json.loads(graph.serialize()))
            else:
                config = {}
        elif isinstance(config, ServerConfig):
            config = config.model_dump()
        config = copy.deepcopy(config)
        if name is not None:
            config['name'] = name
        if workers is not None:
            config.setdefault('execution', {})['workers'] = workers
        if export is not None:
            config['mcp'] = export.model_dump()
        value = ServerConfig.model_validate(config)
        if self.mcp is not None:
            self.mcp.validate_export(value.name, value.mcp)
        if value.name in self.entries:
            raise ValueError(f"Duplicate entry name: {value.name}")
        if graph is not None and value.execution.workers != 1:
            raise ValueError("Multiple workers require a graph_factory, not a shared Graph")
        entry = dict(
            config=value, source=dict(base_dir=str(base_dir), autodetect=not supplied_config and graph_factory is not None, workflow=copy.deepcopy(workflow), graph=graph,
                                     graph_factory=graph_factory, deploy=deploy, handler_factory=handler_factory,
                                     input_model=input_model, output_model=output_model),
            status='stopped', error=None, workers=[], consumers=[], futures={},
            untyped=not supplied_config and input_model is None and (deploy is not None or handler_factory is not None),
            schema=value.public_schema(), closing=None)
        if value.mcp is not None and entry['untyped']:
            raise ValueError('MCP exports require an input_model or explicit input config')
        entry['source']['untyped'] = entry['untyped']
        if self.redis_config is not None:
            from .redis import source_identity
            entry['source']['identity'] = source_identity(entry['source'])
        if router is not None:
            from .api.extensions import prepare_router
            from starlette.routing import Mount
            mounted = prepare_router(self, value.name, router)
            # The editor's SPA catch-all must remain behind newly added HTTP routes.
            routes = self.app.router.routes
            insert_at = next((index for index, route in enumerate(routes)
                              if isinstance(route, Mount) and route.path == ''), len(routes))
            previous_count = len(routes)
            self.app.include_router(mounted)
            added = routes[previous_count:]
            del routes[previous_count:]
            routes[insert_at:insert_at] = added
            self.app.openapi_schema = None
        self.entries[value.name] = entry
        return self

    @audited
    async def start_workflow(self, request):
        """Save using the editor's existing workflow operation and start its entry."""
        from .config import ServerConfig, load_json, suggest_config
        from .errors import ServerError
        from .schemas import EnqueueRequest
        data = request.get('businessContent')
        if not isinstance(data, dict):
            raise ServerError(422, 'invalid_request', 'businessContent must be a workflow object')
        title = data.get('name_', '')
        if not isinstance(title, str) or not title or title in ('.', '..') or any(c in title for c in ('/', '\\', '\x00')):
            raise ServerError(422, 'invalid_request', 'Workflow name must be a nonempty filename')
        saved = await self.save_workflow(EnqueueRequest(root=request))
        identity = saved.result['id']
        path = self.db.get_workflow_path(identity)
        sidecar = path.with_suffix('.server.json')
        config = load_json(sidecar) if sidecar.is_file() else suggest_config(data)
        if set(config) == {'name'}:
            config = dict(suggest_config(data), **config)
        # Existing editor entries gain newly exposed node parameters on Update.
        # Explicit bindings retain their names and configuration.
        configured_nodes = {tuple(value.get('node', [])): value for value in config.get('parameters', {}).values()}
        for parameter_name, field in suggest_config(data)['parameters'].items():
            existing = configured_nodes.get(tuple(field['node']))
            if existing is None:
                config.setdefault('parameters', {}).setdefault(parameter_name, field)
            elif existing.get('type') == 'json':
                # The editor's current Graph is the source of node defaults.
                existing['default'] = field['default']
        previous_name = config.get('name') if sidecar.is_file() else None
        if 'exposed_parameters' in request:
            from .parameters import parameter_paths
            selected = request['exposed_parameters']
            if not isinstance(selected, dict) or selected.keys() - config.get('parameters', {}).keys():
                raise ServerError(422, 'invalid_request', 'Unknown exposed node parameters')
            for parameter_name, field in config.get('parameters', {}).items():
                field['exposed'] = selected.get(parameter_name, [])
        name = request.get('name', previous_name if previous_name and previous_name != 'workflow-' + identity else title)
        if not isinstance(name, str) or not name.strip():
            raise ServerError(422, 'invalid_request', 'Server name must not be empty')
        name = name.strip()
        config['name'] = name
        if 'mcp' in request:
            config['mcp'] = request['mcp']
        try:
            validated = ServerConfig.model_validate(config)
            if validated.mcp is not None and self.mcp is None and 'mcp' in request:
                raise ServerError(409, 'mcp_disabled', 'Start the editor with --mcp to enable MCP exports.')
            if self.mcp is not None:
                self.mcp.validate_export(name, validated.mcp)
        except ValueError as exc:
            raise ServerError(422, 'invalid_request', str(exc)) from exc
        if name in self.entries and name != previous_name and self.entries[name].get('workflow_id') != identity:
            raise ServerError(409, 'name_in_use', 'This Server name is already used by another workflow. Choose another name.')
        async def update_entry():
            if previous_name and previous_name != name and previous_name in self.entries:
                await self.stop_entry(previous_name)
            # Attaching a persisted entry exposes it to the background follower.
            # Own the whole operation before that await, so it cannot initialize
            # an old (possibly broken) Graph ahead of the editor's replacement.
            if self.shared_runtime is not None and (name not in self.entries or not self.entries[name].get('shared_key')):
                await self.shared_runtime.attach_workflow(name)
            if name in self.entries and self.entries[name]['status'] == 'ready':
                await self._reload_entry(name, workflow=data, config=config)
            elif (self.shared_runtime is not None and name in self.entries and self.entries[name].get('shared_key')
                  and self.entries[name]['status'] in ('stopped', 'failed') and not self.entries[name]['workers']):
                await self._reload_entry(name, workflow=data, config=config, start_stopped=True)
            else:
                if name in self.entries and self.entries[name]['status'] not in ('stopped', 'failed'):
                    raise ServerError(409, 'entry_busy', 'Entry is changing state', retryable=True)
                previous = self.entries.get(name)
                if previous and (previous['workers'] or previous.get('shared_holders')):
                    raise ServerError(409, 'entry_busy', 'Previous workers have not confirmed release', retryable=True)
                self.entries.pop(name, None)
                if previous is not None:
                    self._release_assets(previous)
                self.add(name=name, workflow=data, config=config)
                self.entries[name]['workflow_id'] = identity
                # Remember the selected name even if initial model loading
                # fails, so a retry/restart repairs this same workflow entry.
                if not sidecar.is_file():
                    self._write_json_replace(sidecar, {'name': name})
                await self._start_entry(name, self._entry(name))
            self._write_json_replace(sidecar, config)
            return dict(workflow_id=identity, **self.describe(name))
        async def guarded_update():
            from contextlib import nullcontext
            with self.mcp.reserve(name, validated.mcp) if self.mcp is not None else nullcontext():
                return await update_entry()
        return await self._control_entry(name, 'reload', guarded_update)

    def create_app(self):
        return self.app

    def run(self, host="127.0.0.1", port=8000, **options):
        import uvicorn
        from .request import logging_config
        from .logging_taskid import install_taskid_logrecord_factory
        install_taskid_logrecord_factory()
        options.setdefault('access_log', False)
        options.setdefault('log_config', logging_config(editor=hasattr(self, 'workflow_dir')))
        uvicorn.run(self.app, host=host, port=port, **options)

    @audited
    async def start(self):
        from .errors import ServerError
        if self.shutting_down:
            raise ServerError(503, 'server_stopping', 'Create a new Server after shutdown')
        if self.starting_task is None:
            self.starting_task = self.own(self._start())
        await asyncio.shield(self.starting_task)

    async def _start(self):
        if self.redis_config is not None:
            import hashlib
            import socket
            identity = self.storage.backend_id
            if self.storage.backend is None:
                identity += ':' + socket.gethostname() + ':' + str(self.storage.root)
            await asyncio.to_thread(self.db.register_namespace_storage, self.redis_config.namespace,
                                    hashlib.sha256(identity.encode()).hexdigest())
        await asyncio.to_thread(self.storage.recover_pending)
        await self.controls.database(self.controls.reconcile)
        from .config import load_json
        for path in (self.data_dir / 'entries').glob('*.json'):
            saved = load_json(path)
            await asyncio.to_thread(self.db.save_server_config, self.config_scope, saved['config'].get('name', 'default'),
                                    saved, only_if_absent=True)
        for record in await asyncio.to_thread(self.db.load_server_configs, self.config_scope, with_tokens=True):
            saved = record['data']
            name = saved['config'].get('name', 'default')
            if saved.get('kind') == 'graph_factory':
                # Resource metadata cannot restore or import executable code.
                # Explicitly registered factories load it during Start.
                continue
            if name not in self.entries:
                if saved.get('shared_entry_key'):
                    if self.shared_runtime is None:
                        continue
                    active = await asyncio.to_thread(self.db.shared.get, saved['shared_entry_key'])
                    if active['data']['kind'] != 'workflow':
                        raise ValueError('Saved shared code requires explicit registration')
                    saved = dict(active['data'], enabled=True)
                self.add(name=name, workflow=saved['workflow'], config=saved['config'])
                self.entries[name]['saved_token'] = record['write_token']
                if saved.get('identity'):
                    self.entries[name]['source']['identity'] = saved['identity']
                self.entries[name]['restore_enabled'] = saved['enabled'] and self.entries[name]['config'].execution.restore_on_restart
        failures = []
        for name in self.entries:
            if self.redis_config is not None or self.entries[name].get('restore_enabled', True):
                try:
                    await self.start_entry(name, resume=False)
                except Exception as exc:
                    failures.append(exc)
                    logging.exception('Could not start Server entry %s', name)
        if failures and not any(entry['status'] == 'ready' for entry in self.entries.values()):
            raise failures[0]
        if self.redis_config is not None:
            self.shared_control = asyncio.create_task(self._shared_controls())
        self.task_maintenance = asyncio.create_task(self._maintain_tasks())
        from .callbacks import Callbacks
        self.callbacks = Callbacks(self.db, self.config_scope)
        self.callbacks.start()
        self.started = True

    @audited
    async def cleanup_tasks(self, limit=100):
        """Apply recorded retention policies once; useful for explicit maintenance."""
        from .errors import ServerError
        if self.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down')
        return await asyncio.shield(self.own(asyncio.to_thread(self.db.cleanup_tasks, limit)))

    @audited
    async def cleanup_editor_history(self, limit=100):
        from .errors import ServerError
        if self.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down')
        queue = getattr(self, 'queue', None)
        if queue is None:
            return 0
        return await asyncio.shield(self.own(asyncio.to_thread(
            queue.cleanup_history, self.editor_history_retention_seconds, limit)))

    @audited
    async def cleanup_files(self, limit=100):
        from .errors import ServerError
        if self.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down')
        return await asyncio.shield(self.own(asyncio.to_thread(self._cleanup_files, limit)))

    def _cleanup_files(self, limit=100):
        result = self.storage.cleanup(limit)
        result['assets'] = self._asset_cleanup.cleanup(limit)
        return result

    async def _assets_for(self, entry):
        # Entry identity matters: Reload copies its dict but owns a new Graph.
        # Retain registration paths across Stop/Start, including direct Graphs
        # whose resource fields were replaced with private local copies.
        if id(entry) not in self._asset_leases:
            from .assets import AssetLease
            lease = await asyncio.to_thread(AssetLease, self.data_dir, self.db)
            self._asset_leases[id(entry)] = (entry, lease)
        return self._asset_leases[id(entry)][1]

    def _release_assets(self, entry):
        if entry['workers']:
            return
        retained = self._asset_leases.pop(id(entry), None)
        if retained is not None:
            retained[1].close()

    @asynccontextmanager
    async def _publishing_assets(self, entry):
        retained = self._asset_leases.get(id(entry))
        if retained is None:
            yield
            return
        lease = retained[1]
        token = await asyncio.to_thread(lease.begin_publication)
        yield
        await asyncio.to_thread(lease.end_publication, token)

    async def _maintain_tasks(self, interval=60):
        while not self.shutting_down:
            deleted = 0
            files_deleted = 0
            controls_deleted = 0
            controls_reconciled = 0
            audit_deleted = 0
            history_deleted = 0
            try:
                queue = getattr(self, 'queue', None)
                if queue is not None:
                    history_deleted = await self.audit.background('cleanup_editor_history',
                        lambda: asyncio.to_thread(queue.cleanup_history, self.editor_history_retention_seconds))
            except Exception:
                logging.exception('Editor history cleanup failed; retrying at the next interval')
            try:
                deleted = await self.audit.background('cleanup_tasks', lambda: asyncio.to_thread(self.db.cleanup_tasks))
            except Exception:
                logging.exception('Task retention cleanup failed; retrying at the next interval')
            try:
                result = await self.audit.background('cleanup_files', lambda: asyncio.to_thread(self._cleanup_files))
                files_deleted = result['deleted']
            except Exception:
                logging.exception('File cleanup failed; retrying at the next interval')
            try:
                controls_reconciled = await self.audit.background('controls.reconcile',
                    lambda: self.controls.database(self.controls.reconcile))
                controls_deleted = await self.audit.background('controls.cleanup', lambda: asyncio.to_thread(self.controls.cleanup))
            except Exception:
                logging.exception('Control receipt cleanup failed; retrying at the next interval')
            try:
                audit_deleted = await self.audit.background('audit.cleanup', self.audit.cleanup)
            except Exception:
                logging.getLogger('nndeploy.server.audit').error('Audit retention cleanup failed; retrying at the next interval')
            if self.devices is not None:
                try:
                    await self.audit.background('devices.recover', lambda: asyncio.to_thread(self.devices.recover))
                except Exception:
                    logging.exception('Resident device ownership remains unconfirmed')
            try:
                # Catch up in bounded batches without busy-looping on a backlog.
                await asyncio.wait_for(self.maintenance_stop.wait(), timeout=min(interval, 1) if max(deleted, files_deleted, controls_deleted, controls_reconciled, audit_deleted, history_deleted) == 100 else interval)
            except asyncio.TimeoutError:
                pass

    def _entry(self, name):
        from .errors import ServerError
        try:
            return self.entries[name]
        except KeyError:
            raise ServerError(404, 'entry_not_found', 'Unknown Server entry') from None

    async def _control_entry(self, name, action, operation):
        from .errors import ServerError
        if self.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down')
        current = self.entry_controls.get(name)
        if current is not None and not current['task'].done():
            if current['action'] != action or action == 'reload':
                raise ServerError(409, 'entry_busy', 'Entry is changing state', retryable=True)
            return await asyncio.shield(current['task'])
        if self.retiring_entries.get(name) and action in ('start', 'reload'):
            raise ServerError(409, 'entry_busy', 'Previous workers have not confirmed release', retryable=True)
        task = self.own(operation())
        control = self.entry_controls[name] = dict(action=action, task=task)
        def done(_):
            if self.entry_controls.get(name) is control:
                self.entry_controls.pop(name, None)
        task.add_done_callback(done)
        return await asyncio.shield(task)

    async def _persist_entry(self, name, entry, enabled):
        if id(entry) not in self._asset_leases:
            return await self._save_entry(name, entry, enabled)
        async with self._publishing_assets(entry):
            return await self._save_entry(name, entry, enabled)

    async def _save_entry(self, name, entry, enabled):
        from .resources import save_entry
        from .errors import ServerError
        from sqlalchemy.exc import OperationalError, InterfaceError, DisconnectionError, TimeoutError as PoolTimeout
        if entry['source']['workflow'] is None and (entry['source'].get('graph_factory') is None or
                                                   entry['source'].get('resource_snapshot') is None):
            return
        if self.shared_runtime is not None and entry.get('shared_key') and entry['source']['workflow'] is not None:
            # The namespace's canonical config owns resources and restart state.
            # A workspace stores only discovery, never another stale snapshot.
            data = dict(config={'name': name}, shared_entry_key=entry['shared_key'])
            await self.shared_runtime.write(self.db.save_server_config, self.config_scope, name, data,
                                             write_token=uuid.uuid4().hex)
            return
        if 'saved_token' not in entry:
            saved = await asyncio.to_thread(self.db.get_server_config, self.config_scope, name)
            entry['saved_token'] = saved['write_token'] if saved else None
        token = uuid.uuid4().hex
        while True:
            if self.shutting_down:
                raise ServerError(503, 'configuration_unconfirmed', 'Configuration write is not confirmed; inspect saved state after recovery',
                                  retryable=True)
            try:
                entry['saved_token'] = await asyncio.to_thread(save_entry, self.db, self.config_scope, name,
                    entry['source'], entry['config'], enabled, checked=True, expected_token=entry['saved_token'], write_token=token)
                entry['error'] = None
                return
            except (OperationalError, InterfaceError, DisconnectionError, PoolTimeout, OSError):
                entry['error'] = 'Waiting for configuration database recovery'
                logging.warning('Configuration write awaits confirmation for %s', name)
                # Retrying the same conditional write resolves a lost commit
                # acknowledgement without trusting an inconclusive plain read.
                try:
                    await asyncio.wait_for(self.maintenance_stop.wait(), timeout=.5)
                except asyncio.TimeoutError:
                    pass

    @audited
    async def start_entry(self, name, *, resume=True):
        return await self._control_entry(name, 'start', lambda: self._start_entry(name, self._entry(name), resume=resume))

    async def _write_shared_control(self, entry, accepting, expected_token):
        from .errors import ServerError
        from sqlalchemy.exc import OperationalError, InterfaceError, DisconnectionError, TimeoutError as PoolTimeout
        token = uuid.uuid4().hex
        while not self.shutting_down:
            try:
                await asyncio.to_thread(self.db.set_queue_accepting, entry['queue'].key, accepting,
                                        expected_token=expected_token, write_token=token)
                entry['error'] = None
                return
            except (OperationalError, InterfaceError, DisconnectionError, PoolTimeout, OSError):
                entry['error'] = 'Waiting for shared control database recovery'
                logging.warning('Shared control write awaits confirmation for %s', entry['config'].name)
                try:
                    await asyncio.wait_for(self.maintenance_stop.wait(), timeout=.5)
                except asyncio.TimeoutError:
                    pass
        raise ServerError(503, 'shared_control_unconfirmed', 'Shared control write is not confirmed; inspect shared state after recovery', retryable=True)

    async def _prepare_shared_queue(self, name, entry, *, loaded=False, resume=True):
        await self.shared_runtime.prepare(name, entry, loaded=loaded, resume=resume)

    async def _release_entry_files(self, entry):
        entry['release_pending'] = True
        if self.devices is not None:
            await self.devices.release(entry)
        if self.shared_runtime is not None:
            await self.shared_runtime.release(entry)
        await asyncio.to_thread(self.db.release_files, entry['file_holder'])
        if entry.get('control_guard'):
            guard = entry['control_guard']
            guard.close()
            await asyncio.to_thread(self.shared_runtime.control.retire_runtime, guard.path.name)
            # Retrying a lost commit receipt must use the same runtime token.
            # Closing the physical guard is idempotent; forgetting its identity
            # before the database confirms retirement would strand cleanup.
            entry.pop('control_guard')
        entry.pop('release_pending', None)

    async def _start_entry(self, name, entry, *, persist=True, resume=True):
        from .worker import Worker, ProcessWorker, EngineWorker, ConcurrentWorker
        from .errors import ServerError
        if self.shutting_down:
            raise ServerError(503, 'server_stopping', 'Server is shutting down')
        if entry['status'] == 'ready':
            if entry.get('consumer_failed'):
                raise ServerError(503, 'consumer_failed', 'Stop the entry before restarting its failed consumer')
            if self.redis_config is not None and resume:
                await self.shared_runtime.check(entry)
                queue = entry['queue']
                usage = await asyncio.to_thread(self.db.queue_usage, queue.key)
                if not usage['accepting']:
                    await self._write_shared_control(entry, True, usage['control_token'])
            return self.describe(name)
        if entry['status'] == 'failed' and entry['workers']:
            raise ServerError(409, 'entry_busy', 'Previous worker release was not confirmed')
        if entry['status'] == 'failed' and entry.get('release_pending'):
            # Failed initialization may have closed every Worker while its
            # resource transaction failed. Preserve the old holder until that
            # cleanup succeeds; a new load must not orphan the old references.
            await self._release_entry_files(entry)
        if entry['status'] == 'failed' and entry.get('device_guard') and self.devices is not None:
            await self.devices.release(entry)
        if entry['status'] in ('starting', 'stopping', 'reloading'):
            raise ServerError(409, 'entry_busy', 'Entry is changing state', retryable=True)
        entry['status'], entry['error'] = 'starting', None
        config = entry['config']
        initial_config = config
        directory = self.data_dir / 'tmp' / name
        directory.mkdir(parents=True, exist_ok=True)
        entry['workers'] = []
        entry['consumers'], entry['consumer_runs'] = [], []
        entry['consumer_failed'] = False
        if not entry.get('control_guard'):
            entry['file_holder'] = uuid.uuid4().hex
        entry['shared_holders'] = []
        candidate = bool(entry.get('shared_candidate') or entry.get('shared_draining'))
        queue_prepared = False
        control_token = None
        prepared_fingerprint = None
        try:
            if (entry['source']['workflow'] is not None or entry['source'].get('graph_factory') is not None) and 'saved_token' not in entry:
                saved = await asyncio.to_thread(self.db.get_server_config, self.config_scope, name)
                entry['saved_token'] = saved['write_token'] if saved else None
                if saved and saved['data'].get('kind') == 'graph_factory' and entry['source'].get('graph_factory') is not None:
                    entry['source']['resource_fallback'] = saved['data']['resources']
            if self.redis_config is not None and (not entry['source']['autodetect'] or entry.get('queue') is not None):
                # A known stopped shared configuration must not initialize or
                # execute warmup just because another API process restarted.
                await self._prepare_shared_queue(name, entry, resume=resume)
                candidate = bool(entry.get('shared_candidate') or entry.get('shared_draining'))
                config = initial_config = entry['config']
                queue_prepared = True
                usage = await asyncio.to_thread(self.db.queue_usage, entry['queue'].key)
                control_token = usage['control_token']
                from .redis import entry_fingerprint
                prepared_fingerprint = entry_fingerprint(entry['source'], config)
                if not candidate and not resume and not usage['accepting']:
                    entry['status'] = 'stopped'
                    return self.describe(name)
                if not candidate and resume and not usage['accepting'] and usage['inflight']:
                    raise ServerError(409, 'entry_busy', 'Shared entry is still draining', retryable=True)
            if self.shared_runtime is not None and not entry.get('control_guard'):
                await self.shared_runtime.guard(entry, uuid.uuid4().hex)
            if self.devices is not None:
                await self.devices.acquire(name, entry)
            elif config.execution.devices:
                raise ServerError(422, 'device_limits_required', 'Configure Server device_limits to enforce declared resident budgets')
            await asyncio.to_thread(self.db.hold_files, entry['file_holder'],
                                    [entry['source']['workflow'], config.model_dump()])
            if entry['source']['workflow'] is not None:
                from .resources import prepare_workflow
                entry['source']['workflow'] = await asyncio.to_thread(prepare_workflow,
                    entry['source']['workflow'], config, self.data_dir, entry['source']['base_dir'],
                    lease=await self._assets_for(entry))
            for _ in range(config.execution.workers):
                worker_class = {'engine': EngineWorker, 'concurrent': ConcurrentWorker}.get(
                    config.execution.scheduling, ProcessWorker if config.execution.isolation == 'process' else Worker)
                if config.execution.batching.enabled:
                    from .batching import BatchWorker
                    worker_class = BatchWorker
                # Existing editor plugins are trusted installation code. Each
                # isolated Worker must register them before loading the Graph.
                # Derive this locally; it is not a workflow field or HTTP input.
                plugin_root = Path(self.args.resources) if getattr(self, 'args', None) is not None else self.data_dir
                worker_source = dict(entry['source'], plugin_dir=str(plugin_root / 'plugin'), resource_dir=str(self.data_dir),
                                     resource_holder=entry['file_holder'])
                worker = worker_class(worker_source, config, self.storage, directory)
                if entry.get('control_guard'):
                    worker.control_guard_path = str(entry['control_guard'].path)
                if entry.get('device_guard'):
                    worker.device_guard_path = str(entry['device_guard'].path)
                    worker.device_run = lambda values, operation, entry=entry: self.devices.run(entry, values, operation)
                    if config.execution.temporary_memory_bytes:
                        worker.device_prepare = worker.device_run
                entry['workers'].append(worker)
                holder = await self.shared_runtime.hold(entry) if self.shared_runtime is not None else None
                worker.shared_holder = holder
                with self.metrics.measure('worker_initialize', name):
                    if self.devices is not None:
                        await self.devices.initialize(entry, worker)
                    else:
                        await worker.start()
                if getattr(worker, 'asset_paths', None):
                    lease = await self._assets_for(entry)
                    await asyncio.to_thread(lease.hold_values, worker.asset_paths)
                if entry['source'].get('graph_factory') is not None:
                    entry['source']['resource_snapshot'] = worker.resource_snapshot
                if holder is not None:
                    await self.shared_runtime.hold(entry, holder=holder, ready=True)
                if entry['source']['autodetect']:
                    config = worker.config
                    entry['config'], entry['schema'] = config, config.public_schema()
            await asyncio.to_thread(self.db.hold_files, entry['file_holder'], config.model_dump())
            if self.redis_config is not None:
                from .redis import entry_fingerprint
                if queue_prepared and prepared_fingerprint != entry_fingerprint(entry['source'], config):
                    entry['config'], entry['schema'] = initial_config, initial_config.public_schema()
                    raise ServerError(409, 'shared_configuration_changed', 'Discovered workflow configuration changed; register its explicit configuration before starting')
                if not queue_prepared:
                    await self._prepare_shared_queue(name, entry, loaded=True, resume=resume)
                    candidate = bool(entry.get('shared_candidate') or entry.get('shared_draining'))
                    for worker in entry['workers']:
                        worker.shared_holder = await self.shared_runtime.hold(entry, ready=True)
                if resume and not candidate:
                    usage = await asyncio.to_thread(self.db.queue_usage, entry['queue'].key)
                    if control_token is None:
                        # First-time factory discovery cannot observe this queue
                        # before loading its schema. Join an active queue, but
                        # never resume a stopped queue from that late snapshot.
                        if not usage['accepting']:
                            raise ServerError(409, 'shared_control_changed', 'Shared entry is stopped; retry Start after schema discovery', retryable=True)
                    else:
                        await self._write_shared_control(entry, True, control_token)
                usage = await asyncio.to_thread(self.db.queue_usage, entry['queue'].key)
                if not candidate and not usage['accepting']:
                    await asyncio.gather(*(worker.close() for worker in entry['workers']))
                    await self._release_entry_files(entry)
                    entry['workers'], entry['status'] = [], 'stopped'
                    return self.describe(name)
            if self.shared_runtime is not None:
                await self.shared_runtime.started(entry)
            if persist:
                await self._persist_entry(name, entry, True)
            if self.redis_config is None:
                # Deploy history lives in DB with its retention policy. Keep
                # the editor's legacy in-memory history only on its own queue.
                entry['queue'] = TaskQueue(capacity=config.execution.capacity, history_limit=0)
            entry['wake'] = asyncio.Event()
            entry['futures'] = {}
            entry['closing'] = None
            entry['status'] = 'ready'
            entry['local_stop'] = False
            consume = self._consume if self.redis_config is None else self._consume_shared
            # Execution/claim/finalization remain owned even if an internal
            # consumer waiter is cancelled before or during its first await.
            entry['consumer_runs'] = [asyncio.create_task(consume(entry, worker)) for worker in entry['workers']
                                      for _ in range(config.execution.request_slots)]
            for run in entry['consumer_runs']:
                run.add_done_callback(lambda task, entry=entry: self._consumer_finished(entry, task))
            entry['consumers'] = [asyncio.create_task(self._join_consumer(run)) for run in entry['consumer_runs']]
            if self.redis_config is not None:
                entry['maintenance'] = asyncio.create_task(self._maintain_shared(entry))
        except Exception as exc:
            closed = await asyncio.gather(*(worker.close() for worker in entry['workers']), return_exceptions=True)
            if not any(isinstance(result, BaseException) for result in closed):
                try:
                    await self._release_entry_files(entry)
                    if self.shared_runtime is not None:
                        await self.shared_runtime.failed(entry)
                except Exception:
                    logging.exception('Could not release failed entry resource references')
            if not any(isinstance(result, BaseException) for result in closed):
                entry['workers'] = []
            superseded = isinstance(exc, ServerError) and exc.code in ('shared_control_changed', 'entry_busy') and not entry['workers']
            entry['status'], entry['error'] = 'stopped' if superseded else 'failed', str(exc)
            if self.redis_config is not None and entry.get('queue') is not None:
                await entry['queue'].close()
            raise
        return self.describe(name)

    @audited
    async def reload_entry(self, name, *, workflow=None, config=None):
        return await self._control_entry(name, 'reload', lambda: self._reload_entry(name, workflow=workflow, config=config))

    async def _reload_entry(self, name, *, workflow=None, config=None, start_stopped=False):
        from contextlib import nullcontext
        from .config import ServerConfig, load_json
        value = self._entry(name)['config'] if config is None else ServerConfig.model_validate(
            load_json(config) if isinstance(config, (str, Path)) else config)
        with self.mcp.reserve(name, value.mcp) if self.mcp is not None else nullcontext():
            return await self._reload_entry_impl(name, workflow=workflow, config=config, start_stopped=start_stopped)

    async def _reload_entry_impl(self, name, *, workflow=None, config=None, start_stopped=False):
        """Prepare replacement workers before changing the accepting entry."""
        import copy
        from .config import ServerConfig, load_json
        from .errors import ServerError
        old = self._entry(name)
        stopped_update = start_stopped and self.shared_runtime is not None and old['status'] in ('stopped', 'failed') and not old['workers']
        failed_update = self.shared_runtime is not None and old.get('shared_key') and old['status'] == 'failed' and not old['workers']
        if (old['status'] != 'ready' and not stopped_update and not failed_update) or self.shutting_down:
            raise ServerError(409, 'entry_busy', 'Entry must be ready before reload', retryable=True)
        if old['source']['graph'] is not None:
            raise ServerError(409, 'reload_unsupported', 'Reload requires a workflow or a factory')
        source = dict(old['source'])
        if workflow is not None:
            if source['workflow'] is None:
                raise ServerError(422, 'invalid_request', 'This entry does not load a workflow')
            source['workflow'] = load_json(workflow) if isinstance(workflow, (str, Path)) else copy.deepcopy(workflow)
            if self.redis_config is not None:
                from .redis import source_identity
                source['identity'] = source_identity(source)
        value = old['config']
        if config is not None:
            if isinstance(config, (str, Path)):
                config = load_json(config)
            value = ServerConfig.model_validate(config)
            value = value.model_copy(update={'name': name})
            source['autodetect'] = False
            source['untyped'] = False
        candidate = dict(old, source=source, config=value, schema=value.public_schema(),
                         untyped=source['untyped'], status='stopped', error=None, workers=[], consumers=[], futures={}, admitted=set(), closing=None)
        candidate.pop('device_guard', None)
        if self.shared_runtime is not None:
            return await self.shared_runtime.reload(name, old, candidate)
        old['status'] = 'reloading'
        try:
            await self._start_entry(name, candidate, persist=False)
            await self._persist_entry(name, candidate, True)
        except BaseException:
            try:
                if candidate['status'] == 'ready':
                    await self._stop_entry(name, candidate, persist=False)
            finally:
                old['status'] = 'ready'
                if candidate['workers'] or candidate.get('release_pending'):
                    self.retiring_entries.setdefault(name, []).append(candidate)
                else:
                    self._release_assets(candidate)
            raise
        # There is no await between the pointer change and stopping old admission.
        self.entries[name] = candidate
        candidate['status'] = 'reloading'
        old['status'] = 'ready'
        self.retiring_entries.setdefault(name, []).append(old)
        try:
            await self._stop_entry(name, old, persist=False)
        except BaseException:
            candidate['error'] = 'Previous worker release was not confirmed'
            raise
        finally:
            candidate['status'] = 'ready'
        self.retiring_entries[name].remove(old)
        self._release_assets(old)
        return self.describe(name)

    def accepting(self, name, entry=None):
        entry = self._entry(name) if entry is None else entry
        control = self.entry_controls.get(name)
        stopping = control is not None and control['action'] == 'stop' and not control['task'].done()
        return not self.shutting_down and not stopping and not entry.get('consumer_failed') and entry['status'] in ('ready', 'reloading')

    def describe(self, name):
        entry = self._entry(name)
        control = self.entry_controls.get(name)
        stopping = control is not None and control['action'] == 'stop' and not control['task'].done()
        state = 'failed' if entry.get('consumer_failed') and entry['status'] not in ('stopping', 'stopped') else entry['status']
        return dict(name=name, mcp=entry['config'].mcp.model_dump() if entry['config'].mcp else None,
                    mcp_path=self.mcp.config.path if self.mcp is not None else None, status='stopping' if stopping and entry['status'] == 'ready' else state, error=entry['error'],
                    schema=entry['schema'], workers=len(entry['workers']),
                    draining_workers=sum(len(previous['workers']) for previous in self.retiring_entries.get(name, [])),
                    inflight=len(entry['futures']), deploy_path=f'/v1/server/{name}/deploy',
                    isolation=entry['config'].execution.isolation, streaming=entry['config'].streaming.model_dump(),
                    scheduling=entry['config'].execution.scheduling, max_concurrency_per_worker=entry['config'].execution.max_concurrency_per_worker,
                    task_retention_seconds=entry['config'].execution.task_retention_seconds,
                    http_wait_timeout_seconds=entry['config'].execution.http_wait_timeout_seconds,
                    batching=entry['config'].execution.batching.model_dump(),
                    devices=entry['config'].execution.devices, memory_bytes=entry['config'].execution.memory_bytes,
                    temporary_memory_bytes=entry['config'].execution.temporary_memory_bytes,
                    cpu_threads=entry['config'].execution.cpu_threads)

    async def _check_output_files(self, entry, outputs):
        fields = [(name, field) for name, field in entry['config'].outputs.items() if field.is_file()]
        if not fields:
            return
        def check():
            for name, field in fields:
                record = self.db.get_file(outputs[name]['file_id'])
                if record is None or record['state'] != 'ready':
                    raise ValueError('Output file is unavailable')
                field.check_file(record)
                if field.type in ('array', 'tensor'):
                    from .arrays import read_file
                    with self.storage.open(outputs[name]['file_id']) as stream:
                        read_file(stream, field, load=False)
        await asyncio.to_thread(check)

    def _event_writer(self, entry, record):
        limits = entry['config'].streaming
        if not limits.enabled:
            return None
        def emit(value):
            sequence = self.db.append_task_event(record['task_id'], value, attempt=record['attempt'],
                worker_token=record['worker_token'], max_events=limits.max_events, max_bytes=limits.max_event_bytes,
                allow_cancel_requested=entry['config'].execution.scheduling != 'serial')
            if self.event_notifications is not None:
                self.event_notifications.notify(record['task_id'], sequence)
            return sequence
        return emit

    def own(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.control_tasks.add(task)
        def finished(done):
            self.control_tasks.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return task

    @audited
    async def submit(self, name, inputs, parameters=None, *, idempotency_key=None, owner=None, callback_url=None):
        # Admission is owned by Server even if the HTTP request disconnects
        # during the database write.
        identity, future = await asyncio.shield(self.own(self._submit(name, inputs, parameters, idempotency_key=idempotency_key, owner=owner, callback_url=callback_url)))
        observer = self._observe_result(future)
        observer.dispatch_pending = getattr(future, 'dispatch_pending', False)
        return identity, observer

    @staticmethod
    def _observe_result(future):
        """A detachable reader; cancelling it never reaches the execution future.

        Do not attach the HTTP task to the long-lived future's awaited-by graph:
        some asyncio versions retain cancelled shield/wait callers until that
        future finishes. Explicit callback removal also bounds repeated retries.
        """
        observer = asyncio.get_running_loop().create_future()
        def completed(source):
            if observer.done():
                return
            if source.cancelled():
                observer.cancel()
            elif (error := source.exception()) is not None:
                observer.set_exception(error)
            else:
                observer.set_result(source.result())
        def detached(done):
            future.remove_done_callback(completed)
            if not done.cancelled():
                done.exception()
        observer.add_done_callback(detached)
        if future.done():
            completed(future)
        else:
            future.add_done_callback(completed)
        return observer

    @observe_admission
    async def _submit(self, name, inputs, parameters=None, *, idempotency_key=None, owner=None, expected_entry=None, callback_url=None):
        from .errors import ServerError
        import queue
        import time
        from .logging_taskid import get_request_id
        request_id = get_request_id()
        entry = self._entry(name)
        control = self.entry_controls.get(name)
        if control is not None and control['action'] == 'stop' and not control['task'].done():
            raise ServerError(503, 'not_ready', 'Entry is stopping', retryable=True)
        if expected_entry is not None and entry is not expected_entry:
            raise ServerError(409, 'entry_changed', 'Entry changed during batch admission; retry unaccepted items')
        if not self.accepting(name, entry):
            raise ServerError(503, 'not_ready', 'Entry is not accepting requests', retryable=True)
        if self.shared_runtime is not None:
            await self.shared_runtime.check(entry)
        try:
            from .config import validate_request
            from .callbacks import validate_url
            callback_url = validate_url(callback_url)
            values = validate_request(entry['config'], inputs, parameters, input_model=entry['source']['input_model'],
                                      untyped=entry['untyped'])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ServerError(422, 'invalid_request', str(exc)) from exc
        for group in ('inputs', 'parameters'):
            for field_name, field in getattr(entry['config'], group).items():
                if field.is_file():
                    reference = values[group][field_name]
                    if 'path' in reference:
                        _, record = await asyncio.to_thread(self.storage.resolve_input, reference, field)
                    else:
                        record = await asyncio.to_thread(self.db.get_file, reference['file_id'])
                        if record is None or record['owner'] != owner:
                            raise ServerError(422, 'invalid_file', 'Referenced file does not exist')
                    try:
                        field.check_file(record)
                        if field.type in ('array', 'tensor'):
                            def validate_array_file():
                                from .arrays import read_file
                                path, _ = self.storage.resolve_input(reference, field)
                                with path.open('rb') as stream:
                                    read_file(stream, field, load=False)
                            await asyncio.to_thread(validate_array_file)
                    except (ValueError, TypeError, OverflowError, EOFError) as exc:
                        raise ServerError(422, 'invalid_file', str(exc)) from exc
        if idempotency_key is not None:
            if not idempotency_key or len(idempotency_key) > 128:
                raise ServerError(422, 'invalid_request', 'Idempotency key must contain 1 to 128 characters')
            if self.redis_config is not None:
                lookup_key = entry['queue']._idempotency_key(idempotency_key, owner)
            else:
                if owner is not None:
                    import hashlib
                    idempotency_key = hashlib.sha256(json.dumps([owner, idempotency_key]).encode()).hexdigest()
                lookup_key = idempotency_key
            record = await asyncio.to_thread(self.db.find_task, name, lookup_key, values['inputs'], values['parameters'], owner, callback_url)
            if record:
                return self._record_future(record)
            if not self.accepting(name, entry):
                raise ServerError(503, 'not_ready', 'Entry is not accepting requests', retryable=True)
        if not self.accepting(name, entry):
            raise ServerError(503, 'not_ready', 'Entry is not accepting requests', retryable=True)
        if self.inflight >= self.max_inflight_tasks:
            raise ServerError(429, 'server_full', 'Server task limit reached', retryable=True)
        if self.redis_config is not None:
            self.inflight += 1
            try:
                accepted = await entry['queue'].put(values['inputs'], values['parameters'],
                    queue_timeout=entry['config'].execution.queue_timeout_seconds, idempotency_key=idempotency_key, owner=owner,
                    retention_seconds=entry['config'].execution.task_retention_seconds, request_id=request_id,
                    max_tasks_per_subject=self.authentication.max_inflight_tasks_per_subject,
                    input_state=entry['config'].streaming.input_state(), callback_url=callback_url, callback_scope=self.config_scope)
            finally:
                self.inflight -= 1
            identity, future = self._record_future(dict(task_id=accepted['task_id'], state='pending', queue_key=entry['queue'].key))
            future.dispatch_pending = accepted['dispatch_pending']
            return identity, future
        identity = 'task_' + uuid.uuid4().hex
        values['_owner'] = owner
        values['_request_id'] = request_id
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        # Reserve queue capacity without yielding. Persistence happens before the
        # consumer wake-up, so failed recording can never execute native work.
        try:
            entry['queue'].put(dict(id=identity, values=values, queued_at=time.monotonic(), admitted=False))
        except queue.Full as exc:
            raise ServerError(429, 'queue_full', 'Entry queue is full', retryable=True) from exc
        self.inflight += 1
        entry['futures'][identity] = future
        self.task_futures[identity] = future
        try:
            # SQLite transaction is short; offload and gate dispatch until committed.
            accepted = await asyncio.to_thread(self.db.create_task, identity, name, time.time(),
                values['inputs'], values['parameters'], owner=owner, idempotency_key=idempotency_key,
                retention_seconds=entry['config'].execution.task_retention_seconds, request_id=request_id,
                max_tasks_per_subject=self.authentication.max_inflight_tasks_per_subject,
                input_state=entry['config'].streaming.input_state(), callback_url=callback_url, callback_scope=self.config_scope)
        except Exception as exc:
            entry['queue'].cancel_pending(identity)
            entry['futures'].pop(identity, None)
            self.task_futures.pop(identity, None)
            self.inflight -= 1
            entry['wake'].set()
            if isinstance(exc, ServerError):
                raise
            raise ServerError(503, 'recording_unavailable', 'Could not record the task', retryable=True) from exc
        if accepted != identity:
            entry['queue'].cancel_pending(identity)
            entry['futures'].pop(identity, None)
            self.task_futures.pop(identity, None)
            self.inflight -= 1
            entry['wake'].set()
            record = await asyncio.to_thread(self.db.get_task, accepted)
            return self._record_future(record)
        entry.setdefault('admitted', set()).add(identity)
        entry['wake'].set()
        return identity, future

    @audited
    async def send_input(self, task_id, data=None, *, sequence, end=False, owner=None):
        """Append one ordered duplex input; retry the same sequence after uncertainty."""
        if self.shutting_down:
            from .errors import ServerError
            raise ServerError(503, 'not_ready', 'Server is shutting down', retryable=True)
        return await asyncio.shield(self.own(asyncio.to_thread(self.db.append_task_input, task_id,
            dict(sequence=sequence, data=data, end=end), owner)))

    @audited
    async def deploy_batch(self, name, items, *, mode='sync', owner=None, wait_timeout_seconds=None):
        from .protocol import BatchRequest
        from .errors import ServerError
        try:
            request = BatchRequest(items=items, mode=mode, wait_timeout_seconds=wait_timeout_seconds)
        except ValueError as exc:
            raise ServerError(422, 'invalid_batch', 'Batch requires 1 to 64 items, sync or async mode, and a valid sync wait timeout') from exc
        if request.wait_timeout_seconds is not None:
            self.wait_timeout(name, request.wait_timeout_seconds)
        # Complete durable admission even if the caller disappears. Result
        # waiting belongs to that caller, not to the Server lifecycle.
        results, pending = await asyncio.shield(self.own(self._admit_batch(name, request, owner)))
        if request.mode == 'sync':
            observers = [(index, self._observe_result(future)) for index, future in pending]
            try:
                if observers:
                    # One window after admission; only the readers are awaited.
                    await asyncio.wait([future for _, future in observers], timeout=request.wait_timeout_seconds)
                for index, future in observers:
                    try:
                        if not future.done():
                            raise self._wait_error(results[index]['task_id'])
                        value = future.result()
                        results[index].update(status_code=200, outputs=value['outputs'])
                    except Exception as exc:
                        error = exc if isinstance(exc, ServerError) else ServerError(500, 'execution_failed', 'Task execution failed')
                        results[index].update(status_code=error.status_code, **error.response())
            finally:
                for _, future in observers:
                    future.cancel()
        return dict(items=results)

    async def _admit_batch(self, name, request, owner):
        from .errors import ServerError
        entry = self._entry(name)
        results, pending = [], []
        unavailable = None
        for index, item in enumerate(request.items):
            try:
                if unavailable is not None:
                    raise unavailable
                identity, future = await self._submit(name, item.inputs, item.parameters, owner=owner,
                    idempotency_key=item.idempotency_key, expected_entry=entry, callback_url=item.callback_url)
            except Exception as exc:
                error = exc if isinstance(exc, ServerError) else ServerError(503, 'recording_unavailable', 'Could not record the task', retryable=True)
                if error.code == 'recording_unavailable':
                    unavailable = error
                results.append(dict(index=index, accepted=False, status_code=error.status_code, **error.response()))
            else:
                results.append(dict(index=index, accepted=True, status_code=202, task_id=identity,
                    status_url=f'/v1/tasks/{identity}', dispatch_pending=getattr(future, 'dispatch_pending', False)))
                pending.append((index, future))
        return results, pending

    def _record_future(self, record):
        from .errors import ServerError
        if record is None:
            raise ServerError(404, 'task_not_found', 'Task record is no longer available')
        identity = record['task_id']
        future = self.task_futures.get(identity)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            if record['state'] == 'succeeded':
                future.set_result(dict(task_id=identity, outputs=record['outputs']))
            elif record['queue_key'] and record['state'] in ('pending', 'running', 'dispatched') and self.redis_config is not None:
                self.task_futures[identity] = future
                self.inflight += 1
                self.waiters[identity] = asyncio.create_task(self._wait_shared(identity, future))
            else:
                future.set_exception(ServerError.from_task(record))
        return identity, future

    async def _wait_shared(self, identity, future):
        from .errors import ServerError
        try:
            while True:
                try:
                    record = await asyncio.to_thread(self.db.get_task, identity)
                    if record is None:
                        if not future.done():
                            future.set_exception(ServerError(404, 'task_not_found', 'Task record is no longer available',
                                                             task_id=identity))
                        return
                    if record and record['state'] in ('succeeded', 'failed', 'cancelled', 'expired'):
                        if not future.done():
                            if record['state'] == 'succeeded':
                                future.set_result(dict(task_id=identity, outputs=record['outputs']))
                            else:
                                future.set_exception(ServerError.from_task(record))
                        return
                except Exception:
                    logging.warning('Waiting for shared task database recovery: %s', identity)
                await asyncio.sleep(.2)
        finally:
            if not future.done():
                future.set_exception(ServerError(503, 'server_stopping', 'Task remains recorded; query its status after reconnecting',
                                                 task_id=identity, retryable=True))
            self.task_futures.pop(identity, None)
            self.waiters.pop(identity, None)
            self.inflight -= 1

    async def _shared_controls(self):
        while not self.shutting_down:
            for name, entry in list(self.entries.items()):
                queue = entry.get('queue')
                if queue is None or entry['status'] not in ('ready', 'stopped') or name in self.entry_controls:
                    continue
                try:
                    # Read admission before the active configuration. Reload
                    # retires the old queue and changes the active pointer in
                    # one transaction. Reversing these observations can mistake
                    # retirement for Stop and wait for old work before following
                    # the replacement configuration.
                    usage = await asyncio.to_thread(self.db.queue_usage, queue.key)
                    if entry.get('shared_key'):
                        active = await self.shared_runtime.status(entry)
                        if active['active_queue'] != queue.key:
                            active = await self.shared_runtime.current(entry)
                            self.own(self._follow_shared_configuration(name, entry, active))
                            continue
                        await self.shared_runtime.cleanup(entry)
                    import time
                    entry['metrics_shared_usage'] = (time.monotonic(), dict(usage))
                    if usage['accepting'] and entry['status'] == 'stopped':
                        self.own(self._follow_shared_control(name, entry, True))
                    elif not usage['accepting'] and entry['status'] == 'ready':
                        self.own(self._follow_shared_control(name, entry, False))
                    elif (entry.get('shared_key') and not self.retiring_entries.get(name)
                          and time.monotonic() >= entry.get('shared_recovery_after', 0)):
                        states = await self.shared_runtime.recoverable(entry)
                        if states:
                            self.own(self._recover_shared_configurations(name, entry, states))
                except Exception:
                    logging.exception('Shared entry control check failed: %s', name)
            await asyncio.sleep(.2)

    async def _recover_shared_configurations(self, name, entry, states):
        try:
            await self._control_entry(name, 'recover', lambda: self.shared_runtime.recover(name, entry, states))
        except Exception:
            import time
            entry['shared_recovery_after'] = time.monotonic() + 1
            entry['error'] = 'Old queued tasks could not be resumed'
            logging.exception('Shared queued-task recovery failed: %s', name)

    async def _follow_shared_configuration(self, name, entry, state):
        try:
            await self._control_entry(name, 'reload', lambda: self.shared_runtime.follow(name, entry, state))
        except Exception:
            if self.entries.get(name) is entry:
                entry['error'] = 'Active shared configuration could not be initialized'
            logging.exception('Shared configuration application failed: %s', name)

    async def _follow_shared_control(self, name, entry, accepting):
        try:
            if self.entries.get(name) is not entry:
                return
            if accepting:
                await self.start_entry(name, resume=False)
            else:
                await self._control_entry(name, 'stop', lambda: self._stop_entry(
                    name, entry, persist=False, shared_scope=False, drain_shared=True))
        except Exception:
            logging.exception('Shared entry control application failed: %s', name)

    def _consumer_finished(self, entry, run):
        if run.cancelled() or run.exception() is not None or entry['status'] not in ('stopping', 'stopped'):
            entry['consumer_failed'] = True
            entry['error'] = 'Task consumer exited unexpectedly'
            entry['wake'].set()

    async def _join_consumer(self, run):
        while True:
            try:
                return await asyncio.shield(run)
            except asyncio.CancelledError:
                if run.done():
                    return run.result()
                # Stop/Reload changes the entry's admission and drain state.
                # Task cancellation here must not cancel a durable claim or
                # abandon native work and its result transaction.
                continue

    async def _shared_heartbeat(self, queue, record, worker):
        while True:
            await asyncio.sleep(max(.2, queue.config.lease_seconds / 3))
            try:
                current = await asyncio.to_thread(self.db.get_task, record['task_id'])
                if current and current['cancel_requested'] and hasattr(worker, 'cancel'):
                    worker.cancel(record['task_id'])
                if not await queue.heartbeat(record):
                    if hasattr(worker, 'cancel'):
                        worker.cancel(record['task_id'])
                    return
            except Exception:
                # Losing a heartbeat never frees its DB slot or starts a replacement.
                logging.warning('Shared task heartbeat unavailable: %s', record['task_id'])

    async def _consume_shared(self, entry, worker):
        from .errors import ServerError
        queue = entry['queue']
        token = uuid.uuid4().hex
        while True:
            try:
                if entry['status'] == 'stopping':
                    if entry.get('local_stop'):
                        return
                    usage = await asyncio.to_thread(self.db.queue_usage, queue.key)
                    if usage['inflight'] == 0 or usage['accepting']:
                        return
                    if (entry.get('shared_draining') or entry.get('consumer_failed')) and not await asyncio.to_thread(self.db.pending_tasks, queue.key, 1):
                        # Other processes may still own running work. Do not
                        # reclaim it or keep an idle recovery model resident.
                        # An abandoned claim from a failed local consumer is
                        # reconciled only after these Workers have all exited.
                        return
                record = await queue.get(token, holder=worker.shared_holder)
            except Exception:
                logging.warning('Shared queue unavailable for %s', entry['config'].name)
                await asyncio.sleep(.5)
                continue
            if record is None:
                continue
            identity = record['task_id']
            heartbeat = asyncio.create_task(self._shared_heartbeat(queue, record, worker))
            self.running_workers[identity] = worker
            outputs, error, state = None, None, 'succeeded'
            try:
                options = {}
                async def before_run():
                    import time
                    checked_at = time.monotonic()
                    remaining = await asyncio.to_thread(self.db.check_shared_execution, identity, record['worker_token'], record['attempt'])
                    return checked_at + remaining
                limited = self.devices is not None and bool(self.devices.execution_devices(entry))
                values = dict(inputs=record['inputs'], parameters=record['parameters'], _owner=record['owner'], _task_id=identity,
                              _attempt=record['attempt'], _worker_token=record['worker_token'], _request_id=record['request_id'])
                if limited:
                    values['_queue_deadline'] = await before_run()
                if entry['config'].execution.batching.enabled:
                    options['before_run'] = before_run
                self.metrics.queue_wait.labels(entry['config'].name).observe(max(0, record['started_at'] - record['created_at']))
                from .request import worker_log_context
                with worker_log_context(record, worker), self.metrics.measure('worker_request', entry['config'].name):
                    async def execute():
                        if limited and not entry['config'].execution.batching.enabled:
                            await before_run()
                        return await worker.deploy(values, self._event_writer(entry, record), **options)
                    if limited and not entry['config'].execution.batching.enabled:
                        outputs = await self.devices.run(entry, [values], execute)
                    else:
                        outputs = await execute()
                await self._check_output_files(entry, outputs)
            except Exception as exc:
                failure = exc if isinstance(exc, ServerError) else ServerError(500, 'execution_failed', 'Shared worker execution failed')
                failure.task_id = identity
                error = failure.response()['error']
                state = 'cancelled' if failure.code == 'task_cancelled' else 'expired' if failure.code in ('execution_timeout', 'queue_timeout') else 'failed'
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
                self.running_workers.pop(identity, None)
            journal = self.data_dir / 'queue-results' / queue.key / f'{identity}-{record["worker_token"]}.json'
            result = dict(record=record, outputs=outputs, error=error, state=state)
            try:
                await asyncio.to_thread(self._write_json_replace, journal, result)
                await queue.finish(record, outputs=outputs, error=error, state=state)
                journal.unlink(missing_ok=True)
            except Exception:
                # The durable result is replayed by maintenance. No successful
                # response or capacity release is invented while DB/Redis is down.
                logging.exception('Shared result awaits reconciliation: %s', identity)

    async def _reconcile_shared_results(self, entry):
        from .config import load_json
        queue = entry['queue']
        directory = self.data_dir / 'queue-results' / queue.key
        for path in list(directory.glob('*.json'))[:100]:
            value = await asyncio.to_thread(load_json, path)
            try:
                await queue.finish(value['record'], outputs=value['outputs'], error=value['error'], state=value['state'])
            except RuntimeError:
                path.replace(path.with_suffix('.stale'))
                logging.error('Rejected stale shared result: %s', path.name)
            else:
                path.unlink(missing_ok=True)

    async def _maintain_shared(self, entry):
        queue = entry['queue']
        while True:
            try:
                await self._reconcile_shared_results(entry)
                await queue.reconcile()
            except Exception:
                logging.warning('Shared queue reconciliation unavailable for %s', entry['config'].name)
            await asyncio.sleep(.5)

    async def _consume(self, entry, worker):
        from .errors import ServerError
        from .task_queue import ExecutionStatus
        import time
        while True:
            item = entry['queue'].get(timeout=0)
            if item is None:
                if entry['status'] == 'stopping' and (not entry['futures'] or entry.get('consumer_failed')):
                    return
                entry['wake'].clear()
                await entry['wake'].wait()
                continue
            index, payload = item
            identity = payload['id']
            # A consumer already awake may have dequeued a still-persisting task.
            while identity not in entry.get('admitted', set()):
                if identity not in entry['futures']:
                    entry['queue'].task_done(index, ExecutionStatus(False), {}, {})
                    break
                await asyncio.sleep(0.001)
            if identity not in entry['futures']:
                continue
            outputs = None
            error = None
            try:
                record = await asyncio.to_thread(self.db.get_task, identity)
                if record['cancel_requested']:
                    raise ServerError(409, 'task_cancelled', 'Task was cancelled before execution', task_id=identity)
                if time.monotonic() - payload['queued_at'] > entry['config'].execution.queue_timeout_seconds:
                    raise ServerError(504, 'queue_timeout', 'Task expired before execution', task_id=identity)
                self.running_workers[identity] = worker
                async def before_run():
                    if time.monotonic() - payload['queued_at'] > entry['config'].execution.queue_timeout_seconds:
                        raise ServerError(504, 'queue_timeout', 'Task expired before execution', task_id=identity)
                    await asyncio.to_thread(self.db.start_task, identity, time.time())
                    entry['queue'].mark_started(identity, worker.pid)
                    self.metrics.queue_wait.labels(entry['config'].name).observe(max(0, time.monotonic() - payload['queued_at']))
                    return payload['queued_at'] + entry['config'].execution.queue_timeout_seconds
                options = {}
                if entry['config'].execution.batching.enabled:
                    options['before_run'] = before_run
                values = dict(payload['values'], _task_id=identity,
                              _queue_deadline=payload['queued_at'] + entry['config'].execution.queue_timeout_seconds)
                from .request import worker_log_context
                with worker_log_context(record, worker), self.metrics.measure('worker_request', entry['config'].name):
                    async def execute():
                        if not entry['config'].execution.batching.enabled:
                            await before_run()
                        return await worker.deploy(values, self._event_writer(entry, record), **options)
                    if self.devices is not None and not entry['config'].execution.batching.enabled:
                        outputs = await self.devices.run(entry, [values], execute)
                    else:
                        outputs = await execute()
                await self._check_output_files(entry, outputs)
                await asyncio.to_thread(self.db.finish_task, identity, 'succeeded', time.time(), outputs)
            except Exception as exc:
                error = exc if isinstance(exc, ServerError) else ServerError(500, 'execution_failed', 'Execution or result recording failed', task_id=identity)
                if error.task_id is None:
                    error.task_id = identity
                logging.exception('Task failed: %s', identity)
                try:
                    await asyncio.to_thread(self.db.finish_task, identity,
                        'expired' if error.code in ('queue_timeout', 'execution_timeout') else 'cancelled' if error.code == 'task_cancelled' else 'failed',
                        time.time(), error=error.response()['error'])
                except Exception:
                    logging.exception('Could not finalize task %s', identity)
                if outputs is not None:
                    for key, field in entry['config'].outputs.items():
                        if field.is_file():
                            try:
                                output_file = await asyncio.to_thread(self.db.get_file, outputs[key]['file_id'])
                                if output_file and output_file['owner'] == record['owner']:
                                    await asyncio.to_thread(self.storage.discard_output, outputs[key]['file_id'], identity)
                            except Exception:
                                logging.exception('Could not remove unpublished output')
            # Only a known result may settle the observer and release its slot.
            # An unexpected BaseException ends this consumer; entry shutdown
            # then joins the physical Worker before reconciling abandoned work.
            self.running_workers.pop(identity, None)
            entry['queue'].task_done(index, ExecutionStatus(error is None, str(error or '')), outputs if error is None else {}, {})
            future = entry['futures'].pop(identity)
            self.task_futures.pop(identity, None)
            entry['admitted'].discard(identity)
            self.inflight -= 1
            if error is None:
                future.set_result(dict(task_id=identity, outputs=outputs))
            else:
                future.set_exception(error)
            entry['wake'].set()

    def wait_timeout(self, name, seconds=None):
        """Resolve a bounded result-wait policy before accepting a task."""
        from .protocol import DeployRequest
        from .errors import ServerError
        maximum = self._entry(name)['config'].execution.http_wait_timeout_seconds
        if seconds is None:
            return maximum
        try:
            seconds = DeployRequest(wait_timeout_seconds=seconds).wait_timeout_seconds
        except ValueError as exc:
            raise ServerError(422, 'invalid_wait_timeout', 'wait_timeout_seconds must be a finite positive number') from exc
        if seconds > maximum:
            raise ServerError(422, 'invalid_wait_timeout', 'wait_timeout_seconds exceeds this entry\'s HTTP wait limit')
        return seconds

    @staticmethod
    def _wait_error(identity):
        from .errors import ServerError
        return ServerError(504, 'wait_timeout', 'Task is still accepted; query its status or explicitly cancel it',
                           task_id=identity, status_url=f'/v1/tasks/{identity}', accepted=True, retryable=True)

    async def wait_result(self, identity, future, seconds=None):
        """Wait on a caller-owned observer from submit, then detach it."""
        try:
            if seconds is None:
                return await future
            done, _ = await asyncio.wait([future], timeout=seconds)
            if not done:
                raise self._wait_error(identity)
            return future.result()
        finally:
            # submit already shields the Server's execution future. Cancel the
            # observer so timed-out/disconnected readers aren't held by it.
            future.cancel()

    @audited
    async def deploy(self, name, inputs, parameters=None, *, wait_timeout_seconds=None):
        seconds = self.wait_timeout(name, wait_timeout_seconds) if wait_timeout_seconds is not None else None
        identity, future = await self.submit(name, inputs, parameters)
        return await self.wait_result(identity, future, seconds)

    @audited
    async def cancel_task(self, identity):
        return await asyncio.shield(self.own(self._cancel_api_task(identity)))

    async def _cancel_api_task(self, identity):
        from .errors import ServerError
        import time
        record = await asyncio.to_thread(self.db.request_cancel, identity)
        if record is None:
            raise ServerError(404, 'task_not_found', 'Unknown task')
        if record['state'] in ('succeeded', 'failed', 'cancelled', 'expired'):
            return record
        if record['queue_key'] and record['state'] == 'pending':
            try:
                await asyncio.to_thread(self.db.finish_task, identity, 'cancelled', time.time(),
                    error={'code': 'task_cancelled', 'message': 'Cancelled before execution'}, attempt=record['attempt'])
            except RuntimeError:
                pass
            return await asyncio.to_thread(self.db.get_task, identity)
        worker = self.running_workers.get(identity)
        if worker is not None and hasattr(worker, 'cancel'):
            worker.cancel(identity)
        for entry in self.entries.values():
            if identity not in entry['futures']:
                continue
            if self.redis_config is None and entry['queue'].cancel_pending(identity):
                error = ServerError(409, 'task_cancelled', 'Task was cancelled before execution', task_id=identity)
                try:
                    await asyncio.to_thread(self.db.finish_task, identity, 'cancelled', time.time(), error=error.response()['error'])
                finally:
                    future = entry['futures'].pop(identity, None)
                    if future is not None:
                        self.task_futures.pop(identity, None)
                        entry['admitted'].discard(identity)
                        self.inflight -= 1
                        future.set_exception(error)
                        entry['wake'].set()
                return await asyncio.to_thread(self.db.get_task, identity)
        # A request to cancel does not imply native work has actually stopped.
        return record

    async def _finish_abandoned_consumer_tasks(self, entry):
        """Called only after every Worker of this entry confirms physical exit."""
        from .db import TERMINAL
        from .errors import ServerError
        import time
        if self.redis_config is not None:
            # Preserve committed or journaled results before failing unknown
            # executions. Match this runtime, never another Server's Worker.
            await self._reconcile_shared_results(entry)
            guard = entry.get('control_guard')
            if guard is None:
                raise ServerError(503, 'consumer_recovery_unconfirmed', 'Worker runtime ownership is unavailable')
            while True:
                records = await asyncio.to_thread(self.db.runtime_tasks, entry['queue'].key, guard.path.name, entry['file_holder'])
                if not records:
                    break
                for record in records:
                    error = dict(code='task_cancelled' if record['cancel_requested'] else 'consumer_failed',
                                 message='Worker exited before its result was confirmed', retryable=False)
                    try:
                        await asyncio.to_thread(self.db.finish_task, record['task_id'],
                            'cancelled' if record['cancel_requested'] else 'failed', await asyncio.to_thread(self.db.clock), error=error,
                            attempt=record['attempt'], worker_token=record['worker_token'])
                    except RuntimeError:
                        current = await asyncio.to_thread(self.db.get_task, record['task_id'])
                        if current is None or current['state'] not in TERMINAL:
                            raise
            return
        for identity, future in list(entry['futures'].items()):
            record = await asyncio.to_thread(self.db.get_task, identity)
            if record is None:
                raise ServerError(503, 'consumer_recovery_unconfirmed', 'Task admission has not been confirmed')
            if record['state'] not in TERMINAL:
                error = dict(code='task_cancelled' if record['cancel_requested'] else 'consumer_failed',
                             message='Worker exited before its result was confirmed', retryable=False)
                try:
                    await asyncio.to_thread(self.db.finish_task, identity,
                        'cancelled' if record['cancel_requested'] else 'failed', time.time(), error=error,
                        attempt=record['attempt'], worker_token=record['worker_token'])
                except RuntimeError:
                    current = await asyncio.to_thread(self.db.get_task, identity)
                    if current is None or current['state'] not in TERMINAL:
                        raise
                record = await asyncio.to_thread(self.db.get_task, identity)
            entry['queue'].cancel_pending(identity)
            if entry['futures'].pop(identity, None) is None:
                continue
            self.task_futures.pop(identity, None)
            self.running_workers.pop(identity, None)
            entry['admitted'].discard(identity)
            self.inflight -= 1
            if not future.done():
                if record['state'] == 'succeeded':
                    future.set_result(dict(task_id=identity, outputs=record['outputs']))
                else:
                    future.set_exception(ServerError.from_task(record))

    @audited
    async def stop_entry(self, name):
        async def stop():
            await self._stop_entry(name, self._entry(name))
            for previous in list(self.retiring_entries.get(name, [])):
                await self._stop_entry(name, previous, persist=False, shared_scope=False, drain_shared=True)
                self.retiring_entries[name].remove(previous)
                self._release_assets(previous)
            return self.describe(name)
        return await self._control_entry(name, 'stop', stop)

    async def _stop_entry(self, name, entry, *, persist=True, shared_scope=True, drain_shared=None):
        from .errors import ServerError
        if entry['status'] in ('starting', 'reloading'):
            raise ServerError(409, 'entry_busy', 'Entry is changing state', retryable=True)
        if shared_scope and self.shared_runtime is not None and entry.get('shared_key'):
            await self.shared_runtime.check(entry)
        if entry['status'] in ('stopped', 'failed') and not entry['workers']:
            if entry.get('release_pending'):
                await self._release_entry_files(entry)
                entry['status'], entry['error'] = 'stopped', None
            if self.devices is not None:
                await self.devices.release(entry)
            if self.redis_config is not None and entry.get('queue') is not None and shared_scope:
                usage = await asyncio.to_thread(self.db.queue_usage, entry['queue'].key)
                await self._write_shared_control(entry, False, usage['control_token'])
            if persist and entry['status'] == 'stopped':
                await self._persist_entry(name, entry, False)
            return self.describe(name)
        if entry['closing'] is None:
            async def drain():
                usage = None
                if self.redis_config is not None and entry.get('queue') is not None and shared_scope:
                    usage = await asyncio.to_thread(self.db.queue_usage, entry['queue'].key)
                if persist:
                    await self._persist_entry(name, entry, False)
                if usage is not None:
                    await self._write_shared_control(entry, False, usage['control_token'])
                entry['status'] = 'stopping'
                entry['local_stop'] = not (shared_scope if drain_shared is None else drain_shared)
                if entry.get('wake') is not None:
                    entry['wake'].set()
                consumer_results = await asyncio.gather(*entry.get('consumer_runs', entry['consumers']), return_exceptions=True)
                await asyncio.gather(*entry['consumers'], return_exceptions=True)
                if entry.get('maintenance'):
                    entry['maintenance'].cancel()
                    await asyncio.gather(entry['maintenance'], return_exceptions=True)
                failure_message = 'Worker release was not confirmed'
                try:
                    closed = await asyncio.gather(*(worker.close() for worker in entry['workers']), return_exceptions=True)
                    for result in closed:
                        if isinstance(result, BaseException):
                            raise result
                    failure_message = 'Resource cleanup was not confirmed'
                    if entry.get('consumer_failed'):
                        await self._finish_abandoned_consumer_tasks(entry)
                    await self._release_entry_files(entry)
                except BaseException:
                    entry['status'], entry['error'] = 'failed', failure_message
                    raise
                else:
                    entry['workers'] = []
                    entry['status'], entry['error'] = 'stopped', None
                if entry.get('consumer_failed') or any(isinstance(result, BaseException) for result in consumer_results):
                    entry['status'], entry['error'] = 'failed', 'Task consumer exited unexpectedly'
                    raise ServerError(500, 'consumer_failed', 'Task consumer exited unexpectedly; Workers have been closed')
            entry['closing'] = asyncio.create_task(drain())
        closing = entry['closing']
        try:
            await asyncio.shield(closing)
        except BaseException:
            if entry['status'] in ('ready', 'failed') and closing.done() and entry['closing'] is closing:
                # A later Stop may retry idempotent database cleanup. Workers
                # remember actual close failures, so an unconfirmed physical
                # release cannot be turned into success by this retry.
                entry['closing'] = None
            raise
        return self.describe(name)

    async def close(self):
        if self.closing_task is None:
            self.shutting_down = True
            self.maintenance_stop.set()
            self.closing_task = asyncio.create_task(self._close())
            self.closing_task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        await asyncio.shield(self.closing_task)

    async def _close(self):
        stopped = []
        self.shutting_down = True
        if self.editor_access is not None:
            self.queue.pause_editor()
        self.maintenance_stop.set()
        if self.task_maintenance is not None:
            # Do not cancel to_thread during a transaction or dispose its DB
            # underneath it. Wake the timer and join the current bounded batch.
            stopped.extend(await asyncio.gather(self.task_maintenance, return_exceptions=True))
        if self.editor_access is not None:
            self.editor_access.sessions.clear()
        if self.shared_control:
            self.shared_control.cancel()
            await asyncio.gather(self.shared_control, return_exceptions=True)
        if self.control_tasks:
            await asyncio.gather(*self.control_tasks, return_exceptions=True)
        entries = list(self.entries.items()) + [(name, entry) for name, values in self.retiring_entries.items() for entry in values]
        stopped.extend(await asyncio.gather(*(self._stop_entry(name, entry, persist=False, shared_scope=False) for name, entry in entries),
                                            return_exceptions=True))
        for entry, _ in list(self._asset_leases.values()):
            self._release_assets(entry)
        if self.waiters:
            for task in list(self.waiters.values()):
                task.cancel()
            await asyncio.gather(*list(self.waiters.values()), return_exceptions=True)
        if self.redis_config is not None:
            stopped.extend(await asyncio.gather(*(entry['queue'].close() for entry in self.entries.values()
                                                   if entry.get('queue') is not None), return_exceptions=True))
        self.started = False
        if self.editor_access is not None:
            try:
                self.queue.history_store.close()
            except Exception as exc:
                stopped.append(exc)
        try:
            await self.request_limiter.close()
        except Exception as exc:
            stopped.append(exc)
        if self.event_notifications is not None:
            try:
                await self.event_notifications.close()
            except Exception as exc:
                stopped.append(exc)
        try:
            await self.audit.close()
        except Exception as exc:
            stopped.append(exc)
        if getattr(self, 'callbacks', None) is not None:
            stopped.extend(await asyncio.gather(self.callbacks.close(), return_exceptions=True))
        for dependency in (self.controls, self.db, self.storage):
            try:
                dependency.close()
            except Exception as exc:
                stopped.append(exc)
        for result in stopped:
            if isinstance(result, BaseException):
                raise result

    def _write_json_replace(self, dst: Path, data: dict | bytes) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if isinstance(data, dict):
            tmp.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        else:
            tmp.write_bytes(data)
        tmp.replace(dst)

    def _norm_id(self, val) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, str):
            s = val.strip()
            return s or None
        s = str(val).strip()

    def _norm_str(self, s: str | None) -> str | None:
        if s is None:
            return None
        s = str(s)
        s = unicodedata.normalize("NFC", s)
        s = s.strip()
        return s

    def _generate_unique_path(self, desired: Path) -> Path:
        if not desired.exists():
            return desired
        stem, suf = desired.stem, desired.suffix
        i = 1
        while True:
            cand = desired.with_name(f"{stem}({i}){suf}")
            if not cand.exists():
                return cand
            i += 1

    def _get_workdir(self) -> Path:
        return Path(self.args.resources)

    def read_json_recursive(self, root: Path) -> list[dict]:
        result = []
        for json_file in root.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    result.append(content)
            except Exception as e:
                logging.warning(f"Failed to read {json_file}: {e}")
        return result

    def _get_top_level_dir_of_json(self, json_path: str) -> str | None:
        """
        根据 json 文件路径，返回它在 template_path 下所属的顶级目录名称。
        """
        # 规范化路径，防止相对路径问题
        json_path = os.path.abspath(json_path)
        template_root = os.path.abspath(self.template_path)

        # 确保 json 文件在 template_path 下
        if not json_path.startswith(template_root):
            return None

        # 去掉 template_path 部分，得到相对路径
        rel_path = os.path.relpath(json_path, template_root)
        # 获取第一级目录名
        parts = rel_path.split(os.sep)
        if len(parts) > 1:
            return parts[0]
        else:
            # json 文件直接在 template_path 下，没有上级目录
            return None

    def _find_cover_and_requirements(self, json_path: Path) -> tuple[Optional[str], Optional[str]]:
        d = json_path.parent
        base = json_path.stem

        # cover
        cover = None
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            p = d / f"cover.{base}{ext}"
            if p.exists():
                cover = str(p.resolve())
                break

        # requirements
        req = None
        cand = d / f"readme.{base}.md"
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                req = f.read()
        else:
            cand2 = d / "readme.template.md"
            if cand2.exists():
                with open(cand2, "r", encoding="utf-8") as f:
                    req = f.read()

        return cover, req

    def _cancel_task(self, task_id: str):
        result = self.queue.cancel_editor_task(task_id)
        if result is None:
            raise HTTPException(status_code=404, detail='Task not found')
        if (result['state'] in ('PENDING', 'DISPATCHED', 'RUNNING') and self.queue.owns_active(task_id)
                and self.cancel_event_queue is not None and not hasattr(self.queue._job_q, 'cancel')):
            self.cancel_event_queue.put(task_id)
        return result

    def _register_routes(self, args):
        api = APIRouter(prefix="/api")

        def upload_name(file):
            name = file.filename
            if not name or name in ('.', '..') or any(char in name for char in ('/', '\\', '\x00')):
                raise HTTPException(400, 'Invalid upload filename')
            return name

        def resource_file(path):
            resolved = path.resolve()
            roots = [Path(self.args.resources).resolve()]
            if hasattr(self, 'template_path'):
                roots.append(self.template_path.resolve())
            if not any(resolved.is_relative_to(root) for root in roots):
                raise HTTPException(403, 'Path is outside the resource and template directories')
            if not resolved.is_file():
                raise HTTPException(404, 'Not found')
            return resolved

        @api.get(
            "/queue/info",
            tags=["Task"],
            response_model=QueueStateResponse,
            summary="check running / wait task",
        )
        async def queue_state():
            state = await asyncio.to_thread(self.queue.get_current_queue)
            return QueueStateResponse(
                flag="success",
                message="queue state fetched",
                result=QueueStateResult(
                    running=state["RUNNING"],
                    pending=state["PENDING"],
                    dispatched=state["DISPATCHED"],
                ),
            )

        # commit task
        @api.post(
            "/queue",
            tags=["Task"],
            response_model=EnqueueResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="enqueue nndeploy graph task",
        )
        async def enqueue(req: EnqueueRequest, request: Request,
                          workflow_id: str | None = Query(default=None, max_length=128)):
            key = request.headers.get('Idempotency-Key')
            if key is not None:
                try:
                    if str(uuid.UUID(key)) != key:
                        raise ValueError()
                except ValueError:
                    raise HTTPException(422, 'Idempotency-Key must be a canonical UUID') from None
            task_id = key or str(uuid.uuid4())
            payload = {
                "id": task_id,
                "graph_json": req.root,
                "priority": 100,
            }
            if workflow_id is not None:
                payload['workflow_id'] = workflow_id
            def prepare_editor_run(payload, connection):
                from sqlalchemy import select
                from .db import workflows
                from .config import ServerConfig, load_json
                from .control_guard import _boot_id
                if self.devices is not None and not _boot_id():
                    raise ServerError(422, 'editor_device_platform_unsupported',
                        'Editor Run device budgets require verifiable local process-exit evidence', accepted=False)
                path = connection.execute(select(workflows.c.path).where(
                    workflows.c.id == workflow_id)).scalar_one_or_none() if workflow_id else None
                if path is None or not Path(path).with_suffix('.server.json').is_file():
                    if self.devices is None:
                        return payload
                    raise ServerError(422, 'editor_device_config_required',
                        'Save this workflow and configure execution.devices in its .server.json before Run', accepted=False)
                try:
                    config = ServerConfig.model_validate(load_json(Path(path).with_suffix('.server.json')))
                except (ValueError, OSError):
                    raise ServerError(422, 'invalid_editor_execution',
                        'The workflow .server.json is invalid or unavailable', accepted=False) from None
                if self.devices is not None:
                    self.devices.requirements(config)
                elif config.execution.devices:
                    raise ServerError(422, 'device_limits_required',
                        'Configure Server device_limits to enforce the workflow device declarations', accepted=False)
                # Run owns one disposable process; resident-only scheduling
                # options cannot silently change the editor's execution model.
                execution = config.execution
                if execution.workers != 1 or execution.scheduling != 'serial' or execution.batching.enabled:
                    raise ServerError(422, 'editor_execution_unsupported',
                        'Editor Run requires one serial Worker; use Deploy for concurrent or batched execution', accepted=False)
                return dict(payload, _editor_execution=dict(name='workflow-' + workflow_id,
                                                            execution=execution.model_dump()))
            from queue import Full
            from sqlalchemy.exc import SQLAlchemyError
            from .errors import ServerError
            try:
                options = {'prepare': prepare_editor_run} if self.devices is not None or workflow_id is not None else {}
                if key is None:
                    await asyncio.to_thread(self.queue.put, payload, prio=100, **options)
                else:
                    await asyncio.to_thread(self.queue.put, payload, prio=100, idempotent=True, **options)
            except Full:
                raise ServerError(429, 'queue_full', 'Task queue capacity reached',
                                  task_id=task_id, retryable=True, accepted=False) from None
            except SQLAlchemyError:
                raise ServerError(503, 'editor_admission_unconfirmed',
                    'Task admission could not be confirmed; check the task ID before submitting again',
                    task_id=task_id, retryable=True) from None
            flag = "success"
            message = "success"
            result = {"task_id":task_id}
            return EnqueueResponse(flag=flag, message=message, result=result)

        @api.get("/queue/task/{task_id}", tags=["Task"],
                 summary="get editor task status")
        async def editor_task_status(task_id: str):
            record = await asyncio.to_thread(
                self.queue.get_task_by_id, task_id, include_results=False)
            if record is None:
                raise HTTPException(404, 'Task not found')
            fields = ('state', 'status', 'cancel_requested', 'ts_submit',
                      'ts_dispatch', 'ts_start', 'ts_finish', 'worker_pid', 'time_profile',
                      'outputs', 'outputs_truncated', 'output_files', 'outputs_expired')
            return dict(flag='success', message='Task status fetched', result={
                'task_id': task_id, **{key: record[key] for key in fields if key in record},
            })

        @api.get('/queue/task/{task_id}/files/{file_id}', tags=['Task'], summary='read an editor output file')
        async def editor_output_file(task_id: str, file_id: str):
            from .api.deploy import FileContentResponse
            record = await asyncio.to_thread(self.queue.get_task_by_id, task_id, include_results=False)
            if (record is None or record.get('state') != 'SUCCEEDED' or
                    not any(item['file_id'] == file_id for item in record.get('output_files', []))):
                raise HTTPException(404, 'Task output file not found')
            return FileContentResponse(self, file_id, inline=True)

        @api.post(
            "/queue/cancel/{task_id}",
            tags=["Task"],
            summary="cancel task execution",
        )
        async def cancel_task(task_id: str):
            result = await asyncio.to_thread(self._cancel_task, task_id)
            if result['state'] == 'CANCELLED':
                message = 'Task cancelled'
            elif result['state'] in ('SUCCEEDED', 'FAILED'):
                message = 'Task already finished'
            else:
                message = 'Cancellation requested; waiting for execution to finish'
            return dict(flag='success', message=message, result=result)

        @api.post(
            "/queue/flush",
            tags=["Task"],
            summary="clear pending tasks and drain job queue (running tasks not interrupted)"
        )
        async def flush_queue():
            try:
                res = await asyncio.to_thread(self.queue.flush)
                msg = f"pending cleared: {res['cleared_pending']}, job_q drained: {res['drained_job_q']}"
                logging.info("[queue/flush] %s", msg)
                return {"status": "success", "message": msg, "result": res}
            except Exception as e:
                logging.exception("[queue/flush] error")
                raise HTTPException(status_code=500, detail=f"flush queue error: {e}")

        @api.post(
            "/workflow/save",
            tags=["Workflow"],
            response_model=WorkFlowSaveResponse,
            summary="save workflow (update if id provided, else create)",
        )
        async def save_json(req: EnqueueRequest):
            try:
                root = req.root if isinstance(req.root, dict) else {}
                data = root.get("businessContent")
                if data is None:
                    raise HTTPException(status_code=400, detail="businessContent is required")

                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError as e:
                        raise HTTPException(status_code=400, detail=f"businessContent must be valid JSON: {e}")
                elif not isinstance(data, dict):
                    raise HTTPException(status_code=400, detail="businessContent must be a JSON object or JSON string")

                workflow_name = self._norm_str(data.get('name_'))
                if workflow_name and (workflow_name in ('.', '..') or any(
                        char in workflow_name for char in ('/', '\\', '\x00'))):
                    raise HTTPException(400, 'Workflow name must be a filename')

                id = self._norm_id(root.get("id"))

                if id:
                    dst = self.db.get_workflow_path(id)
                    name = self.db.get_workflow_name(id)
                    if not dst:
                        raise HTTPException(status_code=404, detail=f"workflow id {id} not found")
                    save_name = data.get("name_")
                    name = self._norm_str(name)
                    save_name = self._norm_str(save_name)
                    if name == save_name:
                        self._write_json_replace(dst, data)
                        cover, req = self._find_cover_and_requirements(dst)
                        self.db.update_workflow_metadata(id, save_name, dst, cover, req)

                        return WorkFlowSaveResponse(
                            flag="success",
                            message="updated",
                            result={"id": id}
                        )
                    else:
                        base_name = save_name if save_name else f"{uuid.uuid4()}.json"
                        if not base_name.endswith((".json", ".yml", ".yaml")):
                            base_name = f"{base_name}.json"
                        desired = self.workflow_dir / base_name
                        dst = self._generate_unique_path(desired)

                        self._write_json_replace(dst, data)
                        cover, req = self._find_cover_and_requirements(dst)
                        wid = self.db.insert_workflow(save_name, dst, cover, req)

                        return WorkFlowSaveResponse(
                            flag="success",
                            message="created",
                            result={"id": wid}
                        )
                else:
                    name_from_json = data.get("name_")
                    base_name = name_from_json if name_from_json else f"{uuid.uuid4()}.json"
                    if not base_name.endswith((".json", ".yml", ".yaml")):
                        base_name = f"{base_name}.json"
                    desired = self.workflow_dir / base_name
                    dst = self._generate_unique_path(desired)

                    self._write_json_replace(dst, data)
                    cover, req = self._find_cover_and_requirements(dst)
                    wid = self.db.insert_workflow(name_from_json, dst, cover, req)

                    return WorkFlowSaveResponse(
                        flag="success",
                        message="created",
                        result={"id": wid}
                    )

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"save error: {e}")

        self.save_workflow = save_json

        @api.get('/workflow/{id}/server', tags=['Workflow'])
        async def workflow_server(id: str):
            from .config import load_json
            path = self.db.get_workflow_path(id)
            if not path:
                raise HTTPException(404, 'Workflow not found')
            sidecar = path.with_suffix('.server.json')
            config = load_json(sidecar) if sidecar.is_file() else {}
            name = config.get('name')
            if not name or name == 'workflow-' + id:
                name = self.db.get_workflow_name(id)
            active = self.entries.get(name)
            tool = active['config'].mcp.model_dump() if active and active['config'].mcp else config.get('mcp')
            if active and active['config'].mcp is None:
                tool = None
            return {'name': name, 'mcp': tool, 'mcp_path': self.mcp.config.path if self.mcp is not None else None}

        @api.post(
                "/workflow/upload",
                tags=["Workflow"],
                response_model=UploadResponse,
                status_code=status.HTTP_201_CREATED,
                summary="upload workflow",
        )
        async def upload_workflow(
            file: UploadFile = File(...)
        ):
            upload_name(file)
            allowed_extensions = {".json", ".yaml", ".yml"}
            suffix = Path(file.filename).suffix.lower()
            if suffix not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only .json, .yml and .yaml files are allowed, got: {suffix}"
                )
            desired = self.workflow_dir / file.filename
            dst = self._generate_unique_path(desired)
            try:
                content = await file.read()
                data = json.loads(content.decode())
                name = data["name_"]
                self._write_json_replace(dst, content)
                cover, req = self._find_cover_and_requirements(dst)
                wid = self.db.insert_workflow(name, dst, cover, req)

                return UploadResponse(
                    flag="success",
                    message=f"workflow {dst.name} has been uploaded",
                    result={
                        "id": wid,
                        "filename": dst.name,
                        "saved_path": str(dst.resolve()),
                        "size": dst.stat().st_size,
                        "uploaded_at": datetime.utcnow(),
                        "extension": (dst.suffix or "unknown").lstrip("."),
                    }
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"upload error: {e}")

        @api.get(
            "/workflow/download/{id}",
            tags=["Workflow"],
            summary="download workflow by id",
            response_class=FileResponse,
        )
        async def download_workflow(id: str):
            """
            download workflow file by id (UUID)
            """
            f = self.db.get_workflow_path(id)
            if not f:
                raise HTTPException(status_code=404, detail=f"workflow id {id} not found in DB")
            f = Path(f)
            if not f.exists():
                raise HTTPException(status_code=404, detail=f"workflow file not found: {f}")

            MIME_MAP: dict[str, str] = {
                ".json": "application/json",
                ".yaml": "application/x-yaml",
                ".yml": "application/x-yaml"
            }

            media_type = MIME_MAP.get(f.suffix.lower(), "application/octet-stream")
            if not media_type:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported download type: {f.suffix}"
                )

            return FileResponse(f, media_type=media_type, filename=f.name)

        @api.get(
            "/workflows",
            tags=["Workflow"],
            response_model=WorkFlowListResponse,
            summary="list workflows summary",
        )
        async def list_workflows_summary():
            if not self.workflow_dir.exists():
                raise HTTPException(status_code=404, detail="workflow dir is not exist")

            results = []

            try:
                for jf in self.workflow_dir.rglob("*.json"):
                    if jf.name.endswith('.server.json'):
                        continue
                    try:
                        with jf.open("r", encoding="utf-8") as f:
                            content = json.load(f)
                        wid = self.db.upsert_workflow_by_path(jf, content.get("name_"))

                        id_cover = self.db.get_workflow_id_and_cover_by_path(jf)
                        if not id_cover:
                            logging.warning(f"[workflow->db] missing id for {jf}")
                            continue
                        wid, _cover = id_cover

                        results.append({
                            "id": wid,
                            "name_": content.get("name_"),
                            "developer_": content.get("developer_"),
                            "desc_": content.get("desc_"),
                        })
                    except Exception as fe:
                        logging.warning(f"[workflow->scan] failed for {jf}: {fe}")
            except Exception as e:
                logging.warning(f"[workflow->db] batch record failed: {e}")

            return WorkFlowListResponse(
                flag="success",
                message=f"{len(results)} workflows found",
                result=results
            )

        @api.post(
            "/workflow/delete/{id}",
            tags=["Workflow"],
            response_model=WorkFlowDeleteResponse,
            summary="delete workflow by id",
        )
        async def delete_workflow_json(id: str):
            file_path = self.db.get_workflow_path(id)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"workflow id {id} not found in DB")

            file_path = Path(file_path)
            if not file_path.exists():
                self.db.delete_workflow(id)
                raise HTTPException(status_code=404, detail=f"workflow file not found: {file_path}")

            try:
                file_path.unlink(missing_ok=False)
                self.db.delete_workflow(id)
                return WorkFlowDeleteResponse(flag="success", message=f"workflow id {id} has been deleted")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"delete error: {e}")

        @api.get(
            "/workflow/{id}",
            tags=["Workflow"],
            response_model=WorkFlowLoadResponse,
            summary="get workflow json by id",
        )
        async def get_workflow_by_id(id: str):
            file_path = self.db.get_workflow_path(id)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"workflow id {id} not found")
            file_path = Path(file_path)
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"workflow file not found: {file_path}")
            if file_path.suffix.lower() != ".json":
                raise HTTPException(status_code=400, detail=f"unsupported workflow type: {file_path.suffix}")

            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return WorkFlowLoadResponse(flag="success", message="success", result=data)

        @api.get(
            "/template",
            tags=["Template"],
            response_model=TemplateJsonListResponse,
            summary="Recursively return all JSON contents under template directory"
        )
        async def get_all_template_jsons():
            if not self.template_path.exists():
                return TemplateJsonListResponse(
                    flag="error",
                    message="template directory not found",
                    result=[]
                )

            results = []
            new_count = 0

            try:
                for jf in self.template_path.rglob("*.json"):
                    try:
                        cover, req = self._find_cover_and_requirements(jf)
                        category = self._get_top_level_dir_of_json(jf)
                        self.db.insert_or_ignore_template(jf, cover, category, req)

                        meta = self.db.get_template_meta_by_path(jf)
                        if meta is None:
                            logging.warning(f"[template->db] missing id for {jf}")
                            continue
                        tid, cover, requirements, category = meta

                        with jf.open("r", encoding="utf-8") as f:
                            content = json.load(f)

                        item = {
                            "id": tid,
                            "name_": content.get("name_"),
                            "developer_": content.get("developer_"),
                            "source_": content.get("source_"),
                            "desc_": content.get("desc_"),
                            "cover_": cover,
                            "requirements_": requirements,
                            "category_": category,
                        }
                        results.append(item)
                        new_count += 1
                    except Exception as fe:
                        logging.warning(f"[template->scan] failed to handle {jf}: {fe}")

            except Exception as e:
                logging.warning(f"[template->db] batch record failed: {e}")

            return TemplateJsonListResponse(
                flag="success",
                message=f"{len(results)} json files loaded",
                result=results
            )

        @api.get(
            "/template/dir",
            tags=["Template"],
            response_model=TemplateDirListResponse,
            summary="get template top-level dir list",
        )
        async def get_template_dir_json():
            flag = "success"
            message = "get template top-level dir list successfully"
            result = []
            try:
                result =  [
                    name for name in os.listdir(self.template_path)
                    if os.path.isdir(os.path.join(self.template_path, name))
                ]
            except FileNotFoundError:
                flag = "failed"
                message = "template path is not existed"
                result = []
            except Exception as e:
                flag = "failed"
                message = f"Error reading template dir: {e}"
                result = []
            return TemplateDirListResponse(flag=flag, message=message, result=result)

        @api.get(
            "/template/{id}",
            tags=["Template"],
            response_model=WorkFlowLoadResponse,
            summary="get workflow json",
        )
        async def get_template_json(id: str):
            try:
                file_path = self.db.get_template_path(id)
                if file_path is None:
                    raise HTTPException(status_code=404, detail=f"template id {id} not found")
                file_path = Path(file_path)

                if not file_path.exists():
                    raise HTTPException(status_code=404, detail=f"template file not found: {file_path}")

                if file_path.suffix.lower() != ".json":
                    raise HTTPException(status_code=400, detail=f"unsupported template type: {file_path.suffix}")

                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                flag = "success"
                message = "template load response"
                return TemplateLoadResponse(flag=flag, message=message, result=data)

            except HTTPException:
                raise
            except Exception as e:
                logging.exception("[get_template_json] unexpected error")
                raise HTTPException(status_code=500, detail=f"read template error: {e}")

        @api.get(
            "/dag/info",
            tags=["Node"],
            response_model=NodeListResponse,
            summary="return register nodes",
        )
        async def register_nodes():
            json_str = nndeploy.dag.get_dag_json()
            nodes = json.loads(json_str)
            flag = "success"
            message = ""
            return NodeListResponse(flag=flag, message=message, result=nodes)

        @api.post(
                "/nodes/upload",
                tags=["Node"],
                response_model=UploadResponse,
                status_code=status.HTTP_201_CREATED,
                summary="upload node",
        )
        async def upload_plugin(
            file: UploadFile = File(...)
        ):
            upload_name(file)
            allowed_extensions = {".py", ".so"}
            suffix = Path(file.filename).suffix
            if suffix not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only .py and .so files are allowed, got: {suffix}"
                )

            folder = Path(self.args.resources) / "plugin"
            dst = folder / file.filename
            # Keep the upload stream open until its writer exits, including
            # when the HTTP waiter is cancelled.
            write = self.own(asyncio.to_thread(_save_plugin_upload, file.file, dst))
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                await asyncio.shield(write)
                raise

            add_global_import_lib(str(dst.resolve()))
            import_global_import_lib()

            runtime = getattr(self, '_editor_runtime', None)
            if runtime is not None:
                notified = await asyncio.to_thread(runtime.notify_plugin, str(dst.resolve()))
            else:
                notified = False
                if self.plugin_update_q is not None:
                    try:
                        await asyncio.to_thread(self.plugin_update_q.put_nowait, str(dst.resolve()))
                        notified = True
                    except Full:
                        notified = True
                    except (EOFError, OSError):
                        pass

            flag = "success"
            message = f"Node {dst.name} uploaded successfully"
            if not notified:
                message += '; Worker notification deferred; saved plugins are checked before each task'
            result = {
                "filename":file.filename,
                "saved_path":str(dst.resolve()),
                "size":dst.stat().st_size,
                "uploaded_at":datetime.utcnow(),
                "extension": (dst.suffix or "unknown").lstrip(".")
            }
            return UploadResponse(flag=flag, message=message, result=result)

        @api.get(
            "/param/types",
            response_model=ParamTypeResponse,
            summary="return param enum types"
        )
        async def get_param_enum_type():
            type_json = get_type_enum_json()
            flag = "success"
            message = "success"
            return ParamTypeResponse(flag=flag, message=message, result=type_json)

        @api.post('/queue/history/cleanup', tags=['Task'], summary='apply the configured editor history retention')
        async def cleanup_editor_history():
            return dict(flag='success', message='History retention applied',
                        result={'deleted_history': await self.cleanup_editor_history()})

        # history
        @api.get(
            "/queue/history",
            tags=["Task"],
            response_model=HistoryResponse,
            summary="check history",
        )
        async def history(max_items: Optional[int] = Query(None, ge=1, le=1000),
                          before: Optional[str] = Query(None, min_length=1, max_length=64),
                          summary: bool = False):
            from .errors import ServerError
            try:
                hist_map: Dict[str, Dict[str, Any]] = await asyncio.to_thread(self.queue.get_history, max_items=max_items,
                    before=before, include_results=False)

                items: List[Dict[str, Any]] = []
                for task_id, rec in hist_map.items():
                    item = {
                        "task_id": task_id,
                        "task": rec.get("task"),
                        "status": rec.get("status"),
                        "state": rec.get("state"),
                        "ts_submit": rec.get("ts_submit"),
                        "ts_dispatch": rec.get("ts_dispatch"),
                        "ts_start": rec.get("ts_start"),
                        "ts_finish": rec.get("ts_finish"),
                        "worker_pid": rec.get("worker_pid"),
                        "time_profile": rec.get("time_profile", {}),
                    }
                    if summary:
                        task = item.pop('task') or {}
                        item['name'] = (task.get('graph_json') or {}).get('name_', '')
                    items.append(item)

                def _key(x: Dict[str, Any]):
                    return (x.get("ts_finish") or x.get("ts_start") or x.get("ts_submit") or 0.0, x['task_id'])

                items.sort(key=_key, reverse=True)

                return HistoryResponse(
                    flag="success",
                    message="history fetched",
                    result={"items": items, "total": len(items)},
                )
            except ServerError:
                raise
            except Exception:
                logging.exception("Get history failed")
                return HistoryResponse(
                    flag="fail",
                    message="History is temporarily unavailable",
                    result={"items": [], "total": 0},
                )


        # index
        # @self.app.get("/", tags=["Root"])
        # async def root():
        #     return HTMLResponse("<h2>nndeploy backend: API OK</h2>")

        # preview
        # @api.get("/preview", tags=["Files"],
        #         summary="preview images/videos/txt")
        # async def preview_file(file_path: str = Query(..., description="absolute_path or relative path"), time: Optional[str] = None):
        #     file_path = Path(file_path)
        #     # unsafe process for relative path
        #     if file_path.is_absolute():
        #         f = file_path
        #     else:
        #         resource_root = Path(self.args.resources).resolve()
        #         first_part = file_path.parts[0] if file_path.parts else ""
        #         if first_part == resource_root.name:
        #             f = file_path
        #         else:
        #             f = resource_root / file_path

        #     if not f.exists():
        #         raise HTTPException(status_code=404, detail="Not found")

        #     MIME_MAP: dict[str, str] = {
        #         # ---- image ----
        #         ".jpg":  "image/jpeg",
        #         ".jpeg": "image/jpeg",
        #         ".png":  "image/png",
        #         ".webp": "image/webp",
        #         ".gif":  "image/gif",
        #         ".svg":  "image/svg+xml",

        #         # ---- video ----
        #         ".mp4": "video/mp4",
        #         ".mov": "video/quicktime",
        #         ".avi": "video/x-msvideo",
        #         ".mkv": "video/x-matroska",
        #         ".webm": "video/webm",

        #         # ---- text ----
        #         ".txt": "text/plain",
        #     }

        #     mime = MIME_MAP.get(f.suffix.lower())
        #     if mime is None:
        #         raise HTTPException(
        #             status_code=400,
        #             detail="Unsupported preview type"
        #         )
        #     if mime == "text/plain":
        #         return PlainTextResponse(f.read_text(encoding="utf-8"))
        #     else:
        #         return FileResponse(f, media_type=mime, filename=None)

        @api.get("/preview", tags=["Files"], summary="preview images/videos/txt/binary")
        async def preview_file(
            file_path: str = Query(..., description="absolute_path or relative path"),
            time: Optional[str] = None,
            return_mime_type: Literal["video", "image", "text", "binary"] = Query(
                "binary", alias="returnMimeType"
            ),
        ):
            file_path = Path(file_path)
            # unsafe process for relative path
            if file_path.is_absolute():
                f = file_path
            else:
                resource_root = Path(self.args.resources).resolve()
                first_part = file_path.parts[0] if file_path.parts else ""
                if first_part == resource_root.name:
                    f = file_path
                else:
                    f = resource_root / file_path

            f = resource_file(f)

            # ---- mime maps ----
            IMAGE_MIME_MAP: dict[str, str] = {
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png":  "image/png",
                ".webp": "image/webp",
                ".gif":  "image/gif",
                ".svg":  "image/svg+xml",
            }
            VIDEO_MIME_MAP: dict[str, str] = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",
            }

            suffix = f.suffix.lower()

            # ---- route by requested returnMimeType ----
            if return_mime_type == "image":
                mime = IMAGE_MIME_MAP.get(suffix)
                if not mime:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported image type: {suffix}",
                    )
                return FileResponse(f, media_type=mime, filename=None)

            if return_mime_type == "video":
                mime = VIDEO_MIME_MAP.get(suffix)
                if not mime:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported video type: {suffix}",
                    )
                return FileResponse(f, media_type=mime, filename=None)

            if return_mime_type == "text":
                raw_bytes = f.read_bytes()
                detect_result = chardet.detect(raw_bytes)
                encoding = detect_result.get("encoding") or "utf-8"
                confidence = detect_result.get("confidence", 0)

                try:
                    content = raw_bytes.decode(encoding, errors="strict")
                    return {
                        "flag": "success",
                        "message": f"success (detected encoding={encoding}, confidence={confidence:.2f})",
                        "result": content,
                    }
                except UnicodeDecodeError:
                    return {
                        "flag": "error",
                        "message": f"Cannot decode file as {encoding}. Consider using returnMimeType='binary' or specify correct encoding.",
                        "result": None,
                    }

            return FileResponse(f, media_type="application/octet-stream", filename=None)

        # download
        # @api.get("/download", tags=["Files"],
        #         summary="download images/videos/models/txt")
        # async def download_file(file_path: str = Query(..., description="absolute_path or relative path")):
        #     f = Path(file_path)
        #     if not f.exists():
        #         raise HTTPException(status_code=404, detail="Not found")

        #     MIME_MAP: dict[str, str] = {
        #         # ---- image ----
        #         ".jpg":  "image/jpeg",
        #         ".jpeg": "image/jpeg",
        #         ".png":  "image/png",
        #         ".webp": "image/webp",
        #         ".gif":  "image/gif",
        #         ".svg":  "image/svg+xml",

        #         # ---- video ----
        #         ".mp4": "video/mp4",
        #         ".mov": "video/quicktime",
        #         ".avi": "video/x-msvideo",
        #         ".mkv": "video/x-matroska",
        #         ".webm": "video/webm",

        #         # ---- model ----
        #         ".onnx": "application/octet-stream",

        #         # ---- text ----
        #         ".txt": "text/plain"
        #     }

        #     mime = MIME_MAP.get(f.suffix.lower())
        #     if mime is None:
        #         raise HTTPException(
        #             status_code=400,
        #             detail="Unsupported download type"
        #         )
        #     return FileResponse(f, media_type=mime, filename=f.name)

        @api.get("/download", tags=["Files"], summary="download images/videos/models/txt/binary")
        async def download_file(
            file_path: str = Query(..., description="absolute_path or relative path"),
            return_mime_type: Literal["video", "image", "text", "binary"] = Query(
                "binary", alias="returnMimeType"
            ),
        ):
            f = resource_file(Path(file_path))

            IMAGE_MIME_MAP: dict[str, str] = {
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png":  "image/png",
                ".webp": "image/webp",
                ".gif":  "image/gif",
                ".svg":  "image/svg+xml",
            }
            VIDEO_MIME_MAP: dict[str, str] = {
                ".mp4": "video/mp4",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".webm": "video/webm",
            }

            suffix = f.suffix.lower()

            if return_mime_type == "image":
                mime = IMAGE_MIME_MAP.get(suffix)
                if not mime:
                    raise HTTPException(status_code=400, detail=f"Unsupported image type: {suffix}")
                return FileResponse(f, media_type=mime, filename=f.name)

            if return_mime_type == "video":
                mime = VIDEO_MIME_MAP.get(suffix)
                if not mime:
                    raise HTTPException(status_code=400, detail=f"Unsupported video type: {suffix}")
                return FileResponse(f, media_type=mime, filename=f.name)

            if return_mime_type == "text":
                try:
                    return PlainTextResponse(f.read_text(encoding="utf-8"))
                except UnicodeDecodeError:
                    raise HTTPException(
                        status_code=400,
                        detail="File is not valid UTF-8 text. Use returnMimeType='binary' instead."
                    )
            return FileResponse(f, media_type="application/octet-stream", filename=f.name)

        @api.post(
            "/models/download",
            status_code=status.HTTP_202_ACCEPTED,
            summary="start model download by json"
        )
        async def download_models(req: EnqueueRequest):
            task_id = str(uuid.uuid4())
            graph_json = req.root if isinstance(req.root, dict) else {}
            asyncio.create_task(self._download_models_task(task_id, graph_json))
            return JSONResponse({
                "flag": "success",
                "message": "download started",
                "result": {"task_id": task_id}
            }, status_code=status.HTTP_202_ACCEPTED)

        @api.get(
            "/resources",
            tags=["resources"],
            status_code=status.HTTP_200_OK,
            summary="resource directory"
        )
        async def get_resources():
            return JSONResponse({
                "flag": "success",
                "message": "resource directory",
                "result": str(Path(self.args.resources).resolve())
            }, status_code=status.HTTP_200_OK)

        @api.websocket("/ws/progress")
        async def ws_progress(ws: WebSocket):
            if len(self.sockets) >= 64:
                await ws.close(code=1013)
                return
            await ws.accept()
            self.sockets.add(ws)
            self.ws_task_map[ws] = set()

            try:
                while True:
                    message = await ws.receive()
                    if message['type'] == 'websocket.disconnect':
                        break
                    text = message.get('text')
                    if text is None:
                        await ws.close(code=1003)
                        break
                    if len(text.encode('utf-8')) > 16384:
                        await ws.close(code=1009)
                        break
                    try:
                        msg = json.loads(text)
                    except ValueError:
                        await ws.close(code=1008)
                        break
                    if isinstance(msg, dict) and msg.get("type") == "bind" and isinstance(msg.get("task_id"), str):
                        task_id = msg["task_id"]
                        if len(task_id) > 128 or len(self.ws_task_map[ws]) >= 256:
                            await ws.close(code=1008)
                            break
                        self.ws_task_map[ws].add(task_id)
                        self.task_ws_map.setdefault(task_id, set()).add(ws)
            except WebSocketDisconnect:
                logging.info("[WebSocket] client disconnected")
            finally:
                self.sockets.discard(ws)
                task_ids = self.ws_task_map.pop(ws, set())
                for tid in task_ids:
                    self.task_ws_map.get(tid, set()).discard(ws)
                    if not self.task_ws_map.get(tid):
                        self.task_ws_map.pop(tid, None)

        self.app.include_router(
            files_router,
            dependencies=[Depends(self._get_workdir)]
        )
        self.app.dependency_overrides[get_workdir] = self._get_workdir

        self.app.include_router(api,dependencies=[Depends(lambda: get_workdir(self))])

        if not args.debug:
            web_root = getattr(args, 'front_end_dir', None)
            if web_root is None:
                from .frontend import FrontendManager
                web_root = FrontendManager.init_frontend(args.front_end_version)
            dist_inside = Path(web_root) / "dist"
            if dist_inside.is_dir():
                web_root = str(dist_inside)
            if not (Path(web_root) / 'index.html').is_file():
                raise ValueError('Frontend build must contain index.html')
            self.app.mount("/", SPAStaticFiles(directory=web_root, html=True), name="frontend")

    # model download notify
    def notify_download_done(self, task_id: str, success: bool, result: dict | None, error: str | None):
        payload = {
            "flag": "success" if success else "failed",
            "message": "download done" if success else f"download failed: {error}",
            "result": {
                "task_id": task_id,
                "type": "model_download_done",
                "detail": result or {}
            }
        }
        ws_set = self.task_ws_map.get(task_id, set())
        if not ws_set:
            logging.warning("[notify_download_done] Ws_set is empty, timeout")
            return
        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self._broadcast(payload, ws), self.loop)
            else:
                logging.warning("[notify_download_done] Event loop not ready or not running")

    # task progress notify
    def notify_task_progress(self, task_id: str, status_dict: dict):
        flag = "success"
        message = "task running"
        result = {
            "task_id": task_id,
            "type": "progress",
            "detail": status_dict
        }
        payload = {"flag": flag, "message": message, "result": result}
        ws_set = self.task_ws_map.get(task_id, set())
        if not ws_set:
            return

        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(payload, ws),
                    self.loop
                )
            else:
                logging.warning("[notify_task_progress] Event loop not ready or not running")

    # task done notify
    def notify_task_done(self, task_id: str, status: ExecutionStatus, results: Dict, time_profile_map: Dict):
        task_info = self.queue.get_task_by_id(task_id)
        if task_info is None:
            raise HTTPException(status_code=404, detail="task not found")
        graph_json = task_info.get("task").get("graph_json")

        # path, text = extract_encode_output_paths(graph_json)
        # send result
        # flag = status.str
        # message = status.messages
        # result = {"task_id": task_id, "type": "preview", "path": path, "text": text}
        # payload = {"flag": flag, "message": message, "result": result}
        # ws_set = self.task_ws_map.get(task_id, set())
        # for ws in ws_set.copy():
        #     if self.loop and self.loop.is_running():
        #         asyncio.run_coroutine_threadsafe(
        #             self._broadcast(payload, ws),
        #             self.loop
        #         )
        #     else:
        #         logging.warning("[notify_task_done] Event loop not ready or not running")

        # send graph output
        # show_items = []
        # try:
        #     for node_name, out_map in (results or {}).items():
        #         if not isinstance(out_map, dict):
        #             continue
        #         if "String" not in out_map:
        #             continue
        #         img_val = out_map.get("String")
        #         img_list = img_val if isinstance(img_val, list) else [img_val]
        #         text_list = [str(x) for x in img_list]
        #         show_items.append({"name": node_name, "text": text_list})
        # except Exception as e:
        #     logging.exception("[notify_task_done] build show_items failed: %s", e)

        # if show_items:
        #     payload = {
        #         "flag": status.str,
        #         "message": status.messages,
        #         "result": {"task_id": task_id, "type": "show", "string": show_items},
        #     }
        #     for ws in ws_set.copy():
        #         if self.loop and self.loop.is_running():
        #             asyncio.run_coroutine_threadsafe(self._broadcast(payload, ws), self.loop)
        #         else:
        #             logging.warning("[notify_task_done] Event loop not ready or not running")

        # Reuse the preview committed with task completion. Polling and live
        # WebSocket delivery must display the same result after reconnect.
        from .editor_outputs import preview
        content, truncated = preview(results)
        content = task_info.get('outputs', content)
        truncated = task_info.get('outputs_truncated', truncated)

        payload = {
            "flag": status.str,
            "message": status.messages,
            "result": {
                "type": "memory",
                "task_id": task_id,
                "content": content,
                "truncated": truncated,
            },
        }
        ws_set = self.task_ws_map.get(task_id, set())
        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self._broadcast(payload, ws), self.loop)
                logging.warning("[notify_task_done] Event loop ready or running")
            else:
                logging.warning("[notify_task_done] Event loop not ready or not running")

        # send graph run info
        flag = status.str
        message = "graph run info"
        result = {"task_id": task_id, "type": "task_run_info", "time_profile": time_profile_map,
                  "outputs": content, "outputs_truncated": truncated,
                  "output_files": task_info.get('output_files', []),
                  "outputs_expired": task_info.get('outputs_expired', False)}
        payload = {"flag": flag, "message": message, "result": result}
        ws_set = self.task_ws_map.get(task_id, set())
        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._broadcast(payload, ws),
                    self.loop
                )
            else:
                logging.warning("[notify_task_done] Event loop not ready or not running")

    def notify_system_event(self, event: str, data: dict | None = None) -> None:
        """notify system event to all connected websocket clients"""
        payload = {
            "flag": "success",
            "message": "system event",
            "result": {
                "type": "system",
                "event": event,
                "detail": data or {},
                "ts": datetime.utcnow().isoformat() + "Z",
            },
        }
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(payload, None), self.loop)
        else:
            logging.warning("[notify_system_event] Event loop not ready or not running")

    async def _broadcast(self, payload: dict, ws: WebSocket | None = None):
        targets = [ws] if ws else list(self.sockets)
        for w in targets:
            try:
                # logging.info(f"[_broadcast] sending to {w.client}")
                await w.send_json(payload)
            except Exception as e:
                logging.error(f"[_broadcast] failed to send: {e}")
                self.sockets.discard(w)

    async def _wait_ws_binding(self, task_id: str, timeout: float = 60.0, poll: float = 0.05) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self.task_ws_map.get(task_id):
                return True
            await asyncio.sleep(poll)
        return False

    # async def _download_models_task(self, task_id: str, graph_json: dict):
    #     token = set_task_id(task_id)
    #     try:
    #         await self._wait_ws_binding(task_id, timeout=10.0)
    #         with scoped_stdio_to_logging("model-download"):
    #             logging.info("model download task started")

    #             result = await asyncio.to_thread(
    #                 run_func_in_copied_context,
    #                 _handle_urls, graph_json, self.args.resources
    #             )
    #             logging.info("model download task finished: %s", result)
    #             self.notify_download_done(task_id, success=True, result=result, error=None)
    #     except Exception as e:
    #         logging.exception("model download task failed")
    #         self.notify_download_done(task_id, success=False, result=None, error=str(e))
    #     finally:
    #         reset_task_id(token)

        # download progress notify (WS)
    def notify_download_progress(self, task_id: str, ev: dict):
        """
        Broadcast a single download progress event to all sockets bound to `task_id`.

        Expected `ev` example (from DownloadProgressHandler):
        {
            "phase": "progress" | "done" | "error",
            "filename": "detect/yolo11s.sim.onnx",
            "downloaded": 1048576,
            "total": 37958400,
            "percent": 2.76,
            "elapsed": 1.23,
            "logger": "model-download" | "modelscope",
            ... # (optional) task_id, bps, eta, etc.
        }
        """
        payload = {
            "flag": "success",
            "message": "download progress",
            "result": {
                "task_id": task_id,
                "type": "download_progress",
                "detail": ev
            }
        }

        ws_set = self.task_ws_map.get(task_id, set())
        if not ws_set:
            return

        for ws in ws_set.copy():
            if self.loop and self.loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(self._broadcast(payload, ws), self.loop)
                except Exception as e:
                    logging.warning(f"[notify_download_progress] schedule send failed: {e}")
            else:
                logging.warning("[notify_download_progress] Event loop not ready or not running")


    async def _download_models_task(self, task_id: str, graph_json: dict):
        token = set_task_id(task_id)
        handler = None
        try:
            await self._wait_ws_binding(task_id, timeout=10.0)

            loop = asyncio.get_running_loop()
            def emit_cb(ev):
                self.notify_download_progress(task_id, ev)

            handler = DownloadProgressHandler(loop, emit_cb, logger_names=("model-download", "modelscope"))
            loggers = [logging.getLogger("model-download"), logging.getLogger("modelscope")]
            for lg in loggers:
                lg.addHandler(handler)
                lg.setLevel(logging.INFO)
                lg.propagate = True

            with scoped_stdio_to_logging("model-download"):
                logging.info("model download task started")

                result = await asyncio.to_thread(
                    run_func_in_copied_context,
                    _handle_urls, graph_json, self.args.resources
                )

                logging.info("model download task finished: %s", result)
                self.notify_download_done(task_id, success=True, result=result, error=None)

        except Exception as e:
            logging.exception("model download task failed")
            self.notify_download_progress(task_id, {
                "phase": "error", "error": str(e), "task_id": task_id
            })
            self.notify_download_done(task_id, success=False, result=None, error=str(e))
        finally:
            if handler:
                # 卸载 handler，避免影响其它任务
                for lg_name in ("model-download", "modelscope"):
                    lg = logging.getLogger(lg_name)
                    try:
                        lg.removeHandler(handler)
                    except Exception:
                        pass
                handler.close()
            reset_task_id(token)


# Existing editor and CLI imports remain supported.
NnDeployServer = Server


def main(argv=None):
    """Compatibility entry point sharing the editor and Deploy CLI."""
    from .app import main as app_main
    return app_main(argv)

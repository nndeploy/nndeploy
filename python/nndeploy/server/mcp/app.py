"""Stateless MCP transport over the existing Server task runtime."""
import json
import logging
from contextlib import asynccontextmanager, contextmanager

from jsonschema import Draft202012Validator
from mcp import types
from mcp.server.lowlevel import Server as ProtocolServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..errors import ServerError
from ..operations import read_task, cancel_task, submit_task
from ..protocol import TaskResponse
from .schema import object_schema, tool_schema


class MCP:
    def __init__(self, server, config):
        self.server, self.config = server, config
        self.reservations = {}
        self.manager = None
        self.protocol = ProtocolServer('nndeploy')
        self.protocol.list_tools()(self.list_tools)
        # Validate against this caller's current catalog, not the SDK's shared cache.
        self.protocol.call_tool(validate_input=False)(self.call_tool)
        server.app.router.routes.append(Route(config.path, self, methods=['GET', 'POST', 'DELETE']))

    @property
    def exports(self):
        return {entry['config'].mcp.name: dict(entry=name, description=entry['config'].mcp.description)
                for name, entry in self.server.entries.items() if entry['config'].mcp is not None}

    def validate_export(self, entry, tool):
        if tool is None:
            return
        existing = self.exports.get(tool.name)
        if (existing is not None and existing['entry'] != entry) or self.reservations.get(tool.name, entry) != entry:
            raise ServerError(409, 'mcp_name_in_use', 'This MCP tool name is already used by another entry.')

    @contextmanager
    def reserve(self, entry, tool):
        self.validate_export(entry, tool)
        previous = self.reservations.get(tool.name) if tool else None
        if tool:
            self.reservations[tool.name] = entry
        try:
            yield
        finally:
            if tool:
                if previous is None:
                    self.reservations.pop(tool.name, None)
                else:
                    self.reservations[tool.name] = previous

    @asynccontextmanager
    async def lifespan(self):
        self.manager = StreamableHTTPSessionManager(self.protocol, stateless=True, json_response=True,
            security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                allowed_hosts=self.config.allowed_hosts, allowed_origins=self.config.allowed_origins))
        try:
            async with self.manager.run():
                yield
        finally:
            self.manager = None

    async def __call__(self, scope, receive, send):
        request = Request(scope, receive)
        try:
            caller = self.server.authentication.authenticate(request.headers.get('authorization', ''))
            scope.setdefault('state', {})['audit_identity'] = caller
            await self.server.audit.identify(scope, caller)
            await self.server.request_limiter.take(caller.subject)
        except ServerError as exc:
            return await JSONResponse(exc.response(), status_code=exc.status_code)(scope, receive, send)
        if self.manager is None:
            return await JSONResponse({'error': 'MCP is not running'}, status_code=503)(scope, receive, send)
        await self.manager.handle_request(scope, receive, send)

    def caller(self):
        return self.protocol.request_context.request.scope['state']['audit_identity']

    def catalog(self, caller):
        tools = []
        for name, export in self.exports.items():
            entry_name = export['entry']
            if caller.allows('deploy', entry_name):
                tools.append(types.Tool(name=name, description=export['description'],
                                        inputSchema=tool_schema(self.server.entries[entry_name])))
        task_schema = object_schema({'task_id': {'type': 'string', 'minLength': 1}}, ['task_id'])
        if caller.allows('tasks:read'):
            tools.append(types.Tool(name='get_task', description='Read an owned task status and outputs.', inputSchema=task_schema))
        if caller.allows('tasks:cancel'):
            tools.append(types.Tool(name='cancel_task', description='Request cancellation of an owned task; inspect the returned state.', inputSchema=task_schema))
        return tools

    async def list_tools(self):
        return self.catalog(self.caller())

    async def call_tool(self, name, arguments):
        try:
            caller = self.caller()
            tool = next((tool for tool in self.catalog(caller) if tool.name == name), None)
            if tool is None:
                raise ServerError(404, 'tool_not_found', 'Unknown or unavailable tool')
            arguments = {} if arguments is None else arguments
            if next(Draft202012Validator(tool.inputSchema).iter_errors(arguments), None) is not None:
                raise ServerError(422, 'invalid_arguments', 'Arguments do not match the tool inputSchema')
            state = self.protocol.request_context.request.scope['state']
            if name in ('get_task', 'cancel_task'):
                identity = arguments['task_id']
                state['audit_resource'] = identity
                value = await (read_task(self.server, identity, caller) if name == 'get_task'
                               else cancel_task(self.server, identity, caller))
                state['audit_entry'] = value['name']
                value = TaskResponse.model_validate(value).model_dump(mode='json')
            else:
                entry = self.exports[name]['entry']
                state['audit_entry'] = entry
                value = await submit_task(self.server, entry, arguments.get('inputs', {}),
                    arguments.get('parameters', {}), caller, idempotency_key=arguments.get('idempotency_key'))
                state['audit_resource'] = value['task_id']
            return self.result(value)
        except ServerError as exc:
            return self.result(exc.response(), error=True)
        except Exception:
            logging.getLogger(__name__).exception('MCP tool failed')
            return self.result({'error': {'code': 'internal_error', 'message': 'Internal server error'}}, error=True)

    @staticmethod
    def result(value, error=False):
        return types.CallToolResult(content=[types.TextContent(type='text', text=json.dumps(value, ensure_ascii=False))],
                                    structuredContent=value, isError=error)

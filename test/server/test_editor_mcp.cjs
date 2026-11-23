const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');
const compiled = ts.transpileModule(fs.readFileSync('app/workflow/src/components/tools/server-request.ts', 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const context = { exports: {}, URL };
vm.runInNewContext(compiled, context);
const { mcpValidation, mcpConnectionUrl } = context.exports;

test('MCP export validates names, descriptions and reserved task tools', () => {
  assert.equal(mcpValidation(false, '', ''), '');
  assert.equal(mcpValidation(true, 'detect_objects', 'Detect objects'), '');
  for (const name of ['', 'has space', '中文', 'x'.repeat(129), 'get_task', 'cancel_task']) {
    assert.ok(mcpValidation(true, name, 'Description'), name);
  }
  assert.ok(mcpValidation(true, 'valid', '   '));
  assert.ok(mcpValidation(true, 'valid', 'x'.repeat(4097)));
});

test('connection URL follows the configured path without accepting external destinations', () => {
  assert.equal(mcpConnectionUrl('/mcp', 'https://example.com'), 'https://example.com/mcp');
  assert.equal(mcpConnectionUrl('/tools/mcp', 'http://localhost:8000'), 'http://localhost:8000/tools/mcp');
  for (const path of [null, undefined, '//evil.test/mcp', 'https://evil.test/mcp', '/mcp?token=secret']) {
    assert.equal(mcpConnectionUrl(path, 'https://example.com'), '');
  }
});

function availability(fetch) {
  const scoped = { exports: {}, URL, fetch };
  vm.runInNewContext(compiled, scoped);
  return scoped.exports.loadMcpAvailability;
}

test('old backend HTML fallback produces an actionable MCP upgrade error', async () => {
  const load = availability(async () => new Response('<!DOCTYPE html><html></html>', {
    status: 200, headers: { 'Content-Type': 'text/html' },
  }));
  await assert.rejects(() => load(''), /restart.*--mcp/i);
});

test('MCP availability preserves authentication and validates response shape', async () => {
  let headers;
  const load = availability(async (_url, options) => {
    headers = options.headers;
    return Response.json({ enabled: true, path: '/mcp' });
  });
  assert.equal((await load('test-token')).path, '/mcp');
  assert.equal(headers.Authorization, 'Bearer test-token');
  await assert.rejects(() => availability(async () => Response.json({ enabled: true }))(''), /invalid/i);
  await assert.rejects(() => availability(async () => Response.json({}, { status: 401 }))(''), /sign in|token/i);
});

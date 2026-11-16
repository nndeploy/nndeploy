/* Exercise the editor's actual request helper without the React runtime. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const { webcrypto } = require('node:crypto');
const ts = require('../../app/workflow/node_modules/typescript');

const source = fs.readFileSync(path.join(__dirname, '../../app/workflow/src/components/tools/server-request.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
const location = '/api/server?request_id=req_' + 'a'.repeat(32);
const reply = (status, value, headers = {}) => ({ status, ok: status >= 200 && status < 300,
  headers: { get: name => headers[name.toLowerCase()] ?? null }, json: async () => value });
const done = { name: 'model', status: 'ready' };

function client(fetch) {
  let now = 0;
  let timer = 0;
  const timers = new Map();
  const calls = [];
  const context = { exports: {}, URL, Uint8Array, crypto: webcrypto, AbortController,
    window: { location: { origin: 'http://localhost' } }, Date: { now: () => now },
    fetch: async (url, options) => { calls.push({ url, ...options }); return fetch(url, options, calls.length); },
    // Sleeps advance a virtual clock. Abort deadlines fire only when requested.
    setTimeout: (fn, ms) => {
      const id = ++timer;
      timers.set(id, { fn, ms });
      if (ms === 1000) queueMicrotask(() => {
        if (timers.delete(id)) { now += ms; fn(); }
      });
      return id;
    },
    clearTimeout: id => timers.delete(id),
  };
  vm.runInNewContext(compiled, context);
  return { request: context.exports.serverRequest, calls, timers,
    expire: () => { for (const [id, { fn, ms }] of [...timers]) { timers.delete(id); now += ms; fn(); } } };
}

test('slow control becomes authenticated GET polling without a second POST', async () => {
  const c = client((url, options, index) => index <= 2 ? reply(202, { status_url: location }) : reply(200, done));
  assert.deepEqual(await c.request('/api/server/model/start', 'admin', {}), done);
  assert.deepEqual(c.calls.map(call => call.method), ['POST', 'GET', 'GET']);
  assert.equal(c.calls[1].url, location);
  assert.equal(c.calls[1].headers.Authorization, 'Bearer admin');
  assert.ok(c.calls[0].headers['Idempotency-Key']);
  assert.equal(c.calls[1].headers['Idempotency-Key'], undefined);
  assert.equal(c.timers.size, 0);
});

test('lost response retries preserve the key, including a later manual retry', async () => {
  let offline = true;
  const c = client(() => { if (offline) throw new Error('Network unavailable'); return reply(200, done); });
  await assert.rejects(c.request('/api/server/model/start', '', {}), /Network unavailable/);
  assert.equal(c.calls.length, 3);
  offline = false;
  assert.deepEqual(await c.request('/api/server/model/start', '', {}), done);
  assert.equal(new Set(c.calls.map(call => call.headers['Idempotency-Key'])).size, 1);
  await c.request('/api/server/model/start', '', {});
  assert.notEqual(c.calls.at(-1).headers['Idempotency-Key'], c.calls[0].headers['Idempotency-Key']);
});

test('changed workflow has a different key while the unresolved original keeps its key', async () => {
  const c = client(() => { throw new Error('offline'); });
  for (const value of ['old', 'new', 'old']) {
    await assert.rejects(c.request('/api/workflow/server/start', '', { workflow: value }), /offline/);
  }
  assert.notEqual(c.calls[0].headers['Idempotency-Key'], c.calls[3].headers['Idempotency-Key']);
  assert.equal(c.calls[0].headers['Idempotency-Key'], c.calls[6].headers['Idempotency-Key']);
});

test('status URL never sends management credentials to another origin or route', async () => {
  for (const status_url of ['https://example.com/api/server', '/api/workflow', '/api/server', location + '&extra=1']) {
    const c = client(() => reply(202, { status_url }));
    await assert.rejects(c.request('/api/server/model/start', 'secret', {}), /invalid status URL/);
    assert.equal(c.calls.length, 1);
  }
});

test('temporary proxy errors retry the same request without inventing a new action', async () => {
  const c = client((url, options, index) => index < 3 ? reply(503, { error: { message: 'Proxy unavailable' } }) : reply(200, done));
  assert.deepEqual(await c.request('/api/server/model/start', '', {}), done);
  assert.equal(c.calls.length, 3);
  assert.equal(new Set(c.calls.map(call => call.headers['Idempotency-Key'])).size, 1);
});

test('stored failures and unconfirmed controls surface instead of automatic action retries', async () => {
  for (const status of [422, 503]) {
    const c = client(() => reply(status, { error: { message: 'Inspect current state' } },
      { 'idempotency-retention-seconds': '604800', location }));
    await assert.rejects(c.request('/api/server/model/start', '', {}), /Inspect current state/);
    assert.equal(c.calls.length, 1);
  }
});

test('GET refresh never includes a mutation key and polling has an overall deadline', async () => {
  const c = client(() => reply(200, done));
  await c.request('/api/server/model', '');
  assert.equal(c.calls[0].method, 'GET');
  assert.equal(c.calls[0].headers['Idempotency-Key'], undefined);
  const pending = client(() => reply(202, { status_url: location }));
  await assert.rejects(pending.request('/api/server/model/start', '', {}), /still processing/);
  assert.equal(pending.calls.length, 900);
});

test('a stalled response body is aborted and retried with the original key', async () => {
  let c;
  c = client((url, options, index) => {
    if (index > 1) return reply(200, done);
    return { ...reply(200, done), json: () => new Promise((resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new Error('Response timeout')), { once: true });
      queueMicrotask(() => c.expire());
    }) };
  });
  assert.deepEqual(await c.request('/api/server/model/start', '', {}), done);
  assert.equal(c.calls.length, 2);
  assert.equal(c.calls[0].headers['Idempotency-Key'], c.calls[1].headers['Idempotency-Key']);
  assert.equal(c.timers.size, 0);
});

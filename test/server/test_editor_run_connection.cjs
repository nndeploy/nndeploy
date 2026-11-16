/* Execute the actual editor connection helper with controlled network/timers. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');
const { webcrypto } = require('node:crypto');
const ts = require('../../app/workflow/node_modules/typescript');
const source = fs.readFileSync(require('node:path').join(__dirname,
  '../../app/workflow/src/components/tools/run-connection.ts'), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText;
const reply = (status, body) => ({ ok: status < 400, status, json: async () => body });
const flush = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); };
function harness(fetch) {
  const timers = new Map(), calls = [], sockets = [], states = [], messages = [];
  let id = 0;
  const context = { exports: {}, AbortController, crypto: webcrypto,
    require: () => ({ rememberRun: () => true, recordRunState: () => {}, restoreRun: () => undefined }),
    setTimeout: (fn, ms) => { timers.set(++id, { fn, ms }); return id; },
    clearTimeout: id => timers.delete(id),
    fetch: async (url, options) => { calls.push({ url, ...options }); return fetch(url, options, calls.length); },
  };
  vm.runInNewContext(code, context);
  const createSocket = () => {
    const socket = { readyState: 0, sent: [], send(value) { this.sent.push(JSON.parse(value)); },
      close() { this.closed = true; this.onclose?.(); } };
    sockets.push(socket);
    return socket;
  };
  return { ...context.exports, timers, calls, sockets, states, messages,
    watch() { return context.exports.watchRunTask('task-1', createSocket,
      value => states.push(value), value => messages.push(value)); },
    async tick(ms) { for (const [id, timer] of [...timers]) if (timer.ms === ms) {
      timers.delete(id); timer.fn();
    } await flush(); },
  };
}

test('rebinds the same task after disconnect and recovers a missed terminal event', async () => {
  let state = 'RUNNING';
  const c = harness(() => reply(200, { result: { task_id: 'task-1', state } }));
  const stop = c.watch();
  await flush();
  const first = c.sockets[0];
  first.readyState = 1; first.onopen();
  assert.equal(first.sent[0].task_id, 'task-1');
  first.onclose();
  state = 'SUCCEEDED';
  await c.tick(1000);
  const next = c.sockets[1];
  next.readyState = 1; next.onopen();
  await flush();
  assert.equal(next.sent[0].task_id, 'task-1');
  assert.equal(c.states.at(-1).state, 'SUCCEEDED');
  assert.ok(c.calls.every(call => call.url === '/api/queue/task/task-1' && !call.body));
  stop();
  assert.equal(c.timers.size, 0);
  assert.equal(next.closed, true);
});

test('polling survives unknown admission and database failures without resubmission', async () => {
  const c = harness((url, options, index) => index === 1 ? reply(404, {})
    : index === 2 ? reply(503, {}) : reply(200, { result: { task_id: 'task-1', state: 'CANCELLED' } }));
  const stop = c.watch();
  await flush();
  assert.equal(c.states.length, 0);
  await c.tick(2000);
  assert.equal(c.states.length, 0);
  await c.tick(2000);
  assert.equal(c.states.at(-1).state, 'CANCELLED');
  stop();
});

test('stopping observation aborts a query and ignores late network and socket events', async () => {
  let resolve;
  const c = harness(() => new Promise(done => { resolve = done; }));
  const stop = c.watch();
  const socket = c.sockets[0];
  const late = socket.onmessage;
  stop();
  assert.equal(c.calls[0].signal.aborted, true);
  resolve(reply(200, { result: { task_id: 'task-1', state: 'SUCCEEDED' } }));
  late({ data: JSON.stringify({ result: { task_id: 'task-1', type: 'log' } }) });
  await flush();
  assert.equal(c.states.length, 0);
  assert.equal(c.messages.length, 0);
  assert.equal(c.timers.size, 0);
});

test('only this task and system messages are delivered, malformed frames are ignored', () => {
  const c = harness(() => reply(404, {}));
  const stop = c.watch();
  const emit = value => c.sockets[0].onmessage({ data: value });
  emit('{'); emit('null'); emit(JSON.stringify({ result: { task_id: 'other', type: 'log' } }));
  emit(JSON.stringify({ result: { type: 'log' } }));
  emit(JSON.stringify({ result: { task_id: 'task-1', type: 'log' } }));
  emit(JSON.stringify({ result: { type: 'system' } }));
  assert.equal(c.messages.length, 2);
  stop();
});

test('unconfirmed admission retains the client task identity without automatic POST retry', async () => {
  const c = harness((url, options) => reply(503, { error: { code: 'editor_admission_unconfirmed',
    task_id: options.headers['Idempotency-Key'] } }));
  const result = await c.submitRunTask({ name_: 'graph' });
  assert.equal(result.result.task_id, c.calls[0].headers['Idempotency-Key']);
  assert.equal(result.flag, 'pending');
  assert.equal(c.calls.length, 1);
  assert.equal(c.timers.size, 0);
});

test('lost replies can retry the exact original body and key even after the graph is edited', async () => {
  const c = harness((url, options, index) => {
    if (index === 1) throw new Error('Disconnected after commit');
    return reply(202, { flag: 'success', result: { task_id: options.headers['Idempotency-Key'] } });
  });
  const graph = { name_: 'original' };
  const pending = await c.submitRunTask(graph);
  assert.equal(pending.flag, 'pending');
  graph.name_ = 'edited';
  const done = await c.retryRunTask(pending.result.task_id);
  assert.equal(done.flag, 'success');
  assert.equal(c.calls[0].body, c.calls[1].body);
  assert.equal(c.calls[0].headers['Idempotency-Key'], c.calls[1].headers['Idempotency-Key']);
  await assert.rejects(c.retryRunTask(pending.result.task_id), /already confirmed/);
});

test('explicit first rejection is final but a rejected retry cannot erase unknown acceptance', async () => {
  const rejected = harness(() => reply(503, { error: { accepted: false, message: 'Recovering' } }));
  assert.equal((await rejected.submitRunTask({})).flag, 'error');
  const lost = harness((url, options, index) => {
    if (index === 1) throw new Error('Disconnected');
    return reply(401, { error: { message: 'Sign in again' } });
  });
  const pending = await lost.submitRunTask({});
  const retry = await lost.retryRunTask(pending.result.task_id);
  assert.equal(retry.flag, 'pending');
  assert.equal(retry.result.task_id, pending.result.task_id);
  assert.equal(retry.message, 'Sign in again');
  assert.equal(lost.timers.size, 0);
});

test('a stalled status request is aborted before another poll is started', async () => {
  const c = harness((url, options, index) => index === 1 ? new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new Error('aborted')));
  }) : reply(200, { result: { task_id: 'task-1', state: 'FAILED' } }));
  const stop = c.watch();
  const socket = c.sockets[0];
  socket.readyState = 1;
  socket.onopen();
  assert.equal(c.calls.length, 1);
  await c.tick(10000);
  assert.equal(c.calls[0].signal.aborted, true);
  await c.tick(2000);
  assert.equal(c.states.at(-1).state, 'FAILED');
  stop();
  assert.equal(c.timers.size, 0);
});

test('failed socket construction still polls and does not leak retries after cleanup', async () => {
  const c = harness(() => reply(200, { result: { task_id: 'other', state: 'SUCCEEDED' } }));
  let attempts = 0;
  const stop = c.watchRunTask('task-1', () => { attempts++; throw new Error('offline'); },
    state => c.states.push(state), () => {});
  await flush();
  assert.equal(c.states.length, 0);
  await c.tick(1000);
  assert.equal(attempts, 2);
  stop();
  assert.equal(c.timers.size, 0);
});

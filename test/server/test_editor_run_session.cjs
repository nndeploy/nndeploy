/* Fresh module contexts model a page reload sharing the tab's session storage. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');
const { webcrypto } = require('node:crypto');
function storage() {
  const data = new Map();
  return { data, get length() { return data.size; }, key: i => [...data.keys()][i] ?? null,
    getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
}
function page(sessionStorage, fetch) {
  const modules = new Map();
  const calls = [];
  function load(name) {
    if (modules.has(name)) return modules.get(name);
    const code = ts.transpileModule(fs.readFileSync(path.join(__dirname,
      '../../app/workflow/src/components/tools/' + name + '.ts'), 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    const context = { exports: {}, sessionStorage, crypto: webcrypto, AbortController, setTimeout, clearTimeout,
      require: name => load(name.replace('./', '')),
      fetch: async (url, options) => { calls.push({ url, ...options }); return fetch(url, options); },
    };
    vm.runInNewContext(code, context);
    modules.set(name, context.exports);
    return context.exports;
  }
  return { ...load('run-session'), ...load('run-connection'), calls };
}
const tab = { tabId: '2', id: 'workflow-one', name: 'Image copy', flowType: 'workspace' };
const reply = (status, body) => ({ ok: status < 400, status, json: async () => body });
const accepted = (url, options) => reply(202, { flag: 'success', result: { task_id: options.headers['Idempotency-Key'] } });

test('refresh during an in-flight POST restores original body and identity before any reply', async () => {
  const store = storage();
  let resolve;
  const first = page(store, () => new Promise(done => { resolve = done; }));
  const submitting = first.submitRunTask({ name_: 'original' }, tab);
  assert.equal(first.calls.length, 1);
  const next = page(store, accepted);
  assert.equal(next.restoreRunTabs()[0].id, tab.id);
  const recovered = next.restoreRun(tab.tabId);
  assert.equal(recovered.taskId, first.calls[0].headers['Idempotency-Key']);
  assert.equal(recovered.confirmed, false);
  const retry = await next.retryRunTask(recovered.taskId);
  assert.equal(retry.flag, 'success');
  assert.equal(next.calls[0].body, first.calls[0].body);
  assert.equal(first.calls[0].url, '/api/queue?workflow_id=workflow-one');
  assert.equal(next.calls[0].url, first.calls[0].url);
  resolve(reply(503, {}));
  await submitting;
});

test('a pending request from an older page keeps its original URL on retry', async () => {
  const store = storage();
  const previous = page(store, accepted);
  const id = webcrypto.randomUUID();
  previous.rememberRun(id, JSON.stringify({ name_: 'old' }), tab);
  const next = page(store, accepted);
  next.restoreRun(tab.tabId);
  assert.equal((await next.retryRunTask(id)).flag, 'success');
  assert.equal(next.calls[0].url, '/api/queue');
});

test('confirmed tasks restore observation without resubmitting and terminal states clean storage', async () => {
  const store = storage();
  const first = page(store, accepted);
  const response = await first.submitRunTask({ name_: 'original' }, tab);
  const next = page(store, () => { throw new Error('Unexpected POST'); });
  const restored = next.restoreRun(tab.tabId);
  assert.equal(restored.taskId, response.result.task_id);
  assert.equal(restored.confirmed, true);
  assert.equal(next.calls.length, 0);
  next.recordRunState(restored.taskId, 'SUCCEEDED');
  assert.equal(store.length, 0);
  assert.equal(next.restoreRunTabs().length, 0);
});

test('multiple unsaved tabs and saved workflow identities remain distinct across refresh', async () => {
  const store = storage();
  const first = page(store, accepted);
  for (const index of ['2', '3']) await first.submitRunTask({ name_: 'draft-' + index },
    { ...tab, tabId: index, id: '', name: 'Unsaved ' + index });
  const next = page(store, accepted);
  assert.equal(next.restoreRunTabs().length, 2);
  assert.equal(JSON.parse(next.restoreRun('2').payload).name_, 'draft-2');
  assert.equal(JSON.parse(next.restoreRun('3').payload).name_, 'draft-3');
  next.updateRunTab({ ...tab, tabId: '2', id: 'saved-now' });
  assert.equal(page(store, accepted).restoreRunTabs().find(t => t.tabId === '2').id, 'saved-now');
});

test('late completion of a previous task cannot clear a newer task in the same tab', async () => {
  const store = storage();
  const first = page(store, accepted);
  const old = await first.submitRunTask({ name_: 'old' }, tab);
  first.recordRunState(old.result.task_id, 'SUCCEEDED');
  const newer = await first.submitRunTask({ name_: 'new' }, tab);
  first.recordRunState(old.result.task_id, 'FAILED');
  assert.equal(page(store, accepted).restoreRun(tab.tabId).taskId, newer.result.task_id);
});

test('invalid stored data is ignored without deleting other application data', () => {
  const store = storage();
  store.setItem('other-app', 'kept');
  store.setItem('nndeploy.editor.run.v1:2', '{broken');
  store.setItem('nndeploy.editor.run.v1:3', JSON.stringify({ taskId: 'bad', payload: 'not JSON', tab }));
  assert.equal(page(store, accepted).restoreRunTabs().length, 0);
  assert.equal(store.getItem('other-app'), 'kept');
});

test('unavailable browser storage preserves normal Run and explicitly reports the limitation', async () => {
  const store = storage();
  store.setItem = () => { throw new Error('Quota exceeded'); };
  const first = page(store, accepted);
  const result = await first.submitRunTask({ name_: 'graph' }, tab);
  assert.equal(result.flag, 'success');
  assert.match(result.storageWarning, /keep this tab open/i);
  assert.equal(first.calls.length, 1);
});

test('a clear first rejection removes the record, but uncertain retries retain it across refresh', async () => {
  const store = storage();
  const rejected = page(store, () => reply(429, { error: { accepted: false, message: 'Full' } }));
  assert.equal((await rejected.submitRunTask({ name_: 'graph' }, tab)).flag, 'error');
  assert.equal(store.length, 0);
  const lost = page(store, () => { throw new Error('Disconnected'); });
  const pending = await lost.submitRunTask({ name_: 'graph' }, tab);
  const next = page(store, () => reply(401, { error: { message: 'Sign in' } }));
  next.restoreRun(tab.tabId);
  assert.equal((await next.retryRunTask(pending.result.task_id)).flag, 'pending');
  assert.equal(page(store, accepted).restoreRun(tab.tabId).taskId, pending.result.task_id);
});

test('an old page response cannot remove or replace a newer run stored by a refreshed page', async () => {
  const store = storage();
  const oldPage = page(store, accepted);
  const old = await oldPage.submitRunTask({ name_: 'old' }, tab);
  const nextPage = page(store, accepted);
  nextPage.restoreRun(tab.tabId);
  nextPage.recordRunState(old.result.task_id, 'SUCCEEDED');
  const current = await nextPage.submitRunTask({ name_: 'new' }, tab);
  oldPage.recordRunState(old.result.task_id, 'RUNNING');
  oldPage.recordRunState(old.result.task_id, 'SUCCEEDED');
  assert.equal(page(store, accepted).restoreRun(tab.tabId).taskId, current.result.task_id);
});

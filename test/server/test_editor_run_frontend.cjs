/* Exercise the real editor state reducer used by Stop and progress messages. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');
const source = fs.readFileSync(path.join(__dirname, '../../app/workflow/src/components/tools/run-state.ts'), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText;
const context = { exports: {}, Date };
vm.runInNewContext(compiled, context);
const apply = context.exports.applyRunTaskState;
const running = () => ({ isRunning: true, runningTaskId: 'current', result: '', time: 1,
  flowNodesRunningStatus: { native: 'running' }, log: { items: ['kept'] }, outputResource: { old: true } });

test('an accepted cancellation keeps the current execution and progress visible', () => {
  const info = running();
  const next = apply(info, { task_id: 'current', state: 'RUNNING', cancel_requested: true });
  assert.equal(next.isRunning, true);
  assert.equal(next.cancelRequested, true);
  assert.equal(next.runningTaskId, info.runningTaskId);
  assert.equal(next.flowNodesRunningStatus, info.flowNodesRunningStatus);
  assert.equal(next.log, info.log);
  assert.equal(info.cancelRequested, undefined);
});

test('only confirmed cancellation returns the button to Run', () => {
  const requested = apply(running(), { task_id: 'current', state: 'RUNNING', cancel_requested: true });
  const done = apply(requested, { task_id: 'current', state: 'CANCELLED', cancel_requested: true });
  assert.equal(done.isRunning, false);
  assert.equal(done.cancelRequested, false);
  assert.equal(done.result, 'cancelled');
  assert.equal(done.flowNodesRunningStatus, requested.flowNodesRunningStatus);
});

test('a fast task restores final per-node timings even without a progress frame', () => {
  const info = running();
  info.flowNodesRunningStatus = {};
  const nodes = { decode: { status: 'DONE', time: 2.5 }, encode: { status: 'DONE', time: 3.2 } };
  const done = apply(info, { task_id: 'current', state: 'SUCCEEDED', time_profile: { run_time: 5.7, nodes } });
  assert.equal(done.flowNodesRunningStatus, nodes);
  assert.equal(done.log.time_profile.nodes, nodes);
});

test('late cancellation replies cannot restart or overwrite a completed result', () => {
  const done = apply(running(), { task_id: 'current', state: 'SUCCEEDED' });
  assert.equal(apply(done, { task_id: 'current', state: 'RUNNING', cancel_requested: true }), done);
  assert.equal(apply(done, { task_id: 'current', state: 'CANCELLED' }), done);
});

test('a previous task cannot finish a newly started task', () => {
  const info = running();
  assert.equal(apply(info, { task_id: 'previous', state: 'CANCELLED' }), info);
  assert.equal(apply(info, { task_id: '', state: 'SUCCEEDED' }), info);
});

test('success and failure stay distinct after a cancellation request', () => {
  const info = apply(running(), { task_id: 'current', state: 'DISPATCHED', cancel_requested: true });
  assert.equal(apply(info, { task_id: 'current', state: 'SUCCEEDED' }).result, 'success');
  assert.equal(apply(info, { task_id: 'current', state: 'FAILED' }).result, 'error');
});

test('unknown states do not stop a task or mark a cancellation confirmed', () => {
  const info = running();
  for (const state of ['accepted', 'toString', '__proto__']) {
    assert.equal(apply(info, { task_id: 'current', state, cancel_requested: true }), info);
  }
});

test('status polling restores terminal timings without replacing existing logs', () => {
  const info = running();
  const profile = { init_time: 0.2, run_time: 2.5 };
  const next = apply(info, { task_id: 'current', state: 'SUCCEEDED', time_profile: profile });
  assert.equal(next.result, 'success');
  assert.equal(next.log.items, info.log.items);
  assert.equal(next.log.time_profile, profile);
  assert.equal(apply(next, { task_id: 'current', state: 'RUNNING' }), next);
});

test('terminal status restores output when the memory websocket message was lost', () => {
  const info = running();
  const outputs = { sink: 'durable result' };
  const next = apply(info, { task_id: 'current', state: 'SUCCEEDED', outputs });
  assert.equal(next.outputResource.type, 'memory');
  assert.equal(next.outputResource.content, outputs);
  assert.equal(next.outputResource.task_id, 'current');
  assert.equal(next.log, info.log);
  assert.equal(apply(next, { task_id: 'previous', state: 'SUCCEEDED', outputs: {} }), next);
});

test('admission confirmation restores Stop and late retry replies cannot regress it', () => {
  const unknown = { ...running(), submissionPending: true };
  const applySubmission = context.exports.applyRunSubmission;
  assert.equal(applySubmission(unknown, 'previous', 'success'), unknown);
  const pending = applySubmission(unknown, 'current', 'pending');
  assert.equal(pending.submissionPending, true);
  const active = apply(unknown, { task_id: 'current', state: 'RUNNING' });
  assert.equal(active.submissionPending, false);
  assert.equal(active.isRunning, true);
  assert.equal(applySubmission(active, 'current', 'pending'), active);
  const done = apply(unknown, { task_id: 'current', state: 'CANCELLED' });
  assert.equal(done.submissionPending, false);
  assert.equal(applySubmission(done, 'current', 'success'), done);
});

test('terminal polling restores file-only results and clears previous outputs', () => {
  const files = [{ node_path: ['encode'], file_id: 'f', url: '/api/queue/task/current/files/f' }];
  const next = apply(running(), { task_id: 'current', state: 'SUCCEEDED', output_files: files });
  assert.equal(next.outputResource.files, files);
  assert.equal(Object.keys(next.outputResource.content).length, 0);
  assert.equal(next.outputResource.task_id, 'current');
  const failed = apply(running(), { task_id: 'current', state: 'FAILED', output_files: [] });
  assert.equal(failed.outputResource.files.length, 0);
});

test('expired saved results preserve the original success while clearing old previews', () => {
  const next = apply(running(), { task_id: 'current', state: 'SUCCEEDED', outputs: {},
    output_files: [], outputs_expired: true });
  assert.equal(next.isRunning, false);
  assert.equal(next.result, 'success');
  assert.equal(next.outputResource.expired, true);
  assert.equal(Object.keys(next.outputResource.content).length, 0);
  assert.equal(next.outputResource.files.length, 0);
});

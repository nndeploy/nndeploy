/* Exercise the editor's real Axios interceptor against Server error formats. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');
const source = fs.readFileSync(path.join(__dirname,
  '../../app/workflow/src/request/implements/axios.ts'), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText;

function interceptor() {
  const messages = [];
  let receive;
  const context = { exports: {}, setTimeout, console, require(name) {
    if (name === 'axios') return { default: { create: () => ({ interceptors: {
      response: { use: callback => { receive = callback; } },
    } }) } };
    if (name === 'lodash') return { isArray: Array.isArray };
    if (name === '@douyinfe/semi-ui') return { Toast: { warning: value => messages.push(value) } };
    return {};
  } };
  vm.runInNewContext(code, context);
  new context.exports.AxiosRequest();
  return { messages, receive };
}

for (const [name, data, expected] of [
  ['Server message', { message: 'template directory not found' }, 'template directory not found'],
  ['structured Server error', { error: { code: 'queue_full', message: 'Task queue capacity reached' } }, 'Task queue capacity reached'],
  ['legacy error', { error: 'Legacy failure' }, 'Legacy failure'],
  ['legacy msg', { msg: 'Legacy message' }, 'Legacy message'],
  ['empty fields', { error: ' ', message: null, msg: '' }, 'Request failed. Please try again.'],
  ['non-text fields', { error: { code: 'unknown' }, message: 42, msg: [] }, 'Request failed. Please try again.'],
]) {
  test(`shows readable text for ${name}`, () => {
    const { receive, messages } = interceptor();
    const response = { config: {}, data: { flag: 'error', ...data } };
    assert.equal(receive(response), response);
    assert.deepEqual(messages, [expected]);
  });
}

test('success and binary downloads do not show errors', () => {
  const { receive, messages } = interceptor();
  for (const response of [{ config: {}, data: { flag: 'success', result: {} } },
    { config: { responseType: 'blob' }, data: Buffer.from('download') }]) {
    assert.equal(receive(response), response);
  }
  assert.deepEqual(messages, []);
});

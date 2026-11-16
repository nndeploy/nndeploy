const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');
const source = fs.readFileSync(path.join(__dirname,
  '../../app/workflow/src/components/tools/webui/functions.ts'), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true,
} }).outputText;
const context = { exports: {}, require: name => require('../../app/workflow/node_modules/' + name) };
vm.runInNewContext(code, context);
const { getIoTypeFieldName, getIoTypeFieldValue } = context.exports;

test('False output is unchecked rather than treated as a nonempty true string', () => {
  const { getBooleanOutput } = context.exports;
  assert.equal(getBooleanOutput('True'), true);
  assert.equal(getBooleanOutput(' true '), true);
  assert.equal(getBooleanOutput('False'), false);
  assert.equal(getBooleanOutput('false'), false);
  assert.equal(getBooleanOutput(''), false);
  assert.equal(getBooleanOutput(undefined), false);
});

test('output nodes without parameter objects do not crash Webui', () => {
  const node = { name_: 'sink', node_type_: 'Output', io_type_: 'String' };
  assert.equal(getIoTypeFieldName(node), 'empty-field');
  assert.equal(getIoTypeFieldValue(node), undefined);
});

test('input fields still resolve from the node or nested parameters', () => {
  const direct = { io_param_: ['str_'], str_: 'input' };
  assert.equal(getIoTypeFieldName(direct), 'str_');
  assert.equal(getIoTypeFieldValue(direct), 'input');
  const nested = { param: { required_params_: ['value'], value: 'nested' } };
  assert.equal(getIoTypeFieldName(nested), 'param.value');
  assert.equal(getIoTypeFieldValue(nested), 'nested');
});

test('file previews match full node paths even when nested names collide', () => {
  const { getOutputFileUrl } = context.exports;
  const output = { files: [
    { node_path: ['nested', 'encode'], url: '/nested' },
    { node_path: ['encode'], url: '/root' },
  ] };
  assert.equal(getOutputFileUrl(output, ['encode']), '/root');
  assert.equal(getOutputFileUrl(output, ['nested', 'encode']), '/nested');
  assert.equal(getOutputFileUrl(output, ['other', 'encode']), undefined);
  assert.equal(getOutputFileUrl({}, ['encode']), undefined);
});

test('multiple file previews match their declared field and keep legacy history readable', () => {
  const { getOutputFileUrl, getOutputFileLabel } = context.exports;
  const first = { node_path: ['group', 'output'], field_path: ['param_', 'image'], url: '/image' };
  const second = { node_path: ['group', 'output'], field_path: ['param_', 'mask'], url: '/mask' };
  const output = { files: [second, first] };
  assert.equal(getOutputFileUrl(output, ['group', 'output'], ['param_', 'image']), '/image');
  assert.equal(getOutputFileUrl(output, ['group', 'output'], ['param_', 'mask']), '/mask');
  assert.equal(getOutputFileUrl(output, ['group', 'output'], ['wrong']), undefined);
  assert.equal(getOutputFileUrl(output, ['output'], ['param_', 'image']), undefined);
  assert.equal(getOutputFileUrl(output, ['group', 'output']), undefined);
  assert.equal(getOutputFileLabel(second), 'group / output / param_.mask');
  assert.equal(getOutputFileUrl({ files: [{ node_path: ['output'], url: '/old' }] }, ['output'], ['image']), '/old');
  assert.equal(getOutputFileLabel({ node_path: ['output'] }), 'output');
});

test('native io_params and nested canvas ancestry identify the saved output', () => {
  const { getOutputNodePath } = context.exports;
  assert.equal(getIoTypeFieldName({ param_: { io_params_: ['image', 'mask'] } }), 'param_.image');
  const node = (name, parent) => ({ form: { getValueIn: () => name }, parent });
  assert.deepEqual(Array.from(getOutputNodePath(node('output', node('group', {})))), ['group', 'output']);
});

test('file list indices identify separate downloads without selecting an arbitrary preview', () => {
  const { getOutputFileUrl, getOutputFileLabel } = context.exports;
  const file = { node_path: ['output'], field_path: ['param_', 'images', 2, 1], url: '/item' };
  const output = { files: [file, { node_path: ['output'], field_path: ['param_', 'images[2][1]'], url: '/literal' }] };
  assert.equal(getOutputFileUrl(output, ['output'], ['param_', 'images', 2, 1]), '/item');
  assert.equal(getOutputFileUrl(output, ['output'], ['param_', 'images[2][1]']), '/literal');
  assert.equal(getOutputFileUrl(output, ['output'], ['param_', 'images']), undefined);
  assert.equal(getOutputFileUrl(output, ['output'], ['param_', 'images', '2', '1']), undefined);
  assert.equal(getOutputFileLabel(file), 'output / param_.images[2][1]');
});

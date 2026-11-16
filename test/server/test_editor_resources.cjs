const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');
const source = fs.readFileSync(path.join(__dirname, '../../app/workflow/src/components/resource-files.ts'), 'utf8');
function load(fetch) {
  const context = { exports: {}, fetch, Blob, FormData };
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
  } }).outputText, context);
  return context.exports;
}

test('model and directory fields preserve complete paths and skip inline weights and outputs', () => {
  const { getResourceFields, setResourceValue, resourceFieldName } = load();
  const node = { name_: 'model', io_type_: 'Model', param_: { io_params_: ['weights', 'weights', 'literal.path', 'absent'],
    weights: [{ file_id: 'file_original' }], 'literal.path': 'local.bin' } };
  const fields = getResourceFields(node);
  assert.equal(fields.length, 2);
  assert.deepEqual(Array.from(fields[1].path), ['param_', 'literal.path']);
  const lodash = require('../../app/workflow/node_modules/lodash');
  assert.deepEqual(lodash.toPath(resourceFieldName(['param_', 'literal.path', 'quote"[0]'])),
    ['param_', 'literal.path', 'quote"[0]']);
  const updated = setResourceValue(node, fields[1].path, { file_id: 'file_new' });
  assert.equal(updated.param_['literal.path'].file_id, 'file_new');
  assert.equal(node.param_['literal.path'], 'local.bin');
  assert.equal(updated.param_.weights, node.param_.weights);
  assert.equal(getResourceFields({ ...node, node_type_: 'Output' }).length, 0);
  assert.equal(getResourceFields({ io_type_: 'Model', io_params_: ['model_value_'], model_value_: ['inline'], is_path_: false }).length, 0);
  assert.equal(getResourceFields({ io_type_: 'Dir', artifacts: '', io_param_: ['artifacts'] })[0].directory, true);
});

test('managed references and local paths have safe display labels without object string methods', () => {
  const { resourceLabel, isFileReference } = load();
  assert.equal(resourceLabel({ file_id: 'file_model' }), 'Uploaded model file');
  assert.equal(resourceLabel({ file_id: 'file_dir' }, true), 'Uploaded model directory');
  assert.equal(resourceLabel('C:\\models\\weights.bin'), 'weights.bin');
  assert.equal(resourceLabel(undefined), 'No file selected');
  assert.equal(isFileReference({ file_id: 'file_good', path: '/outside' }), false);
  assert.equal(isFileReference({ file_id: 'file_' + 'a'.repeat(60) }), false);
});

test('browser upload uses existing editor endpoint, declared MIME and abort signal', async () => {
  const controller = new AbortController();
  for (const directory of [false, true]) {
    const { uploadResource } = load(async (url, options) => {
      assert.equal(url, '/api/files/upload');
      assert.equal(options.headers, undefined);
      assert.equal(options.signal, controller.signal);
      assert.equal(options.body.get('file').type, directory ? 'application/zip' : 'application/octet-stream');
      assert.equal(await options.body.get('file').text(), 'model data');
      return new Response(JSON.stringify({ flag: 'success', result: { file_id: 'file_uploaded' } }), { status: 201 });
    });
    const result = await uploadResource(new File(['model data'], directory ? 'MODEL.ZIP' : 'model.onnx'), directory, controller.signal);
    assert.equal(result.file_id, 'file_uploaded');
  }
});

test('invalid directories and failed uploads never produce replacement references', async () => {
  const { uploadResource } = load(async () => { throw new Error('Unexpected network request'); });
  await assert.rejects(uploadResource(new File(['bad'], 'folder.bin'), true), /Select a ZIP/);
  const failed = load(async () => new Response(JSON.stringify({ error: { message: 'File limit exceeded' } }), { status: 413 }));
  await assert.rejects(failed.uploadResource(new File(['model'], 'a.bin'), false), /File limit exceeded/);
  const malformed = load(async () => new Response(JSON.stringify({ flag: 'success', result: { saved_path: '/legacy' } }), { status: 201 }));
  await assert.rejects(malformed.uploadResource(new File(['model'], 'a.bin'), false), /invalid file reference/);
});

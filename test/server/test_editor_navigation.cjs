const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');

const source = fs.readFileSync(path.join(__dirname,
  '../../app/workflow/src/pages/Layout/Design/Design.tsx'), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
  jsx: ts.JsxEmit.React, esModuleInterop: true,
} }).outputText;

// Run the real navigation effect, retaining hook state across Home/Design visits.
function editor(restored = []) {
  const states = [], refs = [], effects = [];
  let stateIndex, refIndex;
  let location;
  const react = {
    useEffect: effect => effects.push(effect),
    useState: value => {
      const index = stateIndex++;
      if (!(index in states)) states[index] = value;
      return [states[index], next => { states[index] = next; }];
    },
    useReducer: () => [{}, () => {}],
    useRef: value => {
      const index = refIndex++;
      return refs[index] ||= { current: value };
    },
    createElement: () => null,
  };
  const context = {
    URLSearchParams,
    exports: {},
    require: name => {
      if (name === 'react') return react;
      if (name === 'react-router-dom') return { useLocation: () => location };
      if (name.endsWith('run-session')) return { restoreRunTabs: () => restored };
      if (name === '@douyinfe/semi-ui') return { Typography: {}, Nav: {} };
      if (name.endsWith('/enum')) return { EnumFlowType: { workspace: 'workspace', template: 'template' } };
      return {};
    },
  };
  vm.runInNewContext(code, context);
  return {
    visit(search, visible = true, state = null) {
      location = { search, state, key: String(Math.random()) };
      stateIndex = refIndex = 0;
      effects.length = 0;
      context.exports.default({ visible });
      effects[0]();
      return { tabs: states[1], active: states[0] };
    },
  };
}

for (const flowType of ['workspace', 'template']) {
  test(`Home preserves tabs; a new ${flowType} appends and an existing workflow reuses its tab`, () => {
    const old = { id: 'old', tabId: '5', name: 'Old', flowType: 'workspace' };
    const page = editor([old]);
    const first = page.visit(`?id=first&name=First&flowType=${flowType}`);
    assert.equal(first.tabs.length, 2);
    assert.equal(first.tabs[0], old);
    const home = page.visit('', false);
    assert.equal(home.tabs, first.tabs, 'Home must not change the existing tab state');
    const second = page.visit(`?id=second&name=Second&flowType=${flowType}`);
    assert.equal(second.tabs.length, 3);
    assert.equal(second.tabs[1], first.tabs[1], 'The original tab and canvas identity survive');
    assert.equal(second.active, second.tabs[2].tabId);
    page.visit('', false);
    const reopened = page.visit(`?id=first&name=First&flowType=${flowType}`);
    assert.equal(reopened.tabs.length, 3);
    assert.equal(reopened.active, first.tabs[1].tabId);
  });
}

test('creating a blank workflow from Home appends without discarding existing tabs', () => {
  const page = editor();
  const first = page.visit('?id=first&flowType=workspace');
  page.visit('', false);
  const blank = page.visit('', true, { createWorkflow: true });
  assert.equal(blank.tabs.length, 2);
  assert.equal(blank.tabs[0], first.tabs[0]);
  assert.equal(blank.tabs[1].id, '');
  assert.equal(blank.active, blank.tabs[1].tabId);
});

test('workspace and template with the same ID remain distinct tabs', () => {
  const page = editor();
  page.visit('?id=same&flowType=workspace');
  page.visit('', false);
  const result = page.visit('?id=same&flowType=template');
  assert.equal(result.tabs.length, 2);
  assert.equal(result.tabs[0].flowType, 'workspace');
  assert.equal(result.tabs[1].flowType, 'template');
});

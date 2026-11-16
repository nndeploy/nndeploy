const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const ts = require('../../app/workflow/node_modules/typescript');
const source = fs.readFileSync(path.join(__dirname,
  '../../app/workflow/src/pages/Layout/Design/store/store.tsx'), 'utf8');
const code = ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText;
const context = { exports: {}, require(name) {
  if (name === 'react') return { createContext: value => value };
  if (name.endsWith('actionType')) return { INIT_DAG_GRAPH_INFO: 'catalog' };
  if (name.endsWith('buildNodeRegistry')) return { buildNodeRegistry: node => ({
    type: node.key_, meta: { isContainer: !!node.is_graph_ },
  }) };
  if (name.endsWith('/comment')) return { CommentNodeRegistry: { type: 'comment' } };
  if (name.endsWith('/group')) return { GroupNodeRegistry: { type: 'group' } };
  throw new Error('Unexpected module: ' + name);
} };
vm.runInNewContext(code, context);
const { reducer, initialState } = context.exports;

test('root Graph metadata also renders nested Graphs without another picker entry', () => {
  const graph = { key_: 'nndeploy.dag.Graph', name_: 'Graph', is_graph_: true };
  const node = { key_: 'test.Output', name_: 'Output' };
  const catalog = { graph, nodes: [{ id: node.key_, name: node.name_, type: 'leaf', nodeEntity: node }] };
  const original = JSON.stringify(catalog);
  const state = reducer(initialState, { type: 'catalog', payload: catalog });
  const registry = state.nodeRegistries.find(item => item.type === graph.key_);
  assert.equal(registry.meta.isContainer, true);
  assert.equal(registry.meta.nodePanelVisible, false);
  assert(!JSON.stringify(state.treeData).includes(graph.key_));
  assert.equal(JSON.stringify(catalog), original);
});

test('a Graph already listed as a node is not registered twice', () => {
  const graph = { key_: 'nndeploy.dag.Graph', name_: 'Graph', is_graph_: true };
  const state = reducer(initialState, { type: 'catalog', payload: {
    graph, nodes: [{ id: graph.key_, name: graph.name_, type: 'leaf', nodeEntity: graph }],
  } });
  assert.equal(state.nodeRegistries.filter(item => item.type === graph.key_).length, 1);
});

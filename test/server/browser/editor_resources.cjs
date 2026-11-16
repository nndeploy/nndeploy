// Exercise authenticated model selection through the real editor and Graph.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, request } = require(process.env.NNDEPLOY_PLAYWRIGHT || 'playwright');
const state = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const base = `http://127.0.0.1:${state.port}`;
const file = name => path.join(state.root, name);
async function until(check, description) {
  const deadline = Date.now() + 20000;
  while (!(await check())) {
    if (Date.now() > deadline) throw new Error(description);
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}
(async () => {
  const api = await request.newContext({ baseURL: base,
    extraHTTPHeaders: { Authorization: 'Bearer browser-resource-test' } });
  const browser = await chromium.launch({ headless: true,
    executablePath: process.env.NNDEPLOY_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    const created = await api.post('/api/workflow/save', { data: {
      businessContent: JSON.parse(fs.readFileSync(file('graph.json'), 'utf8')),
    } });
    assert.equal(created.status(), 200, await created.text());
    const identity = (await created.json()).result.id;
    await page.goto(`${base}/design?id=${identity}&name=model-resources&flowType=workspace`);
    await page.getByLabel('Management token', { exact: true }).fill('browser-resource-test');
    await page.getByRole('button', { name: 'Sign in', exact: true }).click();
    await page.getByRole('button', { name: 'Sign out', exact: true }).waitFor();
    await page.getByLabel('Choose model artifacts 1', { exact: true }).waitFor({ state: 'attached' });
    async function choose(container, label, filename, expected = 201) {
      const uploaded = page.waitForResponse(response => response.request().method() === 'POST' &&
        new URL(response.url()).pathname === '/api/files/upload');
      await container.getByLabel(label, { exact: true }).setInputFiles(file(filename));
      const response = await uploaded;
      assert.equal(response.status(), expected, await response.text());
      // Browser session writes must not embed a management token in uploads.
      assert.equal(response.request().headers().authorization, undefined);
      assert.equal(new URL(response.url()).search, '');
      return response.json();
    }
    let releaseUpload;
    const uploadGate = new Promise(resolve => { releaseUpload = resolve; });
    let receivedUpload;
    const received = new Promise(resolve => { receivedUpload = resolve; });
    await page.route('**/api/files/upload', async route => {
      const response = await route.fetch();
      receivedUpload();
      await uploadGate;
      await route.fulfill({ response });
    }, { times: 1 });
    const pendingUpload = choose(page, 'Choose model artifacts 1', 'weights.bin');
    pendingUpload.catch(() => {});
    let model;
    try {
      await received;
      assert.equal(await page.getByRole('button', { name: 'Remove model artifacts 1', exact: true }).isDisabled(), true,
        'A pending upload must not be attached to a different array item after removal');
    } finally {
      releaseUpload();
      model = (await pendingUpload).result;
    }
    const vocab = (await choose(page, 'Choose vocab artifacts.v1', 'vocab.zip')).result;
    await choose(page, 'Choose vocab artifacts.v1', 'bad.zip', 422);
    await page.getByRole('alert').filter({ hasText: /ZIP|archive/i }).waitFor();
    await page.screenshot({ path: file('resources-selected.png') });
    async function save() {
      await page.getByRole('button', { name: 'Save', exact: true }).click();
      await page.locator('.semi-sidesheet:visible input').fill('model-resources');
      const saved = page.waitForResponse(response => response.request().method() === 'POST' &&
        new URL(response.url()).pathname === '/api/workflow/save');
      await page.getByRole('button', { name: 'confirm', exact: true }).click();
      const response = await saved;
      assert.equal(response.status(), 200, await response.text());
      await page.locator('.semi-sidesheet:visible').waitFor({ state: 'hidden' });
      return response.request().postDataJSON().businessContent;
    }
    const saved = await save();
    const resource = (graph, name) => graph.node_repository_.find(node => node.name_ === name)[name === 'vocab' ? 'artifacts.v1' : 'artifacts'];
    assert.deepEqual(resource(saved, 'model'), [{ file_id: model.file_id }]);
    assert.deepEqual(resource(saved, 'vocab'), { file_id: vocab.file_id });
    await page.reload();
    await page.getByText('Uploaded model file', { exact: true }).waitFor();
    await page.getByText('Uploaded model directory', { exact: true }).waitFor();
    async function run(expected, container = page) {
      const accepted = page.waitForResponse(response => response.request().method() === 'POST' &&
        new URL(response.url()).pathname === '/api/queue');
      accepted.catch(() => {});
      await container.getByRole('button', { name: 'Run', exact: true }).click();
      const response = await accepted;
      assert.equal(response.status(), 202, await response.text());
      const task = (await response.json()).result.task_id;
      let result;
      await until(async () => {
        result = (await (await api.get(`/api/queue/task/${task}`)).json()).result;
        return ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(result.state);
      }, 'Model workflow did not finish');
      assert.equal(result.state, 'SUCCEEDED', JSON.stringify(result));
      assert.deepEqual(result.outputs, { sink_model: expected, sink_vocab: 'tokens' });
      await container.getByRole('button', { name: 'Run', exact: true }).waitFor();
      return task;
    }
    const first = await run('weights');
    await page.getByRole('button', { name: 'Webui', exact: true }).click();
    const sheet = page.locator('.semi-sidesheet:visible');
    const replacement = (await choose(sheet, 'Choose model artifacts 1', 'replacement.bin')).result;
    await sheet.getByText('replacement.bin', { exact: true }).waitFor();
    const second = await run('updated', sheet);
    await page.locator('.semi-sidesheet-footer').getByRole('button', { name: 'confirm', exact: true }).click();
    await sheet.waitFor({ state: 'hidden' });
    await page.getByRole('button', { name: 'Server', exact: true }).click();
    const started = page.waitForResponse(response => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === '/api/workflow/server/start');
    await page.getByRole('button', { name: 'Start', exact: true }).click();
    const startResponse = await started;
    assert.equal(startResponse.status(), 200, await startResponse.text());
    const info = await startResponse.json();
    await page.getByText('Status: ready', { exact: true }).waitFor();
    const deployed = await api.post(info.deploy_path, { data: {} });
    assert.equal(deployed.status(), 200, await deployed.text());
    assert.deepEqual((await deployed.json()).outputs, { sink_model: 'updated', sink_vocab: 'tokens' });
    await page.screenshot({ path: file('resources-server.png') });
    await page.getByRole('button', { name: 'Stop', exact: true }).click();
    await page.getByText('Status: stopped', { exact: true }).waitFor();
    assert.deepEqual(errors, []);
    fs.writeFileSync(file('evidence.json'), JSON.stringify({ workflow_id: identity, tasks: [first, second],
      files: [model.file_id, vocab.file_id, replacement.file_id], failed_upload_preserved_selection: true,
      original_field_shapes_preserved: true, reload_preserved_references: true,
      canvas_run: 'weights', webui_run: 'updated', deploy_outputs: (await deployed.json()).outputs,
      cookie_upload_without_token: true, stopped: true, page_errors: errors }, null, 2));
    console.log('PASS: browser session, model/ZIP upload, invalid ZIP, Save/reload, Run, Webui replacement, Start/Deploy/Stop');
  } catch (error) {
    await page.screenshot({ path: file('resources-failure.png') });
    fs.writeFileSync(file('resources-failure.txt'), await page.locator('body').innerText());
    throw error;
  } finally {
    await browser.close();
    await api.dispose();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

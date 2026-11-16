// Run against launch_reconnect.py using a fresh NNDEPLOY_BROWSER_ROOT.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, request } = require(process.env.NNDEPLOY_PLAYWRIGHT || 'playwright');
const state = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const base = `http://127.0.0.1:${state.port}`;
const file = name => path.join(state.root, name);
const flag = name => fs.writeFileSync(file(name), '');
const remove = name => fs.rmSync(file(name), { force: true });
const posts = () => fs.readFileSync(file('posts.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const executions = () => fs.existsSync(file('executions.txt')) ? fs.readFileSync(file('executions.txt'), 'utf8').trim().split('\n').length : 0;
async function until(check, description) {
  const deadline = Date.now() + 20000;
  while (!(await check())) {
    if (Date.now() > deadline) throw new Error(description);
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}

(async () => {
  assert(!fs.existsSync(file('posts.jsonl')), 'Use a fresh browser test directory');
  const api = await request.newContext({ baseURL: base });
  const browser = await chromium.launch({ headless: true,
    executablePath: process.env.NNDEPLOY_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  const businessErrors = [];
  let connections = 0;
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', async response => {
    if (!response.url().includes('/api/')) return;
    try {
      const value = await response.json();
      if (value.flag === 'error') businessErrors.push({ url: response.url(),
        message: value.message, error: value.error });
    } catch { /* Expected transport failures need not contain JSON. */ }
  });
  await page.addInitScript(() => {
    window.emptyToasts = [];
    const seen = new WeakSet();
    new MutationObserver(() => {
      document.querySelectorAll('.semi-toast-content-text').forEach(node => {
        if (!node.textContent.trim() && !seen.has(node)) {
          seen.add(node);
          window.emptyToasts.push('An empty notification was shown');
        }
      });
    }).observe(document, { childList: true, subtree: true });
  });
  page.on('websocket', socket => { if (socket.url().includes('/ws/progress')) connections++; });
  try {
    const response = await api.post('/api/workflow/save', { data: {
      businessContent: JSON.parse(fs.readFileSync(file('graph.json'), 'utf8')),
    } });
    assert.equal(response.status(), 200, await response.text());
    const identity = (await response.json()).result.id;
    await page.goto(`${base}/design?id=${identity}&name=browser-reconnect&flowType=workspace`);
    const run = () => page.getByRole('button', { name: 'Run', exact: true });
    await run().waitFor();
    flag('hold');
    await run().click();
    await until(() => executions() === 1, 'First Graph did not run');
    await until(() => connections > 0, 'Task WebSocket was not created');
    const before = connections;
    const disconnected = await api.post('/__test__/disconnect');
    assert((await disconnected.json()).closed > 0, 'No real socket was disconnected');
    await until(() => connections > before, 'Task WebSocket did not reconnect');
    remove('hold');
    await run().waitFor({ timeout: 20000 });
    assert.equal(posts().length, 1, 'Reconnect resubmitted the task');
    const first = posts()[0].key;
    let result = (await (await api.get(`/api/queue/task/${first}`)).json()).result;
    assert.equal(result.state, 'SUCCEEDED');
    assert.equal(result.output_files.length, 1);
    assert.equal((await api.get(result.output_files[0].url)).status(), 200);
    await page.screenshot({ path: file('reconnected.png') });

    // Lose the accepted POST response and mask status until the browser reloads.
    flag('hold'); flag('drop-after'); flag('hide-status');
    await run().click();
    await until(() => executions() === 2, 'Second Graph did not run');
    await page.getByRole('button', { name: 'Retry', exact: true }).waitFor();
    const original = posts()[1];
    await api.post('/__test__/disconnect');
    await page.reload();
    await page.getByRole('button', { name: 'Retry', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Retry', exact: true }).click();
    await until(() => posts().length === 3, 'Explicit retry was not sent');
    assert.deepEqual(posts()[2], original, 'Refresh lost the original task or workflow identity');
    assert(original.query.includes(`workflow_id=${identity}`));
    remove('hide-status'); remove('hold');
    await run().waitFor({ timeout: 20000 });
    result = (await (await api.get(`/api/queue/task/${original.key}`)).json()).result;
    assert.equal(result.state, 'SUCCEEDED');
    assert.equal(executions(), 2, 'Retry executed the Graph again');
    assert.equal(posts().length, 3);
    await page.getByRole('button', { name: 'Log', exact: true }).click();
    await page.getByRole('tab', { name: 'History', exact: true }).click();
    await page.getByRole('button', { name: new RegExp(original.key) }).click();
    const results = page.getByRole('region', { name: 'Task results' });
    await results.getByRole('heading', { name: 'SUCCEEDED', exact: true }).waitFor();
    await until(() => results.locator('img').evaluateAll(images => images.length === 1 && images.every(image => image.complete && image.naturalWidth > 0)), 'Saved image did not render');
    await page.screenshot({ path: file('restored-history.png') });
    assert.deepEqual(errors, []);
    assert.deepEqual(await page.evaluate(() => window.emptyToasts), [], JSON.stringify(businessErrors));

    // Webui must wait for actual cancellation, just like the canvas toolbar.
    await page.locator('.semi-sidesheet:visible .semi-sidesheet-close').click();
    await page.getByRole('button', { name: 'Webui', exact: true }).click();
    const webui = page.locator('.semi-sidesheet:visible').filter({ has: page.getByRole('button', { name: 'confirm', exact: true }) });
    flag('hold');
    await webui.getByRole('button', { name: 'Run', exact: true }).click();
    await until(() => executions() === 3, 'Webui Graph did not run');
    const cancelled = posts()[3].key;
    await webui.getByRole('button', { name: 'Stop', exact: true }).click();
    await webui.getByRole('button', { name: 'Stopping…', exact: true }).waitFor({ timeout: 3000 });
    result = (await (await api.get(`/api/queue/task/${cancelled}`)).json()).result;
    assert.equal(result.cancel_requested, true);
    assert.equal(result.state, 'RUNNING');
    assert.equal(await webui.getByRole('button', { name: 'Run', exact: true }).count(), 0);
    remove('hold');
    await webui.getByRole('button', { name: 'Run', exact: true }).waitFor({ timeout: 20000 });
    result = (await (await api.get(`/api/queue/task/${cancelled}`)).json()).result;
    assert.equal(result.state, 'CANCELLED');
    assert.equal(result.output_files.length, 0);
    assert.equal(executions(), 3);
    await page.screenshot({ path: file('webui-cancelled.png') });
    assert.deepEqual(errors, []);
    assert.deepEqual(await page.evaluate(() => window.emptyToasts), []);
    fs.writeFileSync(file('evidence.json'), JSON.stringify({ workflow_id: identity,
      task_ids: [first, original.key, cancelled], websocket_connections: connections, intentional_runs: 3,
      actual_executions: executions(), posts: posts(), refreshed_pending_retry: true,
      saved_image_rendered: true, webui_cancellation_waits_for_execution: true, page_errors: errors }, null, 2));
    console.log('PASS: real WebSocket reconnect, pending refresh/retry, no duplicate execution, saved image history, Webui cancellation');
  } catch (error) {
    await page.screenshot({ path: file('failure.png') });
    fs.writeFileSync(file('failure.txt'), await page.locator('body').innerText());
    fs.writeFileSync(file('failure-responses.json'), JSON.stringify(businessErrors, null, 2));
    throw error;
  } finally {
    remove('hold'); remove('hide-status');
    await browser.close();
    await api.dispose();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

// Two nested Graphs intentionally share a leaf name, but have distinct results.
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
  const api = await request.newContext({ baseURL: base });
  const browser = await chromium.launch({ headless: true,
    executablePath: process.env.NNDEPLOY_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    const response = await api.post('/api/workflow/save', { data: {
      businessContent: JSON.parse(fs.readFileSync(file('graph.json'), 'utf8')),
    } });
    assert.equal(response.status(), 200, await response.text());
    const identity = (await response.json()).result.id;
    await page.goto(`${base}/design?id=${identity}&name=nested-previews&flowType=workspace`);
    await page.getByText('left', { exact: true }).waitFor({ timeout: 3000 });
    await page.getByText('right', { exact: true }).waitFor({ timeout: 3000 });
    async function visibleLayout() {
      return page.locator('.gedit-flow-activity-node').evaluateAll(nodes => {
        const viewport = document.querySelector('.demo-container').getBoundingClientRect();
        return nodes.every(node => {
          const box = node.getBoundingClientRect();
          return box.left >= viewport.left && box.right <= viewport.right &&
            box.top >= viewport.top && box.bottom <= viewport.bottom;
        });
      });
    }
    await until(visibleLayout, 'Loaded Graph is outside the editor viewport');
    await page.screenshot({ path: file('nested-before.png') });
    const accepted = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/queue');
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const submitted = await accepted;
    assert.equal(submitted.status(), 202, await submitted.text());
    const task = (await submitted.json()).result.task_id;
    let result;
    await until(async () => {
      result = (await (await api.get(`/api/queue/task/${task}`)).json()).result;
      return ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(result.state);
    }, 'Nested Graph did not finish');
    assert.equal(result.state, 'SUCCEEDED', JSON.stringify(result));
    assert.deepEqual(result.output_files.map(file => file.node_path).sort(), [['left', 'encode'], ['right', 'encode']]);
    await page.getByRole('button', { name: 'Run', exact: true }).waitFor();

    async function rendered(container, output) {
      const image = container.locator(`img[src="${output.url}"]:visible`);
      await until(async () => await image.count() > 0 && await image.first().evaluate(img => img.complete && img.naturalWidth > 0), 'Missing nested preview: ' + output.node_path.join('/'));
      const pixel = await image.first().evaluate(img => {
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = 1;
        const context = canvas.getContext('2d');
        context.drawImage(img, 0, 0);
        return [...context.getImageData(0, 0, 1, 1).data];
      });
      assert.deepEqual(pixel, output.node_path[0] === 'left' ? [20, 140, 220, 255] : [220, 80, 20, 255]);
    }
    for (const output of result.output_files) await rendered(page, output);
    // Output previews change node dimensions. The existing explicit layout
    // action must account for those dimensions without hiding either branch.
    await page.getByRole('button', { name: 'Auto Layout', exact: true }).click();
    async function groupBox(name) {
      return page.getByText(name, { exact: true }).evaluate(element => {
        const box = element.closest('.gedit-flow-activity-node').getBoundingClientRect();
        return { left: box.left, right: box.right, top: box.top, bottom: box.bottom };
      });
    }
    async function separated() {
      const left = await groupBox('left'), right = await groupBox('right');
      return left.right <= right.left || right.right <= left.left ||
        left.bottom <= right.top || right.bottom <= left.top;
    }
    await until(async () => await visibleLayout() && await separated(),
      'Expanded Graph previews overlap or remain outside the editor viewport');
    await page.screenshot({ path: file('nested-canvas.png') });
    await page.getByRole('button', { name: 'Webui', exact: true }).click();
    const outputs = page.getByLabel('Output files', { exact: true });
    for (const output of result.output_files) {
      await outputs.getByRole('link', { name: output.node_path.join(' / ') + ' / path_', exact: true }).waitFor();
      await rendered(outputs, output);
    }
    await page.screenshot({ path: file('nested-webui.png') });
    await page.locator('.semi-sidesheet-footer').getByRole('button', { name: 'close', exact: true }).click();
    async function save() {
      await page.getByRole('button', { name: 'Save', exact: true }).click();
      await page.locator('.semi-sidesheet:visible input').fill('nested-previews');
      const saved = page.waitForResponse(response => response.request().method() === 'POST' &&
        new URL(response.url()).pathname === '/api/workflow/save');
      await page.getByRole('button', { name: 'confirm', exact: true }).click();
      const response = await saved;
      assert.equal(response.status(), 200, await response.text());
      assert.equal((await response.json()).result.id, identity);
      await page.locator('.semi-sidesheet:visible').waitFor({ state: 'hidden' });
      return response.request().postDataJSON().businessContent;
    }
    function positions(layout) {
      return Object.fromEntries(Object.entries(layout).map(([name, node]) => [name, {
        position: node.position, children: node.children ? positions(node.children) : {},
      }]));
    }
    const saved = await save();
    await page.reload();
    await page.getByText('left', { exact: true }).waitFor();
    await until(visibleLayout, 'Saved Graph is outside the editor viewport');
    const reloaded = await save();
    assert.deepEqual(positions(reloaded.nndeploy_ui_layout.layout),
      positions(saved.nndeploy_ui_layout.layout), 'Reload rearranged saved node positions');
    const rerunAccepted = page.waitForResponse(response => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === '/api/queue');
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const rerunResponse = await rerunAccepted;
    assert.equal(rerunResponse.status(), 202, await rerunResponse.text());
    const rerunId = (await rerunResponse.json()).result.task_id;
    let rerun;
    await until(async () => {
      rerun = (await (await api.get(`/api/queue/task/${rerunId}`)).json()).result;
      return ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(rerun.state);
    }, 'Saved nested Graph did not finish');
    assert.equal(rerun.state, 'SUCCEEDED', JSON.stringify(rerun));
    assert.deepEqual(rerun.output_files.map(file => file.node_path).sort(), [['left', 'encode'], ['right', 'encode']]);
    for (const output of rerun.output_files) await rendered(page, output);
    // Historical files survive deletion of both source paths and a page reload.
    for (const name of ['left.png', 'right.png']) fs.unlinkSync(file('resources/' + name));
    await page.reload();
    await page.getByRole('button', { name: 'Log', exact: true }).click();
    await page.getByRole('tab', { name: 'History', exact: true }).click();
    await page.getByRole('button', { name: new RegExp(task) }).click();
    const history = page.getByRole('region', { name: 'Task results' });
    for (const output of result.output_files) await rendered(history, output);
    await page.screenshot({ path: file('nested-history.png') });
    assert.deepEqual(errors, []);
    fs.writeFileSync(file('evidence.json'), JSON.stringify({ workflow_id: identity, task_id: task,
      output_paths: result.output_files.map(file => file.node_path),
      loaded_layout_visible: true, expanded_layout_visible_and_separated: true,
      saved_positions_preserved: true, saved_graph_rerun: rerunId,
      canvas_pixels_verified: true, webui_pixels_verified: true,
      history_after_source_removal: true, page_errors: errors }, null, 2));
    console.log('PASS: nested Graph viewport/layout, saved positions and rerun, distinct canvas/Webui/history pixels, source removal');
  } catch (error) {
    await page.screenshot({ path: file('failure.png') });
    fs.writeFileSync(file('failure.txt'), await page.locator('body').innerText());
    throw error;
  } finally {
    await browser.close();
    await api.dispose();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

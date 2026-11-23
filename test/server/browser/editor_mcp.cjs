// Real Server panel edits against the disposable editor runtime.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium, request } = require(process.env.NNDEPLOY_PLAYWRIGHT || 'playwright');
const state = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const base = `http://127.0.0.1:${state.port}`;
(async () => {
  const api = await request.newContext({ baseURL: base, extraHTTPHeaders: { Authorization: 'Bearer browser-resource-test' } });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.NNDEPLOY_CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  const context = await browser.newContext({ viewport: { width: 1500, height: 1100 }, permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    const graph = JSON.parse(fs.readFileSync(path.join(state.root, 'graph.json'), 'utf8'));
    // Use a built-in encoder with Server output discovery, not the recovery fixture's custom node.
    graph.node_repository_[1].key_ = 'nndeploy.codec.PILImageEncodec';
    const created = await api.post('/api/workflow/save', { data: { businessContent: graph } });
    assert.equal(created.status(), 200, await created.text());
    const identity = (await created.json()).result.id;
    await page.goto(`${base}/design?id=${identity}&name=browser-reconnect&flowType=workspace`);
    await page.getByLabel('Management token', { exact: true }).fill('browser-resource-test');
    await page.getByRole('button', { name: 'Sign in', exact: true }).click();
    await page.getByRole('button', { name: 'Sign out', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Server', exact: true }).click();
    const panel = page.locator('.server-panel');
    await panel.getByLabel('Enable tool export', { exact: true }).check();
    await panel.getByLabel('Tool name', { exact: true }).fill('browser_image');
    await panel.getByLabel('Tool description', { exact: true }).fill('Decode and encode an image workflow.');
    assert.equal(await panel.getByLabel('MCP connection URL', { exact: true }).inputValue(), base + '/mcp');
    await panel.getByRole('button', { name: 'Copy MCP URL', exact: true }).click();
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), base + '/mcp');
    async function apply(label) {
      const response = page.waitForResponse(r => r.request().method() === 'POST' && new URL(r.url()).pathname === '/api/workflow/server/start');
      await panel.getByRole('button', { name: label, exact: true }).click();
      const result = await response;
      assert.equal(result.status(), 200, await result.text());
      return result.json();
    }
    const started = await apply('Start');
    assert.equal(started.mcp.name, 'browser_image');
    await panel.getByRole('button', { name: 'Update', exact: true }).waitFor();
    await page.screenshot({ path: path.join(state.root, 'mcp-panel.png'), fullPage: true });
    await page.reload();
    await page.getByRole('button', { name: 'Server', exact: true }).click();
    await panel.getByLabel('Tool name', { exact: true }).waitFor();
    assert.equal(await panel.getByLabel('Tool name', { exact: true }).inputValue(), 'browser_image');
    await panel.getByRole('button', { name: 'Refresh', exact: true }).click();
    await panel.getByRole('button', { name: 'Update', exact: true }).waitFor();
    await panel.getByLabel('Tool name', { exact: true }).fill('browser_image_updated');
    assert.equal((await apply('Update')).mcp.name, 'browser_image_updated');
    await panel.getByLabel('Enable tool export', { exact: true }).uncheck();
    assert.equal((await apply('Update')).mcp, null);
    assert.deepEqual(errors, []);
    console.log('PASS: enable, edit, copy MCP URL, refresh restoration and disable in real Chrome');
  } finally {
    await browser.close();
    await api.dispose();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

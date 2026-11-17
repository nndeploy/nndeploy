import './server.scss';
import { useState } from 'react';
import { Button, Input, TextArea, SideSheet, Space, Toast } from '@douyinfe/semi-ui';
import { getNodeForm, useClientContext } from '@flowgram.ai/free-layout-editor';
import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';
import { useEditorSession } from '../editor-access';

import { serverRequest, loadMcpAvailability, mcpValidation, mcpConnectionUrl, type MCPToolConfig, type ServerField, type ServerInfo } from './server-request';
export { serverRequest, type ServerInfo } from './server-request';

function exampleValue(field: ServerField, localPath?: unknown) {
  if (!field.required) return field.default;
  if (field.type === 'image' || field.type === 'file' || field.media_type === 'application/x-npy' || field.sample_rate) {
    return { path: typeof localPath === 'string' && localPath ? localPath : '<local file path>' };
  }
  if (field.type === 'array' || field.type === 'tensor') {
    const shape = (field.shape || []).map(size => size ?? 1);
    const count = shape.reduce((product, size) => product * size, 1);
    return { dtype: field.dtype, shape, ...(field.data_format ? { data_format: field.data_format } : {}), data: count <= 16
      ? Array.from({ length: count }, () => field.dtype === 'bool' ? false : 0)
      : [`<${count} flat ${field.dtype} values>`] };
  }
  return field.type === 'integer' || field.type === 'number' ? 0 : field.type === 'boolean' ? false : '<value>';
}

function parameterPaths(value: Record<string, unknown>, prefix: string[] = []): string[][] {
  return Object.entries(value).flatMap(([key, item]) => {
    const path = [...prefix, key];
    return item && typeof item === 'object' && !Array.isArray(item) && Object.keys(item).length
      ? parameterPaths(item as Record<string, unknown>, path) : [path];
  });
}

export function ServerTool({ disabled }: { disabled: boolean }) {
  const editor = useClientContext();
  const environment = useFlowEnviromentContext();
  const session = useEditorSession();
  const [visible, setVisible] = useState(false);
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState<ServerInfo>();
  const [name, setName] = useState('');
  const [callbackUrl, setCallbackUrl] = useState('');
  const [mcpEnabled, setMcpEnabled] = useState(false);
  const [mcpName, setMcpName] = useState('');
  const [mcpDescription, setMcpDescription] = useState('');
  const [mcpPath, setMcpPath] = useState<string | null>(null);
  const [mcpLoaded, setMcpLoaded] = useState(false);
  const [mcpLoadError, setMcpLoadError] = useState('');
  const mcpError = mcpValidation(mcpEnabled, mcpName, mcpDescription);
  const mcpUrl = mcpConnectionUrl(mcpPath, window.location.origin);
  const suggestedMcpName = `run_${name.trim().replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '') || 'workflow'}`.slice(0, 128);
  const suggestedMcpDescription = `Run the ${name.trim().slice(0, 128) || 'current'} workflow and return its results.`;


  function applyMcp(tool?: MCPToolConfig | null) {
    setMcpEnabled(Boolean(tool));
    setMcpName(tool?.name || '');
    setMcpDescription(tool?.description || '');
  }

  async function loadMcp() {
    try {
      const value = await loadMcpAvailability(token);
      setMcpPath(value.enabled ? value.path : null);
      setMcpLoaded(true);
      setMcpLoadError('');
    } catch (error) {
      setMcpPath(null);
      setMcpLoaded(false);
      const message = error instanceof Error ? error.message : 'Unable to load MCP availability';
      setMcpLoadError(message);
      Toast.error(message);
    }
  }

  let callbackError = '';
  if (callbackUrl.trim()) {
    try {
      const url = new URL(callbackUrl.trim());
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.hash || callbackUrl.trim().length > 2048) throw new Error();
    } catch { callbackError = 'Enter an HTTP(S) URL without credentials or a fragment.'; }
  }
  const [fields, setFields] = useState<Record<string, {default: Record<string, unknown>; exposed: string[][]}>>();

  const startLabel = info?.status === 'ready' ? 'Update' : 'Start';
  const startDisabledReason = busy ? 'Please wait for the current operation to finish.'
    : !token && !session ? 'Sign in or enter a management token.'
    : !name.trim() ? 'Enter a Server name.'
    : !fields ? 'Load the public request fields first.'
    : mcpError || (mcpEnabled && !mcpPath
      ? 'MCP is unavailable. Refresh MCP availability, or disable tool export to continue with HTTP only.' : '');

  async function loadParameters() {
    setBusy(true);
    try {
      setFields(await environment.onServerParameters?.(editor.document.toJSON() as any, token));
    } catch (error) { Toast.error(error instanceof Error ? error.message : 'Unable to load parameters'); }
    finally { setBusy(false); }
  }

  async function open() {
    setName(info?.name || environment.serverName || '');
    setVisible(true);
    await loadMcp();
    if (!info && environment.workflowId) {
      setBusy(true);
      try {
        const response = await fetch(`/api/workflow/${encodeURIComponent(environment.workflowId)}/server`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!response.ok) throw new Error('Unable to load Server name');
        const value = await response.json();
        if (value.name) setName(value.name);
        applyMcp(value.mcp);
      } catch (error) {
        Toast.error(error instanceof Error ? error.message : 'Unable to load Server name');
      } finally { setBusy(false); }
    }
    await loadParameters();
  }

  async function action(kind: 'start' | 'stop' | 'refresh') {
    setBusy(true);
    try {
      if (kind === 'start') {
        const forms = editor.document.getAllNodes().map(getNodeForm);
        await Promise.all(forms.map(form => form?.validate()));
        if (forms.some(form => form?.state.invalid)) throw new Error('Fix the highlighted node fields first.');
        const result = await environment.onStartServer?.(editor.document.toJSON() as any, token, name.trim(),
          Object.fromEntries(Object.entries(fields || {}).map(([node, value]) => [node, value.exposed])),
          mcpEnabled ? { name: mcpName.trim(), description: mcpDescription.trim() } : null);
        if (result) { setInfo(result); setName(result.name); applyMcp(result.mcp); }
      } else {
        const activeName = info?.name || name;
        if (!activeName) throw new Error('Start this workflow first.');
        const result = await serverRequest(`/api/server/${encodeURIComponent(activeName)}${kind === 'stop' ? '/stop' : ''}`, token,
          kind === 'stop' ? {} : undefined);
        setInfo(result);
        applyMcp(result.mcp);
        await loadMcp();
      }
    } catch (error) {
      Toast.error(error instanceof Error ? error.message : 'Server request failed');
    } finally {
      setBusy(false);
    }
  }

  const inputs = Object.fromEntries(Object.entries(info?.schema.inputs || {}).map(([name, field]) => {
    const form = editor.document.getAllNodes().map(getNodeForm).find(form => form?.getValueIn('name_') === name);
    return [name, exampleValue(field, form?.getValueIn('path_'))];
  }));
  const body: Record<string, unknown> = Object.fromEntries(Object.entries(info?.schema.parameters || {}).map(([name, field]) => [name, field.default]));
  for (const [name, value] of Object.entries(inputs)) {
    body[name] = body[name] && typeof body[name] === 'object'
      ? {...body[name] as Record<string, unknown>, ...(value && typeof value === 'object' && 'path' in value
          ? {path_: value.path} : {str_: value})} : value;
  }
  if (callbackUrl.trim() && !callbackError) {
    body.mode = 'async';
    body.callback_url = callbackUrl.trim();
  }
  const shellQuote = (value: string) => "'" + value.replace(/'/g, "'\"'\"'") + "'";
  const example = info && !callbackError && `curl -X POST ${shellQuote(window.location.origin + info.deploy_path)} \\\n  -H 'Content-Type: application/json' \\\n  -d ${shellQuote(JSON.stringify(body, null, 2))}`;

  return <>
    <Button disabled={disabled} onClick={open}
      style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}>Server</Button>
    <SideSheet title="Deploy workflow" visible={visible} onCancel={() => setVisible(false)} width="min(720px, 100vw)" className="server-panel">
      <div className="server-sections">
        <section className="server-section">
        <h3>1. Service configuration</h3>
        <p>Start this workflow as an HTTP endpoint. Changes take effect when you select Start or Update.</p>
        <label htmlFor="server-name">Server name</label>
        <Input id="server-name" value={name} onChange={setName} disabled={busy}
          placeholder="Workflow name" />
        {!session && <><label htmlFor="server-admin-token">Management token</label>
        <Input id="server-admin-token" mode="password" value={token} onChange={setToken}
          placeholder="Token configured when starting the server" autoComplete="off" /></>}
        </section>
        <section className="server-section">
          <h3>2. Public request fields</h3>
          <p>Select the node fields callers may change. Unselected fields use the workflow values.</p>
          <Button disabled={busy} onClick={loadParameters}>Reload fields</Button>
          {Object.entries(fields || {}).map(([node, value]) => <details key={node}>
            <summary>{node} ({value.exposed.length} selected)</summary>
            {parameterPaths(value.default).map(path => {
              const key = JSON.stringify(path);
              const checked = value.exposed.some(item => JSON.stringify(item) === key);
              return <label key={key} style={{ display: 'block', margin: '6px 0' }}>
                <input type="checkbox" checked={checked} disabled={busy} onChange={event => {
                  const exposed = event.target.checked ? [...value.exposed, path]
                    : value.exposed.filter(item => JSON.stringify(item) !== key);
                  setFields(previous => ({...previous, [node]: {...value, exposed}}));
                }} /> {path.join('.')}
              </label>;
            })}
          </details>)}
          {!fields && <p role="status">Load node fields to configure public parameters.</p>}
          {fields && !Object.keys(fields).length && <p>No configurable node parameters.</p>}
        </section>
        <section className="server-section" aria-label="MCP configuration">
          <h3>3. MCP tool export</h3>
          <p>HTTP and MCP can be used at the same time. Enabling MCP adds a tool for this workflow; the HTTP endpoint remains available.</p>
          <p>Tool name and description are required when export is enabled. Select Start or Update to apply the configuration. Editing these fields does not stop a running HTTP service.</p>
          <label htmlFor="server-mcp-enabled">
            <input id="server-mcp-enabled" type="checkbox" checked={mcpEnabled}
              disabled={busy || (!mcpPath && !mcpEnabled)} onChange={event => setMcpEnabled(event.target.checked)} /> Enable tool export
          </label>
          {mcpLoadError && <p role="alert">{mcpLoadError}</p>}
          {!mcpLoaded && !mcpLoadError && <p role="status">Load MCP availability after signing in or entering a management token.</p>}
          {mcpLoaded && !mcpPath && <p role="status">MCP is disabled on this server. Install nndeploy[server-mcp] and restart the editor with --mcp.</p>}
          <Button disabled={busy} onClick={loadMcp}>Refresh MCP availability</Button>
          {mcpEnabled && <>
            <label htmlFor="server-mcp-name">Tool name <span className="server-required" aria-hidden="true">*</span> <span aria-hidden="true">(required)</span></label>
            <Input aria-label="Tool name" aria-required="true" aria-describedby="server-mcp-tab-hint server-mcp-validation" id="server-mcp-name" value={mcpName} onChange={setMcpName} disabled={busy}
              placeholder={suggestedMcpName} maxLength={128} onKeyDown={event => {
                if (event.key === 'Tab' && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey && !event.nativeEvent.isComposing && !mcpName.trim()) {
                  setMcpName(suggestedMcpName);
                }
              }} />
            <label htmlFor="server-mcp-description">Tool description <span className="server-required" aria-hidden="true">*</span> <span aria-hidden="true">(required)</span></label>
            <TextArea aria-label="Tool description" aria-required="true" aria-describedby="server-mcp-tab-hint server-mcp-validation" id="server-mcp-description" value={mcpDescription} onChange={setMcpDescription}
              disabled={busy} placeholder={suggestedMcpDescription} maxLength={4096} onKeyDown={event => {
                if (event.key === 'Tab' && !event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey && !event.nativeEvent.isComposing && !mcpDescription.trim()) {
                  setMcpDescription(suggestedMcpDescription);
                }
              }} />
            <p id="server-mcp-tab-hint">Required fields are marked with *. In an empty field, press Tab to accept the suggested value and move to the next field. Existing text is kept.</p>
            <p id="server-mcp-validation" role="status">{mcpError || 'MCP fields are complete. Select Start or Update to apply.'}</p>
          </>}
          {mcpUrl && <>
            <label htmlFor="server-mcp-url">MCP connection URL</label>
            <Input id="server-mcp-url" value={mcpUrl} readOnly />
            <Button onClick={async () => {
              try { await navigator.clipboard.writeText(mcpUrl); Toast.success('MCP URL copied'); }
              catch { Toast.error('Unable to copy. Select the URL above to copy it manually.'); }
            }}>Copy MCP URL</Button>
            <p>Connect using Streamable HTTP. All exported tools share this address. Configure the caller’s Bearer token in the MCP client if authentication is enabled.</p>
            <p>{info?.mcp ? `Applied tool: ${info.mcp.name}. Service status: ${info.status}.` : info ? 'This workflow has no applied tool export.' : 'Select Refresh to check the running service status.'}</p>
          </>}
        </section>
        <section className="server-section">
          <h3><label htmlFor="server-callback">4. Completion callback URL (optional)</label></h3>
          <Input id="server-callback" value={callbackUrl} onChange={setCallbackUrl}
            placeholder="https://your-app.example.com/callback" validateStatus={callbackError ? 'error' : undefined} />
          {callbackError && <p role="alert">{callbackError}</p>}
          <p>The generated request passes this URL as <code>callback_url</code> with <code>mode: async</code>.
            The server sends a notification when the task finishes. This is a per-request setting; your receiver should return HTTP 2xx.</p>
        </section>
        <div className="server-actions">
        <Space>
          <Button theme="solid" loading={busy} disabled={!!startDisabledReason} aria-describedby={startDisabledReason ? 'server-start-reason' : undefined} onClick={() => action('start')}>
            {startLabel}
          </Button>
          <Button disabled={busy || (!token && !session) || !info || info.status !== 'ready'} onClick={() => action('stop')}>Stop</Button>
          <Button disabled={busy || (!token && !session) || (!info && !name)} onClick={() => action('refresh')}>Refresh</Button>
        </Space>
        {startDisabledReason && <p id="server-start-reason" role="status" className="server-action-hint">
          {startLabel} unavailable: {startDisabledReason}
        </p>}
        </div>
        {info && <section className="server-section">
          <h3>5. Endpoint & request example</h3>
          <p>Status: {info.status}</p>
          <p>Endpoint: <code>{info.deploy_path}</code></p>
          {['inputs', 'parameters', 'outputs'].map(group => <div key={group}>
            <strong>{group[0].toUpperCase() + group.slice(1)}</strong>
            <ul>{Object.entries(info.schema[group] || {}).map(([name, field]) =>
              <li key={name}>{name}: {field.type}{field.required ? ' (required)' : ''}
                {group === 'parameters' && field.default !== undefined && <details>
                  <summary>Node parameters</summary>
                  <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(field.default, null, 2)}</pre>
                </details>}
                {field.media_type && <> — {field.media_type}</>}
                {field.sample_rate && <> at {field.sample_rate} Hz</>}
                {field.dtype && <> — {field.dtype}, shape [{(field.shape || []).map(size => size ?? 'N').join(', ')}],
                  {field.data_format && <> format {field.data_format},</>}
                  at most {field.max_elements} elements / {field.max_bytes} bytes</>}
              </li>)}</ul>
          </div>)}
          <p>Pass a local file path directly, absolute or relative to the Server working directory.
            Uploaded file IDs are also supported. Use the declared NumPy or audio format for numeric file fields.</p>
          {Object.keys(info.schema.parameters || {}).length > 0 &&
            <p>Pass nodes directly, for example <code>{'{"node_name": {"param_": {"field": "value"}}}'}</code>.
              Use the fields shown above. Omitted values keep the workflow defaults; overrides apply only to this request.</p>}
          <Button disabled={!example} onClick={async () => {
            try { await navigator.clipboard.writeText(example || ''); Toast.success('Request copied'); }
            catch { Toast.error('Unable to copy. Select the request below to copy it manually.'); }
          }}>Copy request</Button>
          <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxWidth: '100%' }}>{example}</pre>
        </section>}
      </div>
    </SideSheet>
  </>;
}

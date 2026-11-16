import { useState } from 'react';
import { Button, Input, SideSheet, Space, Toast } from '@douyinfe/semi-ui';
import { getNodeForm, useClientContext } from '@flowgram.ai/free-layout-editor';
import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';
import { useEditorSession } from '../editor-access';

import { serverRequest, type ServerField, type ServerInfo } from './server-request';
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
  const [fields, setFields] = useState<Record<string, {default: Record<string, unknown>; exposed: string[][]}>>();

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
    if (!info && environment.workflowId) {
      setBusy(true);
      try {
        const response = await fetch(`/api/workflow/${encodeURIComponent(environment.workflowId)}/server`);
        if (!response.ok) throw new Error('Unable to load Server name');
        const value = await response.json();
        if (value.name) setName(value.name);
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
          Object.fromEntries(Object.entries(fields || {}).map(([node, value]) => [node, value.exposed])));
        if (result) { setInfo(result); setName(result.name); }
      } else {
        const activeName = info?.name || name;
        if (!activeName) throw new Error('Start this workflow first.');
        setInfo(await serverRequest(`/api/server/${encodeURIComponent(activeName)}${kind === 'stop' ? '/stop' : ''}`, token,
          kind === 'stop' ? {} : undefined));
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
  const example = info && `curl -X POST '${window.location.origin}${info.deploy_path}' \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(body)}'`;

  return <>
    <Button disabled={disabled} onClick={open}
      style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}>Server</Button>
    <SideSheet title="Server" visible={visible} onCancel={() => setVisible(false)} width={560}>
      <Space vertical align="start" spacing="medium" style={{ width: '100%' }}>
        <p>Start this workflow as an HTTP endpoint. Changes take effect when you select Start or Update.</p>
        <label htmlFor="server-name">Server name</label>
        <Input id="server-name" value={name} onChange={setName} disabled={busy}
          placeholder="Workflow name" />
        {!session && <><label htmlFor="server-admin-token">Management token</label>
        <Input id="server-admin-token" mode="password" value={token} onChange={setToken}
          placeholder="Token configured when starting the server" autoComplete="off" /></>}
        <details open>
          <summary>Public request fields</summary>
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
        </details>
        <Space>
          <Button theme="solid" loading={busy} disabled={(!token && !session) || busy || !name.trim() || !fields} onClick={() => action('start')}>
            {info?.status === 'ready' ? 'Update' : 'Start'}
          </Button>
          <Button disabled={busy || (!token && !session) || !info || info.status !== 'ready'} onClick={() => action('stop')}>Stop</Button>
          <Button disabled={busy || (!token && !session) || (!info && !name)} onClick={() => action('refresh')}>Refresh</Button>
        </Space>
        {info && <>
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
          <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxWidth: '100%' }}>{example}</pre>
        </>}
      </Space>
    </SideSheet>
  </>;
}

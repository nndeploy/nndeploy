export interface MCPToolConfig {
  name: string;
  description: string;
}

export function mcpValidation(enabled: boolean, name: string, description: string): string {
  if (!enabled) return '';
  if (!name.trim() && !description.trim()) return 'Tool name and description are required when MCP export is enabled.';
  if (!name.trim()) return 'Tool name is required when MCP export is enabled.';
  if (!description.trim()) return 'Tool description is required when MCP export is enabled.';
  if (!/^[A-Za-z0-9_.-]{1,128}$/.test(name.trim())) return 'Use 1–128 letters, numbers, underscores, hyphens or dots for the tool name.';
  if (['get_task', 'cancel_task'].includes(name.trim())) return 'This tool name is reserved.';
  if (!description.trim() || description.trim().length > 4096) return 'Enter a tool description of 1–4096 characters.';
  return '';
}

export function mcpConnectionUrl(path: string | null | undefined, origin: string): string {
  if (!path || !/^\/[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_-]+)*$/.test(path)) return '';
  return new URL(path, origin).href;
}

export interface ServerField {
  type: string;
  required: boolean;
  default?: unknown;
  dtype?: string;
  data_format?: string;
  media_type?: string;
  sample_rate?: number;
  shape?: Array<number | null>;
  max_elements?: number;
  max_bytes?: number;
}

export interface ServerInfo {
  name: string;
  workflow_id?: string;
  mcp?: MCPToolConfig | null;
  mcp_path?: string | null;
  status: string;
  deploy_path: string;
  schema: Record<string, Record<string, ServerField>>;
}

const pendingServerKeys = new Map<string, string>();

export async function serverRequest(path: string, token: string, body?: unknown): Promise<ServerInfo> {
  const payload = body === undefined ? undefined : JSON.stringify(body);
  const identity = `${path}\n${payload}`;
  const key = payload === undefined ? undefined : pendingServerKeys.get(identity) ||
    Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, '0')).join('');
  if (key) pendingServerKeys.set(identity, key);
  let url = path;
  let posting = payload !== undefined;
  let failures = 0;
  const deadline = Date.now() + 15 * 60 * 1000;
  while (Date.now() < deadline) {
    let response: Response;
    let result: any;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), Math.min(15000, deadline - Date.now()));
    try {
      response = await fetch(url, {
        method: posting ? 'POST' : 'GET',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}),
          ...(posting && key ? { 'Idempotency-Key': key } : {}) },
        ...(posting ? { body: payload } : {}),
      });
      // Gateways may return HTML or an empty body after the action was
      // accepted. Keep the original key through transport and body failures.
      if ([408, 429, 502, 503, 504].includes(response.status) &&
          !response.headers.get('Idempotency-Retention-Seconds')) {
        throw new Error('Server is temporarily unavailable. Retry this request.');
      }
      result = await response.json();
      failures = 0;
    } catch (error) {
      if (++failures > 2) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000));
      continue;
    } finally {
      clearTimeout(timeout);
    }
    if (response.status === 202) {
      const next = new URL(result.status_url, window.location.origin);
      if (next.origin !== window.location.origin || next.pathname !== '/api/server' ||
          !/^\?request_id=req_[0-9a-f]{32}$/.test(next.search) || next.hash) {
        throw new Error('Server returned an invalid status URL');
      }
      url = next.pathname + next.search;
      posting = false;
      await new Promise(resolve => setTimeout(resolve, 1000));
      continue;
    }
    pendingServerKeys.delete(identity);
    if (!response.ok) throw new Error(result.error?.message || result.detail || 'Server request failed');
    return result;
  }
  throw new Error('Server is still processing this request. Refresh its status before starting another action.');
}


export async function loadMcpAvailability(token: string): Promise<{ enabled: boolean; path: string | null }> {
  const response = await fetch('/api/server/mcp/config', {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (response.status === 401 || response.status === 403) {
    throw new Error('Sign in or check your management token to load MCP availability.');
  }
  if (response.status === 404 || !response.headers.get('content-type')?.includes('application/json')) {
    throw new Error('The running backend does not provide the MCP configuration API. Restart the updated server with --mcp, then refresh MCP availability.');
  }
  if (!response.ok) throw new Error('Unable to load MCP availability. Check the server and retry.');
  let value: any;
  try { value = await response.json(); }
  catch { throw new Error('The server returned invalid MCP configuration JSON. Check the backend version and retry.'); }
  if (!value || typeof value.enabled !== 'boolean' ||
      (value.enabled ? typeof value.path !== 'string' || !/^\/[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_-]+)*$/.test(value.path) : value.path !== null)) {
    throw new Error('The server returned an invalid MCP configuration. Check the backend version and retry.');
  }
  return value;
}

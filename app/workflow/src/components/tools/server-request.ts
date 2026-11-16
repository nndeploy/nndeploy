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


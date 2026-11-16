import type { RunTaskState } from './run-state';
import { rememberRun, recordRunState, restoreRun as readRunSession } from './run-session';
import type { RunTab } from './run-session';

// Keep only unconfirmed submissions. Retries use the original serialized graph,
// even when the user has since edited the canvas.
const pendingRuns = new Map<string, { payload: string; workflowId?: string; uncertain: boolean; storageWarning?: string }>();

export function submitRunTask(graph: unknown, tab?: RunTab) {
  const taskId = crypto.randomUUID();
  const payload = JSON.stringify(graph);
  const workflowId = tab?.id || undefined;
  const stored = !tab || rememberRun(taskId, payload, tab, workflowId);
  pendingRuns.set(taskId, { payload, workflowId, uncertain: false,
    storageWarning: stored ? undefined : 'Browser storage is unavailable; keep this tab open until the task finishes.' });
  return sendRunTask(taskId);
}

export function restoreRun(tabId: string) {
  const record = readRunSession(tabId);
  if (record && !record.confirmed) {
    pendingRuns.set(record.taskId, { payload: record.payload, workflowId: record.workflowId, uncertain: true });
  }
  return record;
}

export async function retryRunTask(taskId: string) {
  if (!pendingRuns.has(taskId)) throw new Error('Task submission is already confirmed');
  return sendRunTask(taskId);
}

async function sendRunTask(taskId: string) {
  const submission = pendingRuns.get(taskId)!;
  const pending = (message = 'Submission is unconfirmed. Checking the original task; Retry uses the same request.') => {
    submission.uncertain = true;
    return { flag: 'pending', result: { task_id: taskId }, message, storageWarning: submission.storageWarning };
  };
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const url = '/api/queue' + (submission.workflowId ? `?workflow_id=${encodeURIComponent(submission.workflowId)}` : '');
    const response = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'Idempotency-Key': taskId },
      body: submission.payload, signal: controller.signal,
    });
    const body = await response.json();
    if (response.ok && body?.flag === 'success' && body.result?.task_id === taskId) {
      pendingRuns.delete(taskId);
      recordRunState(taskId, 'PENDING');
      return { ...body, storageWarning: submission.storageWarning };
    }
    if (body?.error?.accepted === false ||
        (response.status >= 400 && response.status < 500 && response.status !== 408)) {
      const message = body?.error?.message || body?.message || 'Workflow submission rejected';
      // A proxy/auth rejection of a retry says nothing about the earlier POST.
      if (submission.uncertain) return pending(message);
      pendingRuns.delete(taskId);
      recordRunState(taskId, 'REJECTED');
      return { flag: 'error', message };
    }
    return pending();
  } catch {
    return pending();
  } finally {
    clearTimeout(timeout);
  }
}

// Progress is best-effort; exact task status recovers terminal events lost before
// binding or during reconnect. A missing record is not proof of rejection.
export function watchRunTask(
  taskId: string,
  createSocket: () => WebSocket,
  onState: (state: RunTaskState) => void,
  onMessage: (message: any) => void,
): () => void {
  let stopped = false;
  let socket: WebSocket | undefined;
  let reconnect: ReturnType<typeof setTimeout> | undefined;
  let connecting: ReturnType<typeof setTimeout> | undefined;
  let polling: ReturnType<typeof setTimeout> | undefined;
  let deadline: ReturnType<typeof setTimeout> | undefined;
  let query: AbortController | undefined;

  async function refresh() {
    if (stopped || query) return;
    clearTimeout(polling);
    const controller = new AbortController();
    query = controller;
    deadline = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(`/api/queue/task/${encodeURIComponent(taskId)}`, { signal: controller.signal });
      if (response.ok) {
        const body = await response.json();
        if (!stopped && body?.result?.task_id === taskId) {
          if (['PENDING', 'DISPATCHED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED'].includes(body.result.state)) {
            pendingRuns.delete(taskId);
            recordRunState(taskId, body.result.state);
          }
          onState(body.result);
        }
      }
    } catch {
      // Keep the original identity while the server or connection recovers.
    } finally {
      clearTimeout(deadline);
      query = undefined;
      if (!stopped) polling = setTimeout(refresh, 2000);
    }
  }

  function connect() {
    if (stopped) return;
    try {
      const current = createSocket();
      socket = current;
      connecting = setTimeout(() => current.close(), 10000);
      current.onopen = () => {
        if (stopped || socket !== current) return;
        clearTimeout(connecting);
        current.send(JSON.stringify({ type: 'bind', task_id: taskId }));
        void refresh();
      };
      current.onmessage = event => {
        if (stopped || socket !== current) return;
        try {
          const body = JSON.parse(event.data);
          if (body?.result && (body.result.task_id === taskId ||
              (!body.result.task_id && body.result.type === 'system'))) onMessage(body);
        } catch { /* Ignore malformed progress frames; status polling remains active. */ }
      };
      current.onerror = () => current.close();
      current.onclose = () => {
        if (stopped || socket !== current) return;
        clearTimeout(connecting);
        socket = undefined;
        reconnect = setTimeout(connect, 1000);
      };
    } catch {
      reconnect = setTimeout(connect, 1000);
    }
  }
  connect();
  void refresh();
  return () => {
    stopped = true;
    clearTimeout(reconnect);
    clearTimeout(connecting);
    clearTimeout(polling);
    clearTimeout(deadline);
    query?.abort();
    socket?.close();
  };
}

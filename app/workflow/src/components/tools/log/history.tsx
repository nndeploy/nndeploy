import { useEffect, useState } from 'react';
import { Button } from '@douyinfe/semi-ui';
import type { IOutputFile } from '../../../pages/components/flow/entity';
import { getOutputFileLabel } from '../webui/functions';

interface Task {
  task_id: string;
  name?: string;
  state: string;
  ts_finish?: number;
  status?: { messages?: string[] };
  time_profile?: { init_time?: number; run_time?: number };
  outputs?: Record<string, string>;
  outputs_truncated?: boolean;
  outputs_expired?: boolean;
  output_files?: IOutputFile[];
}
const pageSize = 25;

async function read<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  const value = await response.json();
  if (!response.ok || value.flag !== 'success') {
    throw new Error(value.error?.message || value.message || 'Unable to load task history');
  }
  return value.result;
}

/** Inspect saved results without changing the active workflow or its Run state. */
export default function History() {
  const [before, setBefore] = useState<string>();
  const [refresh, setRefresh] = useState(0);
  const [items, setItems] = useState<Task[]>([]);
  const [selected, setSelected] = useState<string>();
  const [detailRefresh, setDetailRefresh] = useState(0);
  const [detail, setDetail] = useState<Task>();
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState('');
  const [detailError, setDetailError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    let active = true;
    setLoading(true);
    setError('');
    setItems([]);
    setSelected(undefined);
    read<{ items: Task[] }>(`/api/queue/history?summary=true&max_items=${pageSize}${before ? `&before=${encodeURIComponent(before)}` : ''}`, controller.signal)
      .then(value => { if (active) setItems(value.items); })
      .catch(() => { if (active) setError('Unable to load history. Try Latest again.'); })
      .finally(() => { clearTimeout(timeout); if (active) setLoading(false); });
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, [before, refresh]);

  useEffect(() => {
    setDetail(undefined);
    setDetailError('');
    setDetailLoading(!!selected);
    if (!selected) return;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    let active = true;
    read<Task>(`/api/queue/task/${encodeURIComponent(selected)}`, controller.signal)
      .then(value => { if (active) setDetail(value); })
      .catch(() => { if (active) setDetailError('Unable to load this task. Select it again to retry.'); })
      .finally(() => { clearTimeout(timeout); if (active) setDetailLoading(false); });
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, [selected, detailRefresh]);

  return <div className="tools-task-history">
    <section aria-label="Task history">
      <p>Completed tasks in this workspace</p>
      <div className="history-actions">
        <Button onClick={() => { setBefore(undefined); setRefresh(value => value + 1); }} disabled={loading}>Latest</Button>
        <Button onClick={() => setBefore(items[items.length - 1].task_id)} disabled={loading || items.length < pageSize}>Older</Button>
      </div>
      {loading && <p role="status">Loading history…</p>}
      {error && <p role="alert">{error}</p>}
      {!loading && !error && !items.length && <p>No completed tasks on this page.</p>}
      <ul className="history-tasks">
        {items.map(task => <li key={task.task_id}>
          <button type="button" aria-pressed={selected === task.task_id}
            onClick={() => { setSelected(task.task_id); setDetailRefresh(value => value + 1); }}>
            <strong>{task.name || 'Workflow'}</strong> · {task.state}
            <br /><small>{task.ts_finish ? new Date(task.ts_finish * 1000).toLocaleString() : ''}</small>
            <br /><code>{task.task_id}</code>
          </button>
        </li>)}
      </ul>
    </section>
    <section aria-label="Task results">
      {detailLoading && <p role="status">Loading results…</p>}
      {detailError && <p role="alert">{detailError}</p>}
      {!selected && <p>Select a task to view its saved results.</p>}
      {detail && <>
        <h3>{detail.state}</h3>
        <code>{detail.task_id}</code>
        {detail.status?.messages?.map((message, i) => <p key={i}>{message}</p>)}
        {detail.time_profile?.run_time !== undefined && <p>Run time: {detail.time_profile.run_time.toFixed(2)} ms</p>}
        {detail.outputs_expired && <p role="status">Saved results have expired.</p>}
        {detail.outputs_truncated && <p role="status">Output preview was truncated.</p>}
        {Object.entries(detail.outputs || {}).map(([name, value]) => <div key={name}>
          <h4>{name}</h4><pre>{value}</pre>
        </div>)}
        {detail.output_files?.map(file => <div key={file.file_id} className="history-file">
          <h4>{getOutputFileLabel(file)}</h4>
          <a href={file.url} download={file.file_id + (file.media_type === 'application/zip' ? '.zip' : '')}>Download ({file.size} bytes)</a>
          {file.media_type.startsWith('image/') && <img src={file.url} alt={getOutputFileLabel(file)} loading="lazy" />}
          {file.media_type.startsWith('video/') && <video controls src={file.url} preload="none" />}
          {file.media_type.startsWith('audio/') && <audio controls src={file.url} preload="none" />}
        </div>)}
        {!Object.keys(detail.outputs || {}).length && !detail.output_files?.length &&
          <p>No saved output preview for this task.</p>}
      </>}
    </section>
  </div>;
}

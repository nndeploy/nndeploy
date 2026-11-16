import type { EnumFlowType } from '../../enum';

export interface RunTab {
  tabId: string;
  id: string;
  name: string;
  flowType: EnumFlowType;
}
interface StoredRun {
  version: 1;
  taskId: string;
  payload: string;
  confirmed: boolean;
  tab: RunTab;
  workflowId?: string;
}
const prefix = 'nndeploy.editor.run.v1:';
const records = new Map<string, StoredRun>();

function parse(value: string | null): StoredRun | undefined {
  try {
    const record = JSON.parse(value || 'null');
    if (record?.version !== 1 || typeof record.confirmed !== 'boolean' ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(record.taskId) ||
        typeof record.payload !== 'string' || !record.tab ||
        typeof record.tab.tabId !== 'string' || !record.tab.tabId || record.tab.tabId.length > 128 ||
        typeof record.tab.id !== 'string' || typeof record.tab.name !== 'string' ||
        (record.workflowId !== undefined && (typeof record.workflowId !== 'string' || record.workflowId.length > 128)) ||
        !['workspace', 'template'].includes(record.tab.flowType)) return;
    const graph = JSON.parse(record.payload);
    if (!graph || typeof graph !== 'object' || Array.isArray(graph)) return;
    return record;
  } catch { return; }
}

function save(record: StoredRun, replace = false): boolean {
  try {
    const key = prefix + record.tab.tabId;
    const current = parse(sessionStorage.getItem(key));
    // A late response from an older page/request cannot replace a newer run.
    if (!replace && current?.taskId !== record.taskId) return false;
    sessionStorage.setItem(key, JSON.stringify(record));
    return true;
  } catch { return false; }
}

export function rememberRun(taskId: string, payload: string, tab: RunTab, workflowId?: string): boolean {
  const record: StoredRun = { version: 1, taskId, payload, confirmed: false, tab: { ...tab }, workflowId };
  records.set(tab.tabId, record);
  return save(record, true);
}

export function restoreRun(tabId: string): StoredRun | undefined {
  try {
    const record = parse(sessionStorage.getItem(prefix + tabId));
    if (record && record.tab.tabId === tabId) {
      records.set(tabId, record);
      return record;
    }
  } catch { /* Storage may be disabled; the live page can still observe its task. */ }
  return records.get(tabId);
}

export function restoreRunTabs(): RunTab[] {
  try {
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key?.startsWith(prefix)) restoreRun(key.slice(prefix.length));
    }
  } catch { /* Restoring the editor must not depend on storage availability. */ }
  return [...records.values()].map(record => ({ ...record.tab }));
}

export function updateRunTab(tab: RunTab) {
  const record = records.get(tab.tabId);
  if (!record) return;
  record.tab = { ...tab };
  save(record);
}

export function recordRunState(taskId: string, state: string) {
  const terminal = ['SUCCEEDED', 'FAILED', 'CANCELLED', 'REJECTED'].includes(state);
  if (!terminal && !['PENDING', 'DISPATCHED', 'RUNNING'].includes(state)) return;
  for (const [tabId, record] of records) {
    if (record.taskId !== taskId) continue;
    if (terminal) {
      records.delete(tabId);
      try {
        const key = prefix + tabId;
        if (parse(sessionStorage.getItem(key))?.taskId === taskId) sessionStorage.removeItem(key);
      } catch { /* A retained terminal record is reconciled on the next refresh. */ }
    } else {
      record.confirmed = true;
      save(record);
    }
  }
}

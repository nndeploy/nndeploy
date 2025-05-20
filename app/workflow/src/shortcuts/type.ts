import type { WorkflowJSON } from '@flowgram.ai/free-layout-editor';

import type { WorkflowClipboardDataID } from './constants';

export interface WorkflowClipboardSource {
  host: string;
  // more: id?, workspaceId? etc.
}

export interface WorkflowClipboardRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WorkflowClipboardData {
  type: typeof WorkflowClipboardDataID;
  json: WorkflowJSON;
  source: WorkflowClipboardSource;
  bounds: WorkflowClipboardRect;
}

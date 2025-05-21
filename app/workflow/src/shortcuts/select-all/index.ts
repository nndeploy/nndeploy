import {
  FreeLayoutPluginContext,
  Playground,
  ShortcutsHandler,
  WorkflowDocument,
} from '@flowgram.ai/free-layout-editor';

import { FlowCommandId } from '../constants';

export class SelectAllShortcut implements ShortcutsHandler {
  public commandId = FlowCommandId.SELECT_ALL;

  public shortcuts = ['meta a', 'ctrl a'];

  private document: WorkflowDocument;

  private playground: Playground;

  constructor(context: FreeLayoutPluginContext) {
    this.document = context.get(WorkflowDocument);
    this.playground = context.playground;
    this.execute = this.execute.bind(this);
  }

  public async execute(): Promise<void> {
    const allNodes = this.document.getAllNodes();
    this.playground.selectionService.selection = allNodes;
  }
}

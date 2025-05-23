import {
  FreeLayoutPluginContext,
  ShortcutsHandler,
  WorkflowSelectService,
} from '@flowgram.ai/free-layout-editor';

import { FlowCommandId } from '../constants';

export class CollapseShortcut implements ShortcutsHandler {
  public commandId = FlowCommandId.COLLAPSE;

  public commandDetail: ShortcutsHandler['commandDetail'] = {
    label: 'Collapse',
  };

  public shortcuts = ['meta alt openbracket', 'ctrl alt openbracket'];

  private selectService: WorkflowSelectService;

  constructor(context: FreeLayoutPluginContext) {
    this.selectService = context.get(WorkflowSelectService);
    this.execute = this.execute.bind(this);
  }

  public async execute(): Promise<void> {
    this.selectService.selectedNodes.forEach((node) => {
      node.renderData.expanded = false;
    });
  }
}

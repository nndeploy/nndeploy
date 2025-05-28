import {
  FreeLayoutPluginContext,
  PlaygroundConfigEntity,
  ShortcutsHandler,
} from '@flowgram.ai/free-layout-editor';

import { FlowCommandId } from '../constants';

export class ZoomOutShortcut implements ShortcutsHandler {
  public commandId = FlowCommandId.ZOOM_OUT;

  public shortcuts = ['meta -', 'ctrl -'];

  private playgroundConfig: PlaygroundConfigEntity;

  constructor(context: FreeLayoutPluginContext) {
    this.playgroundConfig = context.get(PlaygroundConfigEntity);
    this.execute = this.execute.bind(this);
  }

  public async execute(): Promise<void> {
    if (this.playgroundConfig.zoom > 1.9) {
      return;
    }
    this.playgroundConfig.zoomout();
  }
}

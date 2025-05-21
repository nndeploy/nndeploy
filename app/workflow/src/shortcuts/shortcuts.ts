import { FreeLayoutPluginContext, ShortcutsRegistry } from '@flowgram.ai/free-layout-editor';

import { ZoomOutShortcut } from './zoom-out';
import { ZoomInShortcut } from './zoom-in';
import { SelectAllShortcut } from './select-all';
import { PasteShortcut } from './paste';
import { ExpandShortcut } from './expand';
import { DeleteShortcut } from './delete';
import { CopyShortcut } from './copy';
import { CollapseShortcut } from './collapse';

export function shortcuts(shortcutsRegistry: ShortcutsRegistry, ctx: FreeLayoutPluginContext) {
  shortcutsRegistry.addHandlers(
    new CopyShortcut(ctx),
    new PasteShortcut(ctx),
    new SelectAllShortcut(ctx),
    new CollapseShortcut(ctx),
    new ExpandShortcut(ctx),
    new DeleteShortcut(ctx),
    new ZoomInShortcut(ctx),
    new ZoomOutShortcut(ctx)
  );
}

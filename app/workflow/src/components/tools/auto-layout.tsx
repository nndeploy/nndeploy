import { forwardRef, useCallback } from 'react';

import { usePlayground, usePlaygroundTools } from '@flowgram.ai/free-layout-editor';
import { IconButton, Tooltip } from '@douyinfe/semi-ui';

import { IconAutoLayout } from '../../assets/icon-auto-layout';

export const AutoLayout =  forwardRef<any>((props, ref) => {
  const tools = usePlaygroundTools();
  const playground = usePlayground();
  const autoLayout = useCallback(async () => {
    await tools.autoLayout(
      //{layoutConfig: {rankdir: 'TB'}}
    );
  }, [tools]);

  return (
    <Tooltip content={'Auto Layout'}>
      <IconButton
        disabled={playground.config.readonly}
        type="tertiary"
        theme="borderless"
        onClick={autoLayout}
        ref = {ref}
        icon={IconAutoLayout}
      />
    </Tooltip>
  );
});

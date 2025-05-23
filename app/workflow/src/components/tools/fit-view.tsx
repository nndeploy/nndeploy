import { usePlaygroundTools } from '@flowgram.ai/free-layout-editor';
import { IconButton, Tooltip } from '@douyinfe/semi-ui';
import { IconExpand } from '@douyinfe/semi-icons';

export const FitView = () => {
  const tools = usePlaygroundTools();
  return (
    <Tooltip content="FitView">
      <IconButton
        icon={<IconExpand />}
        type="tertiary"
        theme="borderless"
        onClick={() => tools.fitView()}
      />
    </Tooltip>
  );
};

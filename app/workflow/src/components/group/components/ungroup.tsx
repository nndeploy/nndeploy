import { CSSProperties, FC } from 'react';

import { CommandRegistry, useService, WorkflowNodeEntity } from '@flowgram.ai/free-layout-editor';
import { WorkflowGroupCommand } from '@flowgram.ai/free-group-plugin';
import { Button, Tooltip } from '@douyinfe/semi-ui';

import { IconUngroup } from './icon-group';

interface UngroupButtonProps {
  node: WorkflowNodeEntity;
  style?: CSSProperties;
}

export const UngroupButton: FC<UngroupButtonProps> = ({ node, style }) => {
  const commandRegistry = useService(CommandRegistry);
  return (
    <Tooltip content="Ungroup">
      <div className="workflow-group-ungroup" style={style}>
        <Button
          icon={<IconUngroup size={14} />}
          style={{ height: 30, width: 30 }}
          theme="borderless"
          type="tertiary"
          onClick={() => {
            commandRegistry.executeCommand(WorkflowGroupCommand.Ungroup, node);
          }}
        />
      </div>
    </Tooltip>
  );
};

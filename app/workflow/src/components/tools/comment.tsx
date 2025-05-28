import { useState, useCallback } from 'react';

import {
  delay,
  usePlayground,
  useService,
  WorkflowDocument,
  WorkflowDragService,
  WorkflowSelectService,
} from '@flowgram.ai/free-layout-editor';
import { IconButton, Tooltip } from '@douyinfe/semi-ui';

import { WorkflowNodeType } from '../../nodes';
import { IconComment } from '../../assets/icon-comment';

export const Comment = () => {
  const playground = usePlayground();
  const document = useService(WorkflowDocument);
  const selectService = useService(WorkflowSelectService);
  const dragService = useService(WorkflowDragService);

  const [tooltipVisible, setTooltipVisible] = useState(false);

  const calcNodePosition = useCallback(
    (mouseEvent: React.MouseEvent<HTMLButtonElement>) => {
      const mousePosition = playground.config.getPosFromMouseEvent(mouseEvent);
      return {
        x: mousePosition.x,
        y: mousePosition.y - 75,
      };
    },
    [playground]
  );

  const createComment = useCallback(
    async (mouseEvent: React.MouseEvent<HTMLButtonElement>) => {
      setTooltipVisible(false);
      const canvasPosition = calcNodePosition(mouseEvent);
      // 创建节点
      const node = document.createWorkflowNodeByType(WorkflowNodeType.Comment, canvasPosition);
      // 等待节点渲染
      await delay(16);
      // 选中节点
      selectService.selectNode(node);
      // 开始拖拽
      dragService.startDragSelectedNodes(mouseEvent);
    },
    [selectService, calcNodePosition, document, dragService]
  );

  return (
    <Tooltip
      trigger="custom"
      visible={tooltipVisible}
      onVisibleChange={setTooltipVisible}
      content="Comment"
    >
      <IconButton
        disabled={playground.config.readonly}
        icon={
          <IconComment
            style={{
              width: 16,
              height: 16,
            }}
          />
        }
        type="tertiary"
        theme="borderless"
        onClick={createComment}
        onMouseEnter={() => setTooltipVisible(true)}
        onMouseLeave={() => setTooltipVisible(false)}
      />
    </Tooltip>
  );
};

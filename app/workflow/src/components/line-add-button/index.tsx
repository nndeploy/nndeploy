import { useCallback } from 'react';

import {
  WorkflowNodePanelService,
  WorkflowNodePanelUtils,
} from '@flowgram.ai/free-node-panel-plugin';
import { LineRenderProps } from '@flowgram.ai/free-lines-plugin';
import {
  delay,
  HistoryService,
  useService,
  WorkflowDocument,
  WorkflowDragService,
  WorkflowLinesManager,
  WorkflowNodeEntity,
  WorkflowNodeJSON,
} from '@flowgram.ai/free-layout-editor';

import './index.less';
import { useVisible } from './use-visible';
import { IconPlusCircle } from './button';

export const LineAddButton = (props: LineRenderProps) => {
  const { line, selected, hovered, color } = props;
  const visible = useVisible({ line, selected, hovered });
  const nodePanelService = useService<WorkflowNodePanelService>(WorkflowNodePanelService);
  const document = useService(WorkflowDocument);
  const dragService = useService(WorkflowDragService);
  const linesManager = useService(WorkflowLinesManager);
  const historyService = useService(HistoryService);

  const { fromPort, toPort } = line;

  const onClick = useCallback(async () => {
    // calculate the middle point of the line - 计算线条的中点位置
    const position = {
      x: (line.position.from.x + line.position.to.x) / 2,
      y: (line.position.from.y + line.position.to.y) / 2,
    };

    // get container node for the new node - 获取新节点的容器节点
    const containerNode = WorkflowNodePanelUtils.getContainerNode({
      fromPort,
    });

    // show node selection panel - 显示节点选择面板
    const result = await nodePanelService.singleSelectNodePanel({
      position,
      containerNode,
      panelProps: {
        enableScrollClose: true,
      },
    });
    if (!result) {
      return;
    }

    const { nodeType, nodeJSON } = result;

    // adjust position for the new node - 调整新节点的位置
    const nodePosition = WorkflowNodePanelUtils.adjustNodePosition({
      nodeType,
      position,
      fromPort,
      toPort,
      containerNode,
      document,
      dragService,
    });

    // create new workflow node - 创建新的工作流节点
    const node: WorkflowNodeEntity = document.createWorkflowNodeByType(
      nodeType,
      nodePosition,
      nodeJSON ?? ({} as WorkflowNodeJSON),
      containerNode?.id
    );

    // auto offset subsequent nodes - 自动偏移后续节点
    if (fromPort && toPort) {
      WorkflowNodePanelUtils.subNodesAutoOffset({
        node,
        fromPort,
        toPort,
        containerNode,
        historyService,
        dragService,
        linesManager,
      });
    }

    // wait for node render - 等待节点渲染
    await delay(20);

    // build connection lines - 构建连接线
    WorkflowNodePanelUtils.buildLine({
      fromPort,
      node,
      toPort,
      linesManager,
    });

    // remove original line - 移除原始线条
    line.dispose();
  }, []);

  if (!visible) {
    return <></>;
  }

  return (
    <div
      className="line-add-button"
      style={{
        left: '50%',
        top: '50%',
        color,
      }}
      data-testid="sdk.workflow.canvas.line.add"
      data-line-id={line.id}
      onClick={onClick}
    >
      <IconPlusCircle />
    </div>
  );
};

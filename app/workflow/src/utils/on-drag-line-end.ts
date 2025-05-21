import {
  WorkflowNodePanelService,
  WorkflowNodePanelUtils,
} from '@flowgram.ai/free-node-panel-plugin';
import {
  delay,
  FreeLayoutPluginContext,
  onDragLineEndParams,
  WorkflowDragService,
  WorkflowLinesManager,
  WorkflowNodeEntity,
  WorkflowNodeJSON,
} from '@flowgram.ai/free-layout-editor';

/**
 * Drag the end of the line to create an add panel (feature optional)
 * 拖拽线条结束需要创建一个添加面板 （功能可选）
 */
export const onDragLineEnd = async (ctx: FreeLayoutPluginContext, params: onDragLineEndParams) => {
  // get services from context - 从上下文获取服务
  const nodePanelService = ctx.get(WorkflowNodePanelService);
  const document = ctx.document;
  const dragService = ctx.get(WorkflowDragService);
  const linesManager = ctx.get(WorkflowLinesManager);

  // get params from drag event - 从拖拽事件获取参数
  const { fromPort, toPort, mousePos, line, originLine } = params;

  // return if invalid line state - 如果线条状态无效则返回
  if (originLine || !line) {
    return;
  }

  // return if target port exists - 如果目标端口存在则返回
  if (toPort) {
    return;
  }

  // get container node for the new node - 获取新节点的容器节点
  const containerNode = WorkflowNodePanelUtils.getContainerNode({
    fromPort,
  });

  // open node selection panel - 打开节点选择面板
  const result = await nodePanelService.singleSelectNodePanel({
    position: mousePos,
    containerNode,
    panelProps: {
      enableNodePlaceholder: true,
      enableScrollClose: true,
    },
  });

  // return if no node selected - 如果没有选择节点则返回
  if (!result) {
    return;
  }

  // get selected node type and data - 获取选择的节点类型和数据
  const { nodeType, nodeJSON } = result;

  // calculate position for the new node - 计算新节点的位置
  const nodePosition = WorkflowNodePanelUtils.adjustNodePosition({
    nodeType,
    position: mousePos,
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

  // wait for node render - 等待节点渲染
  await delay(20);

  // build connection line - 构建连接线
  WorkflowNodePanelUtils.buildLine({
    fromPort,
    node,
    toPort,
    linesManager,
  });
};

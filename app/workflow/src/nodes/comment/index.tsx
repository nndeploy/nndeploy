import { WorkflowNodeType } from '../constants';
import { FlowNodeRegistry } from '../../typings';

export const CommentNodeRegistry: FlowNodeRegistry = {
  type: WorkflowNodeType.Comment,
  meta: {
    disableSideBar: true,
    defaultPorts: [],
    renderKey: WorkflowNodeType.Comment,
    size: {
      width: 240,
      height: 150,
    },
  },
  formMeta: {
    render: () => <></>,
  },
  getInputPoints: () => [], // Comment 节点没有输入
  getOutputPoints: () => [], // Comment 节点没有输出
};

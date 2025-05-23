import { FlowNodeRegistry } from '../../typings';
import iconStart from '../../assets/icon-start.jpg';
import { formMeta } from './form-meta';
import { WorkflowNodeType } from '../constants';

export const StartNodeRegistry: FlowNodeRegistry = {
  type: WorkflowNodeType.Start,
  meta: {
    isStart: true,
    deleteDisable: true,
    copyDisable: true,
    defaultPorts: [{ type: 'output' }],
    size: {
      width: 360,
      height: 211,
    },
  },
  info: {
    icon: iconStart,
    description:
      'The starting node of the workflow, used to set the information needed to initiate the workflow.',
  },
  /**
   * Render node via formMeta
   */
  formMeta,
  /**
   * Start Node cannot be added
   */
  canAdd() {
    return false;
  },
};

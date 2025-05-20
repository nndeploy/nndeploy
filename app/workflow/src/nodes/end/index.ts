import { FlowNodeRegistry } from '../../typings';
import iconEnd from '../../assets/icon-end.jpg';
import { formMeta } from './form-meta';
import { WorkflowNodeType } from '../constants';

export const EndNodeRegistry: FlowNodeRegistry = {
  type: WorkflowNodeType.End,
  meta: {
    deleteDisable: true,
    copyDisable: true,
    defaultPorts: [{ type: 'input' }],
    size: {
      width: 360,
      height: 211,
    },
  },
  info: {
    icon: iconEnd,
    description:
      'The final node of the workflow, used to return the result information after the workflow is run.',
  },
  /**
   * Render node via formMeta
   */
  formMeta,
  /**
   * End Node cannot be added
   */
  canAdd() {
    return false;
  },
};

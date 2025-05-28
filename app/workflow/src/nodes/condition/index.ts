import { nanoid } from 'nanoid';

import { FlowNodeRegistry } from '../../typings';
import iconCondition from '../../assets/icon-condition.svg';
import { formMeta } from './form-meta';
import { WorkflowNodeType } from '../constants';

export const ConditionNodeRegistry: FlowNodeRegistry = {
  type: WorkflowNodeType.Condition,
  info: {
    icon: iconCondition,
    description:
      'Connect multiple downstream branches. Only the corresponding branch will be executed if the set conditions are met.',
  },
  meta: {
    defaultPorts: [{ type: 'input' }],
    // Condition Outputs use dynamic port
    useDynamicPort: true,
    expandable: false, // disable expanded
  },
  formMeta,
  onAdd() {
    return {
      id: `condition_${nanoid(5)}`,
      type: 'condition',
      data: {
        title: 'Condition',
        inputsValues: {
          conditions: [
            {
              key: `if_${nanoid(5)}`,
              value: '',
            },
            {
              key: `if_${nanoid(5)}`,
              value: '',
            },
          ],
        },
        inputs: {
          type: 'object',
          properties: {
            conditions: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  key: {
                    type: 'string',
                  },
                  value: {
                    type: 'string',
                  },
                },
              },
            },
          },
        },
      },
    };
  },
};

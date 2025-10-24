/**
 * Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 * SPDX-License-Identifier: MIT
 */
import {
  FlowNodeBaseType,
  WorkflowNodeEntity,
  PositionSchema,
  FlowNodeTransformData,
  nanoid,
} from '@flowgram.ai/free-layout-editor';

import { FlowNodeRegistry } from '../../typings';

let index = 0;
export const GroupNodeRegistry: FlowNodeRegistry = {
  type: FlowNodeBaseType.GROUP,
  meta: {
    renderKey: FlowNodeBaseType.GROUP,
    defaultPorts: [],
    isContainer: true,
    disableSideBar: true,
    size: {
      width: 560,
      height: 400,
    },
    padding: () => ({
      top: 80,
      bottom: 40,
      left: 65,
      right: 65,
    }),
    selectable(node: WorkflowNodeEntity, mousePos?: PositionSchema): boolean {
      if (!mousePos) {
        return true;
      }
      const transform = node.getData<FlowNodeTransformData>(FlowNodeTransformData);
      return !transform.bounds.contains(mousePos.x, mousePos.y);
    },
    expandable: false,
    /**
     * It cannot be added through the panel
     * 不能通过面板添加
     */
    nodePanelVisible: false,
  },
  formMeta: {
    render: () => <></>,
  },
  onAdd() {
    return {
      type: FlowNodeBaseType.GROUP,
      id: `group_${nanoid(5)}`,
      meta: {
        position: {
          x: 0,
          y: 0,
        },
      },
      data: {
        color: 'Green',
        title: `Group_${++index}`,
      },
    };
  },
};

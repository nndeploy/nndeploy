import React, { useState, useContext } from 'react';

import { WorkflowPortRender } from '@flowgram.ai/free-layout-editor';
import { useClientContext } from '@flowgram.ai/free-layout-editor';

import { useNodeRenderContext } from '../../hooks';
import { SidebarContext } from '../../context';
import { scrollToView } from './utils';
import { NodeWrapperStyle } from './styles';
import { BorderArea } from './border-area';
import { useSize } from '../comment/hooks';
import { isContainerNode } from '../../pages/components/flow/functions';

export interface NodeWrapperProps {
  isScrollToView?: boolean;
  children: React.ReactNode;
}

/**
 * Used for drag-and-drop/click events and ports rendering of nodes
 * 用于节点的拖拽/点击事件和点位渲染
 */
export const NodeWrapper: React.FC<NodeWrapperProps> = (props) => {
  const { children, isScrollToView = false } = props;
  const nodeRender = useNodeRenderContext();
  const { selected, startDrag, ports, selectNode, nodeRef, onFocus, onBlur, node } = nodeRender;
  const [isDragging, setIsDragging] = useState(false);
  const sidebar = useContext(SidebarContext);
  const form = nodeRender.form;
  const ctx = useClientContext();

  const isContainer = isContainerNode(node.id, ctx);

  const { width, height, onResize } = useSize();

  if (form?.getValueIn('name_') === 'decode_infer') {
    let i = 0;

  }



  const portsRender = ports.map((p) => <WorkflowPortRender key={p.id} entity={p} />);

  return (
    <>
      <NodeWrapperStyle
        className={'my-node-wrapper ' + (selected ? 'selected' : '')}
        ref={nodeRef}
        draggable
        onDragStart={(e) => {
          startDrag(e);
          setIsDragging(true);
        }}
        onClick={(e) => {
          selectNode(e);
          ///@ts-ignore
          if (e?.nativeEvent?.currentTarget?.classList?.contains('semi-portal')) {
            return;
          }
          if (!isDragging) {
            sidebar.setNodeRender(nodeRender);
            // 可选：将 isScrollToView 设为 true，可以让节点选中后滚动到画布中间
            // Optional: Set isScrollToView to true to scroll the node to the center of the canvas after it is selected.
            if (isScrollToView) {
              scrollToView(ctx, nodeRender.node);
            }
          }
        }}
        onMouseUp={() => setIsDragging(false)}
        onFocus={onFocus}
        onBlur={onBlur}
        data-node-selected={String(selected)}
        style={{
          outline: form?.state.invalid ? '1px solid red' : 'none',

          width: isContainer ? 'auto' : width,
          height: isContainer ? 'auto' : (height <= 80 ? 'auto' : height)

        }}

      >
        {children}
      </NodeWrapperStyle>

      {!isContainer && <BorderArea onResize={onResize} />}
      <div className='my-port-parent'>
        {portsRender}
      </div>

    </>
  );
};

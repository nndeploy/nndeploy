import { CSSProperties, type FC } from 'react';

import { useNodeRender, usePlayground } from '@flowgram.ai/free-layout-editor';

import type { CommentEditorModel } from '../model';

interface IResizeArea {
  model: CommentEditorModel;
  onResize?: () => {
    resizing: (delta: { top: number; right: number; bottom: number; left: number }) => void;
    resizeEnd: () => void;
  };
  getDelta?: (delta: { x: number; y: number }) => {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  style?: CSSProperties;
}

export const ResizeArea: FC<IResizeArea> = (props) => {
  const { model, onResize, getDelta, style } = props;

  const playground = usePlayground();

  const { selectNode } = useNodeRender();

  const handleMouseDown = (mouseDownEvent: React.MouseEvent) => {
    mouseDownEvent.preventDefault();
    mouseDownEvent.stopPropagation();
    if (!onResize) {
      return;
    }
    const { resizing, resizeEnd } = onResize();
    model.setFocus(false);
    selectNode(mouseDownEvent);
    playground.node.focus(); // 防止节点无法被删除

    const startX = mouseDownEvent.clientX;
    const startY = mouseDownEvent.clientY;

    const handleMouseMove = (mouseMoveEvent: MouseEvent) => {
      const deltaX = mouseMoveEvent.clientX - startX;
      const deltaY = mouseMoveEvent.clientY - startY;
      const delta = getDelta?.({ x: deltaX, y: deltaY });
      if (!delta || !resizing) {
        return;
      }
      resizing(delta);
    };

    const handleMouseUp = () => {
      resizeEnd();
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('click', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('click', handleMouseUp);
  };

  return (
    <div
      className="workflow-comment-resize-area"
      style={style}
      data-flow-editor-selectable="false"
      onMouseDown={handleMouseDown}
    />
  );
};

import { type FC, useState, useEffect, type WheelEventHandler } from 'react';

import { useNodeRender, usePlayground } from '@flowgram.ai/free-layout-editor';

import type { CommentEditorModel } from '../model';
import { DragArea } from './drag-area';

interface IContentDragArea {
  model: CommentEditorModel;
  focused: boolean;
  overflow: boolean;
}

export const ContentDragArea: FC<IContentDragArea> = (props) => {
  const { model, focused, overflow } = props;
  const playground = usePlayground();
  const { selectNode } = useNodeRender();

  const [active, setActive] = useState(false);

  useEffect(() => {
    // 当编辑器失去焦点时，取消激活状态
    if (!focused) {
      setActive(false);
    }
  }, [focused]);

  const handleWheel: WheelEventHandler<HTMLDivElement> = (e) => {
    const editorElement = model.element;
    if (active || !overflow || !editorElement) {
      return;
    }
    e.stopPropagation();
    const maxScroll = editorElement.scrollHeight - editorElement.clientHeight;
    const newScrollTop = Math.min(Math.max(editorElement.scrollTop + e.deltaY, 0), maxScroll);
    editorElement.scroll(0, newScrollTop);
  };

  const handleMouseDown = (mouseDownEvent: React.MouseEvent) => {
    if (active) {
      return;
    }
    mouseDownEvent.preventDefault();
    mouseDownEvent.stopPropagation();
    model.setFocus(false);
    selectNode(mouseDownEvent);
    playground.node.focus(); // 防止节点无法被删除

    const startX = mouseDownEvent.clientX;
    const startY = mouseDownEvent.clientY;

    const handleMouseUp = (mouseMoveEvent: MouseEvent) => {
      const deltaX = mouseMoveEvent.clientX - startX;
      const deltaY = mouseMoveEvent.clientY - startY;
      // 判断是拖拽还是点击
      const delta = 5;
      if (Math.abs(deltaX) < delta && Math.abs(deltaY) < delta) {
        // 点击后隐藏
        setActive(true);
      }
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('click', handleMouseUp);
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('click', handleMouseUp);
  };

  return (
    <div
      className="workflow-comment-content-drag-area"
      onMouseDown={handleMouseDown}
      onWheel={handleWheel}
      style={{
        display: active ? 'none' : undefined,
      }}
    >
      <DragArea
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
        }}
        model={model}
        stopEvent={false}
      />
    </div>
  );
};

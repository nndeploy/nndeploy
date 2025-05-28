import { CSSProperties, type FC } from 'react';

import { useNodeRender, usePlayground } from '@flowgram.ai/free-layout-editor';

import { type CommentEditorModel } from '../model';

interface IDragArea {
  model: CommentEditorModel;
  stopEvent?: boolean;
  style?: CSSProperties;
}

export const DragArea: FC<IDragArea> = (props) => {
  const { model, stopEvent = true, style } = props;

  const playground = usePlayground();

  const { startDrag: onStartDrag, onFocus, onBlur, selectNode } = useNodeRender();

  return (
    <div
      className="workflow-comment-drag-area"
      data-flow-editor-selectable="false"
      draggable={true}
      style={style}
      onMouseDown={(e) => {
        if (stopEvent) {
          e.preventDefault();
          e.stopPropagation();
        }
        model.setFocus(false);
        onStartDrag(e);
        selectNode(e);
        playground.node.focus(); // 防止节点无法被删除
      }}
      onFocus={onFocus}
      onBlur={onBlur}
    />
  );
};

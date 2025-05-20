import { type FC } from 'react';

import type { CommentEditorModel } from '../model';
import { ResizeArea } from './resize-area';
import { DragArea } from './drag-area';

interface IBorderArea {
  model: CommentEditorModel;
  overflow: boolean;
  onResize?: () => {
    resizing: (delta: { top: number; right: number; bottom: number; left: number }) => void;
    resizeEnd: () => void;
  };
}

export const BorderArea: FC<IBorderArea> = (props) => {
  const { model, overflow, onResize } = props;

  return (
    <div style={{ zIndex: 999 }}>
      {/* 左边 */}
      <DragArea
        style={{
          position: 'absolute',
          left: -10,
          top: 10,
          width: 20,
          height: 'calc(100% - 20px)',
        }}
        model={model}
      />
      {/* 右边 */}
      <DragArea
        style={{
          position: 'absolute',
          right: -10,
          top: 10,
          height: 'calc(100% - 20px)',
          width: overflow ? 10 : 20, // 防止遮挡滚动条
        }}
        model={model}
      />
      {/* 上边 */}
      <DragArea
        style={{
          position: 'absolute',
          top: -10,
          left: 10,
          width: 'calc(100% - 20px)',
          height: 20,
        }}
        model={model}
      />
      {/* 下边 */}
      <DragArea
        style={{
          position: 'absolute',
          bottom: -10,
          left: 10,
          width: 'calc(100% - 20px)',
          height: 20,
        }}
        model={model}
      />
      {/** 左上角 */}
      <ResizeArea
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          cursor: 'nwse-resize',
        }}
        model={model}
        getDelta={({ x, y }) => ({ top: y, right: 0, bottom: 0, left: x })}
        onResize={onResize}
      />
      {/** 右上角 */}
      <ResizeArea
        style={{
          position: 'absolute',
          right: 0,
          top: 0,
          cursor: 'nesw-resize',
        }}
        model={model}
        getDelta={({ x, y }) => ({ top: y, right: x, bottom: 0, left: 0 })}
        onResize={onResize}
      />
      {/** 右下角 */}
      <ResizeArea
        style={{
          position: 'absolute',
          right: 0,
          bottom: 0,
          cursor: 'nwse-resize',
        }}
        model={model}
        getDelta={({ x, y }) => ({ top: 0, right: x, bottom: y, left: 0 })}
        onResize={onResize}
      />
      {/** 左下角 */}
      <ResizeArea
        style={{
          position: 'absolute',
          left: 0,
          bottom: 0,
          cursor: 'nesw-resize',
        }}
        model={model}
        getDelta={({ x, y }) => ({ top: 0, right: 0, bottom: y, left: x })}
        onResize={onResize}
      />
    </div>
  );
};

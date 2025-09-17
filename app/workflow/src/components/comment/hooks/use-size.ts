import { useCallback, useEffect, useState } from 'react';

import {
  FlowNodeFormData,
  FormModelV2,
  FreeOperationType,
  HistoryService,
  TransformData,
  useCurrentEntity,
  usePlayground,
  useService,
} from '@flowgram.ai/free-layout-editor';

import { CommentEditorFormField } from '../constant';

export const useSize = () => {
  const node = useCurrentEntity();
  const nodeMeta = node.getNodeMeta();
  const playground = usePlayground();
  const historyService = useService(HistoryService);
  //const { size = { width: 200, height: 150 } } = nodeMeta;
  const  size = { width: 200, height: 80 }
  const transform = node.getData(TransformData);
  const formModel = node.getData(FlowNodeFormData).getFormModel<FormModelV2>();
  const formSize = formModel.getValueIn<{ width: number; height: number }>(
    CommentEditorFormField.Size
  );

  const [width, setWidth] = useState(formSize?.width ?? size.width);
  const [height, setHeight] = useState(formSize?.height ?? size.height);

  // 初始化表单值
  useEffect(() => {
    const initSize = formModel.getValueIn<{ width: number; height: number }>(
      CommentEditorFormField.Size
    );
    if (!initSize) {
      formModel.setValueIn(CommentEditorFormField.Size, {
        width,
        height,
      });
    }
  }, [formModel, width, height]);

  // 同步表单外部值变化：初始化/undo/redo/协同
  useEffect(() => {
    const disposer = formModel.onFormValuesChange(({ name }) => {
      if (name !== CommentEditorFormField.Size) {
        return;
      }
      const newSize = formModel.getValueIn<{ width: number; height: number }>(
        CommentEditorFormField.Size
      );
      if (!newSize) {
        return;
      }
      setWidth(newSize.width);
      setHeight(newSize.height);
    });
    return () => disposer.dispose();
  }, [formModel]);

  const onResize = useCallback(() => {
    const resizeState = {
      width,
      height,
      originalWidth: width,
      originalHeight: height,
      positionX: transform.position.x,
      positionY: transform.position.y,
      offsetX: 0,
      offsetY: 0,
    };
    const resizing = (delta: { top: number; right: number; bottom: number; left: number }) => {
      if (!resizeState) {
        return;
      }

      const { zoom } = playground.config;

      const top = delta.top / zoom;
      const right = delta.right / zoom;
      const bottom = delta.bottom / zoom;
      const left = delta.left / zoom;

      const minWidth = 120;
      const minHeight = 80;

      const newWidth = Math.max(minWidth, resizeState.originalWidth + right - left);
      const newHeight = Math.max(minHeight, resizeState.originalHeight + bottom - top);

      // 如果宽度或高度小于最小值，则不更新偏移量
      const newOffsetX =
        (left > 0 || right < 0) && newWidth <= minWidth
          ? resizeState.offsetX
          : left / 2 + right / 2;
      const newOffsetY =
        (top > 0 || bottom < 0) && newHeight <= minHeight ? resizeState.offsetY : top;

      const newPositionX = resizeState.positionX + newOffsetX;
      const newPositionY = resizeState.positionY + newOffsetY;

      resizeState.width = newWidth;
      resizeState.height = newHeight;
      resizeState.offsetX = newOffsetX;
      resizeState.offsetY = newOffsetY;

      // 更新状态
      setWidth(newWidth);
      setHeight(newHeight);

      // 更新偏移量
      transform.update({
        position: {
          x: newPositionX,
          y: newPositionY,
        },
      });
    };

    const resizeEnd = () => {
      historyService.transact(() => {
        historyService.pushOperation(
          {
            type: FreeOperationType.dragNodes,
            value: {
              ids: [node.id],
              value: [
                {
                  x: resizeState.positionX + resizeState.offsetX,
                  y: resizeState.positionY + resizeState.offsetY,
                },
              ],
              oldValue: [
                {
                  x: resizeState.positionX,
                  y: resizeState.positionY,
                },
              ],
            },
          },
          {
            noApply: true,
          }
        );
        formModel.setValueIn(CommentEditorFormField.Size, {
          width: resizeState.width,
          height: resizeState.height,
        });
      });
    };

    return {
      resizing,
      resizeEnd,
    };
  }, [node, width, height, transform, playground, formModel, historyService]);

  return {
    width,
    height,
    onResize,
  };
};

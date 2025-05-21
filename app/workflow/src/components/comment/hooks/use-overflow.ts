import { useCallback, useState, useEffect } from 'react';

import { usePlayground } from '@flowgram.ai/free-layout-editor';

import { CommentEditorModel } from '../model';
import { CommentEditorEvent } from '../constant';

export const useOverflow = (params: { model: CommentEditorModel; height: number }) => {
  const { model, height } = params;
  const playground = usePlayground();

  const [overflow, setOverflow] = useState(false);

  const isOverflow = useCallback((): boolean => {
    if (!model.element) {
      return false;
    }
    return model.element.scrollHeight > model.element.clientHeight;
  }, [model, height, playground]);

  // 更新 overflow
  const updateOverflow = useCallback(() => {
    setOverflow(isOverflow());
  }, [isOverflow]);

  // 监听高度变化
  useEffect(() => {
    updateOverflow();
  }, [height, updateOverflow]);

  // 监听 change 事件
  useEffect(() => {
    const changeDisposer = model.on((params) => {
      if (params.type !== CommentEditorEvent.Change) {
        return;
      }
      updateOverflow();
    });
    return () => {
      changeDisposer.dispose();
    };
  }, [model, updateOverflow]);

  return { overflow, updateOverflow };
};

import { type FC, type CSSProperties, useEffect, useRef, useState, useMemo } from 'react';

import { usePlayground } from '@flowgram.ai/free-layout-editor';

import { CommentEditorModel } from '../model';
import { CommentEditorEvent } from '../constant';

interface ICommentEditor {
  model: CommentEditorModel;
  style?: CSSProperties;
  value?: string;
  onChange?: (value: string) => void;
}

export const CommentEditor: FC<ICommentEditor> = (props) => {
  const { model, style, onChange } = props;
  const playground = usePlayground();
  const editorRef = useRef<HTMLTextAreaElement | null>(null);
  const placeholder = model.value || model.focused ? undefined : 'Enter a comment...';

  // 同步编辑器内部值变化
  useEffect(() => {
    const disposer = model.on((params) => {
      if (params.type !== CommentEditorEvent.Change) {
        return;
      }
      onChange?.(model.value);
    });
    return () => disposer.dispose();
  }, [model, onChange]);

  useEffect(() => {
    if (!editorRef.current) {
      return;
    }
    model.element = editorRef.current;
  }, [editorRef]);

  return (
    <div className="workflow-comment-editor">
      <p className="workflow-comment-editor-placeholder">{placeholder}</p>
      <textarea
        className="workflow-comment-editor-textarea"
        ref={editorRef}
        style={style}
        readOnly={playground.config.readonly}
        onChange={(e) => {
          const { value } = e.target;
          model.setValue(value);
        }}
        onFocus={() => {
          model.setFocus(true);
        }}
        onBlur={() => {
          model.setFocus(false);
        }}
      />
    </div>
  );
};

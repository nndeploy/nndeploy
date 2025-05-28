import type { CommentEditorEvent } from './constant';

interface CommentEditorChangeEvent {
  type: CommentEditorEvent.Change;
  value: string;
}

interface CommentEditorMultiSelectEvent {
  type: CommentEditorEvent.MultiSelect;
}

interface CommentEditorSelectEvent {
  type: CommentEditorEvent.Select;
}

interface CommentEditorBlurEvent {
  type: CommentEditorEvent.Blur;
}

export type CommentEditorEventParams =
  | CommentEditorChangeEvent
  | CommentEditorMultiSelectEvent
  | CommentEditorSelectEvent
  | CommentEditorBlurEvent;

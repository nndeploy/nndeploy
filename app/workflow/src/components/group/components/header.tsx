import type { FC, ReactNode, MouseEvent, CSSProperties } from 'react';

import { useWatch } from '@flowgram.ai/free-layout-editor';

import { GroupField } from '../constant';
import { defaultColor, groupColors } from '../color';

interface GroupHeaderProps {
  onMouseDown: (e: MouseEvent) => void;
  onFocus: () => void;
  onBlur: () => void;
  children: ReactNode;
  style?: CSSProperties;
}

export const GroupHeader: FC<GroupHeaderProps> = ({
  onMouseDown,
  onFocus,
  onBlur,
  children,
  style,
}) => {
  const colorName = useWatch<string>(GroupField.Color) ?? defaultColor;
  const color = groupColors[colorName];
  return (
    <div
      className="workflow-group-header"
      data-flow-editor-selectable="false"
      onMouseDown={onMouseDown}
      onFocus={onFocus}
      onBlur={onBlur}
      style={{
        ...style,
        backgroundColor: color['50'],
        borderColor: color['300'],
      }}
    >
      {children}
    </div>
  );
};

import { CSSProperties, FC, useEffect } from 'react';

import { useWatch, WorkflowNodeEntity } from '@flowgram.ai/free-layout-editor';

import { GroupField } from '../constant';
import { defaultColor, groupColors } from '../color';

interface GroupBackgroundProps {
  node: WorkflowNodeEntity;
  style?: CSSProperties;
}

export const GroupBackground: FC<GroupBackgroundProps> = ({ node, style }) => {
  const colorName = useWatch<string>(GroupField.Color) ?? defaultColor;
  const color = groupColors[colorName];

  useEffect(() => {
    const styleElement = document.createElement('style');

    // 使用独特的选择器
    const styleContent = `
      .workflow-group-render[data-group-id="${node.id}"] .workflow-group-background {
        border: 1px solid ${color['300']};
      }

      .workflow-group-render.selected[data-group-id="${node.id}"] .workflow-group-background {
        border: 1px solid ${color['400']};
      }
    `;

    styleElement.textContent = styleContent;
    document.head.appendChild(styleElement);

    return () => {
      styleElement.remove();
    };
  }, [color]);

  return (
    <div
      className="workflow-group-background"
      data-flow-editor-selectable="true"
      style={{
        ...style,
        backgroundColor: `${color['300']}29`,
      }}
    />
  );
};

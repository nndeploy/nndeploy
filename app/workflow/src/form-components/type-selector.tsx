import React from 'react';

import { VariableTypeIcons } from '@flowgram.ai/form-materials';
import { Tag, Dropdown } from '@douyinfe/semi-ui';

export interface TypeSelectorProps {
  value?: string;
  disabled?: boolean;
  onChange?: (value?: string) => void;
  style?: React.CSSProperties;
}
const dropdownMenus = ['object', 'boolean', 'array', 'string', 'integer', 'number'];

export const TypeSelector: React.FC<TypeSelectorProps> = (props) => {
  const { value, disabled } = props;
  const icon = VariableTypeIcons[value as any];
  return (
    <Dropdown
      trigger="hover"
      position="bottomRight"
      disabled={disabled}
      render={
        <Dropdown.Menu>
          {dropdownMenus.map((key) => (
            <Dropdown.Item
              key={key}
              onClick={() => {
                props.onChange?.(key);
              }}
            >
              {VariableTypeIcons[key]}
              <span style={{ paddingLeft: '4px' }}>{key}</span>
            </Dropdown.Item>
          ))}
        </Dropdown.Menu>
      }
    >
      <Tag
        color="white"
        style={props.style}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
        }}
      >
        {icon}
      </Tag>
    </Dropdown>
  );
};

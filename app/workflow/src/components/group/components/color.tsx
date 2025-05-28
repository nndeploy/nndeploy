import { FC } from 'react';

import { Field } from '@flowgram.ai/free-layout-editor';
import { Popover, Tooltip } from '@douyinfe/semi-ui';

import { GroupField } from '../constant';
import { defaultColor, groupColors } from '../color';

export const GroupColor: FC = () => (
  <Field<string> name={GroupField.Color}>
    {({ field }) => {
      const colorName = field.value ?? defaultColor;
      return (
        <Popover
          position="top"
          mouseLeaveDelay={300}
          content={
            <div className="workflow-group-color-palette">
              {Object.entries(groupColors).map(([name, color]) => (
                <Tooltip content={name} key={name} mouseEnterDelay={300}>
                  <span
                    className="workflow-group-color-item"
                    key={name}
                    style={{
                      backgroundColor: color['300'],
                      borderColor: name === colorName ? color['400'] : '#fff',
                    }}
                    onClick={() => field.onChange(name)}
                  />
                </Tooltip>
              ))}
            </div>
          }
        >
          <span
            className="workflow-group-color"
            style={{
              backgroundColor: groupColors[colorName]['300'],
            }}
          />
        </Popover>
      );
    }}
  </Field>
);

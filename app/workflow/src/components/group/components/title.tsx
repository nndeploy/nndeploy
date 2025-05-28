import { FC, useState } from 'react';

import { Field } from '@flowgram.ai/free-layout-editor';
import { Input } from '@douyinfe/semi-ui';

import { GroupField } from '../constant';

export const GroupTitle: FC = () => {
  const [inputting, setInputting] = useState(false);
  return (
    <Field<string> name={GroupField.Title}>
      {({ field }) =>
        inputting ? (
          <Input
            autoFocus
            className="workflow-group-title-input"
            size="small"
            value={field.value}
            onChange={field.onChange}
            onMouseDown={(e) => e.stopPropagation()}
            onBlur={() => setInputting(false)}
            draggable={false}
            onEnterPress={() => setInputting(false)}
          />
        ) : (
          <p className="workflow-group-title" onDoubleClick={() => setInputting(true)}>
            {field.value ?? 'Group'}
          </p>
        )
      }
    </Field>
  );
};

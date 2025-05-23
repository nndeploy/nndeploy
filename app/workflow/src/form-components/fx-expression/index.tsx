import React, { type SVGProps } from 'react';

import { VariableSelector } from '@flowgram.ai/form-materials';
import { Input, Button } from '@douyinfe/semi-ui';

import { ValueDisplay } from '../value-display';
import { FlowRefValueSchema, FlowLiteralValueSchema } from '../../typings';

export function FxIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 16 16" {...props}>
      <path
        fill="currentColor"
        fillRule="evenodd"
        d="M5.581 4.49A2.75 2.75 0 0 1 8.319 2h.931a.75.75 0 0 1 0 1.5h-.931a1.25 1.25 0 0 0-1.245 1.131l-.083.869H9.25a.75.75 0 0 1 0 1.5H6.849l-.43 4.51A2.75 2.75 0 0 1 3.681 14H2.75a.75.75 0 0 1 0-1.5h.931a1.25 1.25 0 0 0 1.245-1.132L5.342 7H3.75a.75.75 0 0 1 0-1.5h1.735zM9.22 9.22a.75.75 0 0 1 1.06 0l1.22 1.22l1.22-1.22a.75.75 0 1 1 1.06 1.06l-1.22 1.22l1.22 1.22a.75.75 0 1 1-1.06 1.06l-1.22-1.22l-1.22 1.22a.75.75 0 1 1-1.06-1.06l1.22-1.22l-1.22-1.22a.75.75 0 0 1 0-1.06"
        clipRule="evenodd"
      ></path>
    </svg>
  );
}

function InputWrap({
  value,
  onChange,
  readonly,
  hasError,
  style,
}: {
  value: string;
  onChange: (v: string) => void;
  readonly?: boolean;
  hasError?: boolean;
  style?: React.CSSProperties;
}) {
  if (readonly) {
    return <ValueDisplay value={value} hasError={hasError} />;
  }
  return (
    <Input
      value={value as string}
      onChange={onChange}
      validateStatus={hasError ? 'error' : undefined}
      style={style}
    />
  );
}

export interface FxExpressionProps {
  value?: FlowLiteralValueSchema | FlowRefValueSchema;
  onChange: (value: FlowLiteralValueSchema | FlowRefValueSchema) => void;
  literal?: boolean;
  hasError?: boolean;
  readonly?: boolean;
  icon?: React.ReactNode;
}

export function FxExpression(props: FxExpressionProps) {
  const { value, onChange, readonly, literal, icon } = props;
  if (literal) return <InputWrap value={value as string} onChange={onChange} readonly={readonly} />;
  const isExpression = typeof value === 'object' && value.type === 'expression';
  const toggleExpression = () => {
    if (isExpression) {
      onChange((value as FlowRefValueSchema).content as string);
    } else {
      onChange({ content: value as string, type: 'expression' });
    }
  };
  return (
    <div style={{ display: 'flex', maxWidth: 300 }}>
      {isExpression ? (
        <VariableSelector
          value={value.content}
          hasError={props.hasError}
          style={{ flexGrow: 1 }}
          onChange={(v) => onChange({ type: 'expression', content: v })}
          readonly={readonly}
        />
      ) : (
        <InputWrap
          value={value as string}
          onChange={onChange}
          hasError={props.hasError}
          readonly={readonly}
          style={{ flexGrow: 1, outline: props.hasError ? '1px solid red' : undefined }}
        />
      )}
      {!readonly &&
        (icon || <Button theme="borderless" icon={<FxIcon />} onClick={toggleExpression} />)}
    </div>
  );
}

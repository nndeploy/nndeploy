// import { TypeTag } from '../type-tag'
import { ValueDisplayStyle } from './styles';

export interface ValueDisplayProps {
  value: string;
  placeholder?: string;
  hasError?: boolean;
}

export const ValueDisplay: React.FC<ValueDisplayProps> = (props) => (
  <ValueDisplayStyle className={props.hasError ? 'has-error' : ''}>
    {props.value}
    {props.value === undefined || props.value === '' ? (
      <span style={{ color: 'var(--semi-color-text-2)' }}>{props.placeholder || '--'}</span>
    ) : null}
  </ValueDisplayStyle>
);

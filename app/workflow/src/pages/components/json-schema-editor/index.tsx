import React, { useMemo, useState } from 'react';

import { Button, Checkbox, IconButton, Input } from '@douyinfe/semi-ui';
import {
  IconExpand,
  IconShrink,
  IconPlus,
  IconChevronDown,
  IconChevronRight,
  IconMinus,
} from '@douyinfe/semi-icons';

import { JsonSchema } from '../type-selector/types';
import { TypeSelector } from '../type-selector';
import { PropertyValueType } from './types';
import {
  IconAddChildren,
  UIActions,
  UICollapseTrigger,
  UICollapsible,
  UIContainer,
  UIExpandDetail,
  UILabel,
  UIProperties,
  UIPropertyLeft,
  UIPropertyMain,
  UIPropertyRight,
  UIRequired,
  UIType,
} from './styles';
import { UIName } from './styles';
import { UIRow } from './styles';
import { usePropertiesEdit } from './hooks';

export function JsonSchemaEditor(props: {
  value?: JsonSchema;
  onChange?: (value: JsonSchema) => void;
}) {
  const { value = { type: 'object' }, onChange: onChangeProps } = props;
  const { propertyList, onAddProperty, onRemoveProperty, onEditProperty } = usePropertiesEdit(
    value,
    onChangeProps
  );

  return (
    <UIContainer>
      <UIProperties>
        {propertyList.map((_property) => (
          <PropertyEdit
            key={_property.key}
            value={_property}
            onChange={(_v) => {
              onEditProperty(_property.key!, _v);
            }}
            onRemove={() => {
              onRemoveProperty(_property.key!);
            }}
          />
        ))}
      </UIProperties>
      <Button size="small" style={{ marginTop: 10 }} icon={<IconPlus />} onClick={onAddProperty}>
        Add
      </Button>
    </UIContainer>
  );
}

function PropertyEdit(props: {
  value?: PropertyValueType;
  onChange?: (value: PropertyValueType) => void;
  onRemove?: () => void;
  $isLast?: boolean;
  $showLine?: boolean;
}) {
  const { value, onChange: onChangeProps, onRemove, $isLast, $showLine } = props;

  const [expand, setExpand] = useState(false);
  const [collapse, setCollapse] = useState(false);

  const { name, type, items, description, isPropertyRequired } = value || {};

  const typeSelectorValue = useMemo(() => ({ type, items }), [type, items]);

  const { propertyList, isDrilldownObject, onAddProperty, onRemoveProperty, onEditProperty } =
    usePropertiesEdit(value, onChangeProps);

  const onChange = (key: string, _value: any) => {
    onChangeProps?.({
      ...(value || {}),
      [key]: _value,
    });
  };

  const showCollapse = isDrilldownObject && propertyList.length > 0;

  return (
    <>
      <UIPropertyLeft $isLast={$isLast} $showLine={$showLine}>
        {showCollapse && (
          <UICollapseTrigger onClick={() => setCollapse((_collapse) => !_collapse)}>
            {collapse ? <IconChevronDown size="small" /> : <IconChevronRight size="small" />}
          </UICollapseTrigger>
        )}
      </UIPropertyLeft>
      <UIPropertyRight>
        <UIPropertyMain $expand={expand}>
          <UIRow>
            <UIName>
              <Input
                placeholder="Input Variable Name"
                size="small"
                value={name}
                onChange={(value) => onChange('name', value)}
              />
            </UIName>
            <UIType>
              <TypeSelector
                value={typeSelectorValue}
                onChange={(_value) => {
                  onChangeProps?.({
                    ...(value || {}),
                    ..._value,
                  });
                }}
              />
            </UIType>
            <UIRequired>
              <Checkbox
                checked={isPropertyRequired}
                onChange={(e) => onChange('isPropertyRequired', e.target.checked)}
              />
            </UIRequired>
            <UIActions>
              <IconButton
                size="small"
                theme="borderless"
                icon={expand ? <IconShrink size="small" /> : <IconExpand size="small" />}
                onClick={() => setExpand((_expand) => !_expand)}
              />
              {isDrilldownObject && (
                <IconButton
                  size="small"
                  theme="borderless"
                  icon={<IconAddChildren />}
                  onClick={() => {
                    onAddProperty();
                    setCollapse(true);
                  }}
                />
              )}
              <IconButton
                size="small"
                theme="borderless"
                icon={<IconMinus size="small" />}
                onClick={onRemove}
              />
            </UIActions>
          </UIRow>
          {expand && (
            <UIExpandDetail>
              <UILabel>Description</UILabel>
              <Input
                size="small"
                value={description}
                onChange={(value) => onChange('description', value)}
                placeholder="Help LLM to understand the property"
              />
            </UIExpandDetail>
          )}
        </UIPropertyMain>
        {showCollapse && (
          <UICollapsible $collapse={collapse}>
            <UIProperties $shrink={true}>
              {propertyList.map((_property, index) => (
                <PropertyEdit
                  key={_property.key}
                  value={_property}
                  onChange={(_v) => {
                    onEditProperty(_property.key!, _v);
                  }}
                  onRemove={() => {
                    onRemoveProperty(_property.key!);
                  }}
                  $isLast={index === propertyList.length - 1}
                  $showLine={true}
                />
              ))}
            </UIProperties>
          </UICollapsible>
        )}
      </UIPropertyRight>
    </>
  );
}

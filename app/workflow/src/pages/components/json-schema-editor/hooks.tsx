import { useEffect, useMemo, useRef, useState } from 'react';

import { PropertyValueType } from './types';
import { JsonSchema } from '../type-selector';

let _id = 0;
function genId() {
  return _id++;
}

function getDrilldownSchema(
  value?: PropertyValueType,
  path?: (keyof PropertyValueType)[]
): { schema?: PropertyValueType | null; path?: (keyof PropertyValueType)[] } {
  if (!value) {
    return {};
  }

  if (value.type === 'array' && value.items) {
    return getDrilldownSchema(value.items, [...(path || []), 'items']);
  }

  return { schema: value, path };
}

export function usePropertiesEdit(
  value?: PropertyValueType,
  onChange?: (value: PropertyValueType) => void
) {
  // Get drilldown (array.items.items...)
  const drilldown = useMemo(() => getDrilldownSchema(value), [value, value?.type, value?.items]);

  const isDrilldownObject = drilldown.schema?.type === 'object';

  // Generate Init Property List
  const initPropertyList = useMemo(
    () =>
      isDrilldownObject
        ? Object.entries(drilldown.schema?.properties || {}).map(
            ([name, _value]) =>
              ({
                key: genId(),
                name,
                isPropertyRequired: drilldown.schema?.required?.includes(name) || false,
                ..._value,
              } as PropertyValueType)
          )
        : [],
    [isDrilldownObject]
  );

  const [propertyList, setPropertyList] = useState<PropertyValueType[]>(initPropertyList);

  const mountRef = useRef(false);

  useEffect(() => {
    // If initRef is true, it means the component has been mounted
    if (mountRef.current) {
      // If the value is changed, update the property list
      setPropertyList((_list) => {
        const nameMap = new Map<string, PropertyValueType>();

        for (const _property of _list) {
          if (_property.name) {
            nameMap.set(_property.name, _property);
          }
        }
        return Object.entries(drilldown.schema?.properties || {}).map(([name, _value]) => {
          const _property = nameMap.get(name);
          if (_property) {
            return {
              key: _property.key,
              name,
              isPropertyRequired: drilldown.schema?.required?.includes(name) || false,
              ..._value,
            };
          }
          return {
            key: genId(),
            name,
            isPropertyRequired: drilldown.schema?.required?.includes(name) || false,
            ..._value,
          };
        });
      });
    }
    mountRef.current = true;
  }, [drilldown.schema]);

  const updatePropertyList = (updater: (list: PropertyValueType[]) => PropertyValueType[]) => {
    setPropertyList((_list) => {
      const next = updater(_list);

      // onChange to parent
      const nextProperties: Record<string, JsonSchema> = {};
      const nextRequired: string[] = [];

      for (const _property of next) {
        if (!_property.name) {
          continue;
        }

        nextProperties[_property.name] = _property;

        if (_property.isPropertyRequired) {
          nextRequired.push(_property.name);
        }
      }

      let drilldownSchema = value || {};
      if (drilldown.path) {
        drilldownSchema = drilldown.path.reduce((acc, key) => acc[key], value || {});
      }
      drilldownSchema.properties = nextProperties;
      drilldownSchema.required = nextRequired;

      onChange?.(value || {});

      return next;
    });
  };

  const onAddProperty = () => {
    updatePropertyList((_list) => [..._list, { key: genId(), name: '', type: 'string' }]);
  };

  const onRemoveProperty = (key: number) => {
    updatePropertyList((_list) => _list.filter((_property) => _property.key !== key));
  };

  const onEditProperty = (key: number, nextValue: PropertyValueType) => {
    updatePropertyList((_list) =>
      _list.map((_property) => (_property.key === key ? nextValue : _property))
    );
  };

  useEffect(() => {
    if (!isDrilldownObject) {
      setPropertyList([]);
    }
  }, [isDrilldownObject]);

  return {
    propertyList,
    isDrilldownObject,
    onAddProperty,
    onRemoveProperty,
    onEditProperty,
  };
}

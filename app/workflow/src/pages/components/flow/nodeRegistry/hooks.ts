import { useState } from "react";
import { getFieldType } from "../functions";

let _id = 0;
function genId() {
  return _id++;
}

interface PropertyValueType {
  key: number;
  name: string;
  isArrayItem?: boolean,
  value: any

}

export function usePropertiesEdit(
  fieldName: string,
  parentPaths: string[],
  value: any,
  form: any,
  nodeList: any,
  paramTypes: any,
  onChange?: (value: any) => void
) {
  // Get drilldown (array.items.items...)

  const fieldType = getFieldType([...parentPaths, fieldName], form, nodeList, paramTypes)

  const isDrilldownObject = fieldType.isArray || fieldType.componentType == 'object';


  let initPropertyList: PropertyValueType[] = []

  if (fieldType.isArray) {
    initPropertyList = (value as any[]).map((item) => {

      return {
        key: genId(),
        name: '',
        isArrayItem: true,
        value: item
      }
    })
  } else if (fieldType.componentType == 'object') {
    initPropertyList = Object.entries(value).map(([name, _value]) => {
      return {
        key: genId(),
        name,
        value: _value,
        isArrayItem: false
      }
    })
  } else {
    initPropertyList = []
  }



  const [propertyList, setPropertyList] = useState<PropertyValueType[]>(initPropertyList);


  const updatePropertyList = (updater: (list: PropertyValueType[]) => PropertyValueType[]) => {
    setPropertyList((_list) => {
      const next = updater(_list);

      // onChange to parent
      //const nextProperties: Record<string, PropertyValueType> = {};

      if (fieldType.isArray) {
        onChange?.(next.map(item => item.value))
      } else {

        for (const _property of next) {
          if (!_property.name) {
            continue;
          }

          //nextProperties[_property.name] = _property;

          value[_property.name] = _property.value
        }
        onChange?.(value || {});
      }

      return next;
    });
  };

  const onAddProperty = (value?: any) => {
    updatePropertyList((_list) => [..._list, { key: genId(), name: '', value: value ?? '', isArrayItem: true }]);
  };

  const onAddObjectProperty = (name:string, value?: any) => {
    updatePropertyList((_list) => [..._list, { key: genId(), name, value: value ?? '', isArrayItem: false }]);

  };

  const onRemoveProperty = (key: number) => {
    updatePropertyList((_list) => _list.filter((_property) => _property.key !== key));
  };

  const onEditProperty = (key: number, nextValue: Partial<PropertyValueType>) => {
    updatePropertyList((_list) =>
      _list.map((_property) => (_property.key === key ? {..._property, ...nextValue }: _property))
    );
  };

  return {
    propertyList,
    isDrilldownObject,
    onAddProperty,
    onAddObjectProperty, 
    onRemoveProperty,
    onEditProperty,
  };
}

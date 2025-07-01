import { IFieldType, IParamTypes } from "../../Layout/Design/WorkFlow/entity";
import { INodeEntity } from "../../Node/entity";
import lodash from 'lodash'

function getNodeRegistry(form: any, nodeList: INodeEntity[]) {
  const registryKey = form.values['key_']
  const nodeRegistry = nodeList.find(item => item.key_ == registryKey)
  return nodeRegistry!

}


function isNumberArrayFields(fieldNames: string[]): boolean {
  const lastFieldName = fieldNames[fieldNames.length - 1]

  const numberArrayFields = ["scale_", "mean_", "std_"];
  return numberArrayFields.includes(lastFieldName);
}
export function getFieldType(fieldNames: string[], form: any, nodeList: INodeEntity[], paramTypes: IParamTypes): IFieldType {

  var result: IFieldType = {
    isArray: false,
    componentType: 'string',
    primateType: 'string',
    selectOptions: [],
    selectKey: ''
  }

  if (isNumberArrayFields(fieldNames)) {
    return result = {
      isArray: true,
      primateType: 'number',
      componentType: 'number',
      selectOptions: [],
      selectKey: ''
    }
  }
  const nodeRegistry = getNodeRegistry(form, nodeList)

  let fieldValue: any = nodeRegistry

  for (let i = 0; i < fieldNames.length; i++) {
    let fieldName = fieldNames[i]
    fieldValue = fieldValue[fieldName]
  }

  //const fieldValue = nodeRegistry['param_']![fieldName]

  if (lodash.isArray(fieldValue)) {
    result.isArray = true;
    if (lodash.isNumber(fieldValue[0])) {
      result.componentType = 'number'
      result.primateType = 'number'

    }
    if (lodash.isBoolean(fieldValue[0])) {
      result.componentType = 'boolean'
      result.primateType = 'boolean'
    }
    if (lodash.isString(fieldValue[0])) {
      result.componentType = 'string'
      result.primateType = 'string'
    }

  } else {
    if (lodash.isNumber(fieldValue)) {
      result.componentType = 'number'
      result.primateType = 'number'
    }
    if (lodash.isBoolean(fieldValue)) {
      result.componentType = 'boolean'
      result.primateType = 'boolean'
    }
    if (lodash.isString(fieldValue)) {
      result.componentType = 'string'
      result.primateType = 'string'
    }
  }

  // function isParamField(){
  //   return fieldNames.length == 2 && fieldNames[1] == 'params_'
  // }

  if (lodash.isString(fieldValue) && paramTypes.hasOwnProperty(fieldValue)) {
    result.componentType = 'select'
    result.selectOptions = paramTypes[fieldValue]
    result.selectKey = fieldValue
  }


  return result

}
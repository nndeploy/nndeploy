import { INodeEntity } from "../../Node/entity";
import lodash from 'lodash'

function getNodeRegistry(form: any, nodeList: INodeEntity[]) {
  const registryKey = form.values['key_']
  const nodeRegistry = nodeList.find(item => item.key_ == registryKey)
  return nodeRegistry!

}


function isNumberArrayFields(fieldNames: string[]): boolean {
  const lastFieldName = fieldNames[fieldNames.length -1]

  const numberArrayFields = ["scale_", "mean_", "std_"];
  return numberArrayFields.includes(lastFieldName);
}
export function getFieldType(fieldNames: string[], form: any, nodeList: INodeEntity[]) {

  var result = {
    isArray: false,
    primateType: 'string'
  }

  if (isNumberArrayFields(fieldNames)) {
    return result = {
      isArray: true,
      primateType: 'number'
    }
  }
  const nodeRegistry = getNodeRegistry(form, nodeList)

  let  fieldValue 

  for(let i = 0; i < fieldNames.length; i++){
      let fieldName = fieldNames[i]
      fieldValue = nodeRegistry[fieldName]
  }

  //const fieldValue = nodeRegistry['param_']![fieldName]

  if (lodash.isArray(fieldValue)) {
    result.isArray = true;
    if (lodash.isNumber(fieldValue[0])) {
      result.primateType = 'number'
    }
    if (lodash.isBoolean(fieldValue[0])) {
      result.primateType = 'boolean'
    }
    if (lodash.isString(fieldValue[0])) {
      result.primateType = 'string'
    }

  } else {
    if (lodash.isNumber(fieldValue)) {
      result.primateType = 'number'
    }
    if (lodash.isBoolean(fieldValue)) {
      result.primateType = 'boolean'
    }
    if (lodash.isString(fieldValue)) {
      result.primateType = 'string'
    }
  }

  return result

}
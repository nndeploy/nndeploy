
import { FlowDocumentJSON } from "../../../typings";
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
    if (fieldValue == null) {
      let i = 0
    }

    // if(fieldName == '1'){
    //   debugger
    // }

    if (lodash.isNumber(new Number(fieldName)) && !fieldValue[fieldName] && fieldValue[0]) {

      fieldValue = fieldValue[0]
    } else {
      fieldValue = fieldValue[fieldName]
    }


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

  } else if (lodash.isObject(fieldValue)) {
    result.componentType = 'object'
    result.primateType = 'object'
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

export function getNextNameNumberSuffix(documentJSON: FlowDocumentJSON) {
  let result = 0;
  //const allNode = ref?.current?.document.toJSON() as FlowDocumentJSON;
  documentJSON.nodes.map(item => {

    var nameParts = item.data.name_.split('_')
    if (item.data.name_ && nameParts.length > 1) {
      var numberPart = parseInt(nameParts[nameParts.length - 1])
      if (!isNaN(numberPart)) {
        result = Math.max(result, numberPart);
      }
    }
  })
  return result + 1;
}
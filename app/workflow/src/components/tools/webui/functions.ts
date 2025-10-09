
import lodash from 'lodash'
import { Map } from 'immutable';
export function getIoTypeFieldName(node: any) {

  const parents = ['', 'param_', 'param']
  for (const parent of parents) {

    const params = ['io_param_', 'required_params_']

    for (const param of params) {
      let ioPosition = parent ? node[parent][param] : node[param]
      if (ioPosition && ioPosition.length > 0) {
        return (parent ? parent + '.' : '') + ioPosition[0]
      }
    }
  }

  return "empty-field"
}

export function getIoTypeFieldValue(node: any) {


  const name = getIoTypeFieldName(node)
  const nameParts = name.split('.')

  let value = node
  for (const part of nameParts) {
    value = value[part]
    if (lodash.isEmpty(value)) {
      break
    }
  }

  return value

}

export function setNodeFieldValue(node: any, fieldName: string, value: any) {

  const map = Map(node);
  const fieldNames = fieldName.split('.')
  const newMap = map.setIn(fieldNames, value);

  return newMap.toObject()
}


import type { IOutputFile, IOutputResource } from '../../../pages/components/flow/entity';

import lodash from 'lodash'
import { Map } from 'immutable';

export function getBooleanOutput(value?: string) {
  return value?.trim().toLowerCase() === 'true';
}

export function getIoTypeFieldName(node: any) {

  const parents = ['', 'param_', 'param']
  for (const parent of parents) {

    const params = ['io_param_', 'io_params_', 'required_params_']

    for (const param of params) {
      let ioPosition = parent ? node[parent]?.[param] : node[param]
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


/** Match the complete path so nested nodes with identical names never collide. */
export function getOutputFileUrl(output: IOutputResource, nodePath: string[], fieldPath?: (string | number)[]) {
  const matches = output.files?.filter(file => file.node_path.length === nodePath.length
    && file.node_path.every((name, index) => name === nodePath[index])) ?? [];
  if (fieldPath) {
    const exact = matches.find(file => file.field_path?.length === fieldPath.length
      && file.field_path.every((name, index) => name === fieldPath[index]));
    // Old task history has only the node's original primary preview.
    return (exact ?? matches.find(file => !file.field_path))?.url;
  }
  return matches.length === 1 ? matches[0].url : undefined;
}

export function getOutputFileLabel(file: IOutputFile) {
  const field = file.field_path?.reduce<string>((label, part) => typeof part === 'number'
    ? `${label}[${part}]` : `${label ? label + '.' : ''}${part}`, '');
  return [...file.node_path, ...(field ? [field] : [])].join(' / ');
}

/** Canvas node names are local to their containing Graph. */
export function getOutputNodePath(node: any): string[] {
  const path: string[] = [];
  for (let current = node; current; current = current.parent) {
    const name = current.form?.getValueIn('name_');
    if (typeof name === 'string' && name) path.unshift(name);
  }
  return path;
}

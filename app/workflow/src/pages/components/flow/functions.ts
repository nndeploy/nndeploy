
import { FlowNodeEntity, FreeLayoutPluginContext } from "@flowgram.ai/free-layout-editor";
import { FlowDocumentJSON, FlowNodeJSON } from "../../../typings";
import { IFieldType, IParamTypes } from "../../Layout/Design/WorkFlow/entity";
import { INodeEntity } from "../../Node/entity";
import lodash from 'lodash'
import { IExpandInfo } from "./entity";

function getNodeRegistry(registryKey: any, nodeList: INodeEntity[]) {
  //const registryKey = form.values['key_']
  const nodeRegistry = nodeList.find(item => item.key_ == registryKey)
  return nodeRegistry!

}


function isNumberArrayFields(fieldNames: string[]): boolean {
  const lastFieldName = fieldNames[fieldNames.length - 1]

  const numberArrayFields = ["scale_", "mean_", "std_"];
  return numberArrayFields.includes(lastFieldName);
}
export function getFieldType(fieldNames: string[], registryKey: any, nodeList: INodeEntity[], paramTypes: IParamTypes): IFieldType {

  var result: IFieldType = {
    isArray: false,
    componentType: 'string',
    primateType: 'string',
    selectOptions: [],
    selectKey: '',
    originValue: ''
  }

  if (fieldNames[fieldNames.length - 1] == 'color_mode_') {
    let j = 0;
  }

  const nodeRegistry = getNodeRegistry(registryKey, nodeList)

  let fieldValue: any = nodeRegistry
  let fieldName = ''
  for (let i = 0; i < fieldNames.length; i++) {
    fieldName = fieldNames[i]
    if (fieldValue == null) {
      let i = 0
    }

    // if(fieldName == '1'){
    //   debugger
    // }

    if (lodash.isNumber(new Number(fieldName)) && !fieldValue[fieldName] && fieldValue[0]) {

      fieldValue = fieldValue[0]
    } else {
      if (fieldName in fieldValue) {
        fieldValue = fieldValue[fieldName]
      } else {
        // let parts = fieldName.split("_") 
        //  if (parts.length > 1 && lodash.isNumber(new Number(parts[parts.length - 1]))) {

        //   let firstField = parts[0] + "_0" ; 
        //   if(firstField  in fieldValue ){ 
        //     fieldValue = fieldValue[firstField]
        //   }else{

        //   }
        // }
        const [firstKey] = Object.keys(fieldValue);
        fieldValue = fieldValue[firstKey]

      }

    }


  }

  if (isNumberArrayFields(fieldNames)) {
    return result = {
      isArray: true,
      primateType: 'number',
      componentType: 'number',
      selectOptions: [],
      selectKey: '',
      originValue: fieldValue
    }
  }

  result.originValue = fieldValue


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

  function getSelectOptions() {
    if (!lodash.isString(fieldName)) {
      return undefined
    }

    let parents: string[] = ['', 'param_', 'param']

    let options: any[] = []
    for (let parent of parents) {


      options = parent ? nodeRegistry[parent]?.['dropdown_params_']?.[fieldName]
        : nodeRegistry['dropdown_params_']?.[fieldName]

      if (options) {
        return options
      }

    }

    if (paramTypes.hasOwnProperty(fieldValue)) {
      options = paramTypes[fieldValue]
      return options
    }

    return undefined
  }

  const selectOptions = getSelectOptions()

  if (!result.isArray && selectOptions) { //lodash.isString(fieldValue) &&
    result.componentType = 'select'
    result.selectOptions = selectOptions
    result.selectKey = fieldValue
  }


  return result

}

export function getNextNameNumberSuffix(documentJSON: FlowDocumentJSON) {
  let result = 0;
  //const allNode = ref?.current?.document.toJSON() as FlowDocumentJSON;
  documentJSON.nodes.map(item => {

    if (!item.data.name_) {
      let j = 0;
    }
    if (item.type == 'group') {
      return
    }
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

// function buildContainerNodeInnerLines(node: FlowNodeJSON){

// }

export function isContainerNode(nodeId: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  let form = node?.form
  let result = (
    form?.getValueIn('is_graph_')

    || form?.getValueIn('is_loop_')
    || form?.getValueIn('is_composite_node_')
  )
    ?? false
  return result
}

export function isGraphNode(nodeId: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  let form = node?.form
  let result = (
    form?.getValueIn('is_graph_')

    || form?.getValueIn('is_loop_')

  )
    ?? false
  return result
}

export const freeGraphContainerKeys = ['nndeploy::dag::FixedLoop', 'nndeploy::dag::Graph']

export function isFxiedGraphNode(nodeId: string|undefined, clientContext: FreeLayoutPluginContext) {
  if(!nodeId){
    return false
  }
  let isDynamic = isGraphNode(nodeId, clientContext)

  let key = getNodeNamFieldValue(nodeId, 'key_', clientContext)

  if (isDynamic && !freeGraphContainerKeys.includes(key)) {
    return true
  }

  return false

}

export function isFreeGraphNode(nodeId: string, clientContext: FreeLayoutPluginContext) {

  let key = getNodeNamFieldValue(nodeId, 'key_', clientContext)

  if (freeGraphContainerKeys.includes(key)) {
    return true
  }

  return false

}


export function isLoopNode(nodeId: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  let form = node?.form
  let result = (
    form?.getValueIn('is_loop_')

  )
    ?? false
  return result
}

export function isCompositeNode(nodeId: string|undefined, clientContext: FreeLayoutPluginContext) {

  if(!nodeId){
    return false
  }
  let node = clientContext.document.getNode(nodeId)
  let form = node?.form
  let result = (
    form?.getValueIn('is_composite_node_')

  )
    ?? false
  return result
}

// export function isGraphNode(nodeId: string, clientContext: FreeLayoutPluginContext) {
//   let node = clientContext.document.getNode(nodeId)
//   let form = node?.form
//   let result = (
//     form?.getValueIn('is_graph_')

//   )
//     ?? false
//   return result
// }

export function isNodeOffspringOfCompositeNode(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {
  const parents = getNodeParents(node)
  return lodash.some(parents, (parent) => isCompositeNode(parent.id, clientContext))
}

export function getNodeParents(node: FlowNodeEntity) {
  const parents: FlowNodeEntity[] = []
  while (node.parent) {
    parents.push(node.parent)
    node = node.parent
  }
  return parents
}


export function isNodeOffSpringOfFixedGraph(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {


  let parents = getNodeParents(node)
  return parents.some((parent) => isFxiedGraphNode(parent.id, clientContext))

}



export function isbothNodeOffSpringOftheSameFixedGraph(first: FlowNodeEntity | undefined, second: FlowNodeEntity | undefined, clientContext: FreeLayoutPluginContext) {

  if (first == null || second == null) {
    return false
  }
  let firstParents = getNodeParents(first)
  firstParents = firstParents.filter((parent) => isFxiedGraphNode(parent.id, clientContext))


  let secondParents = getNodeParents(second)
  secondParents = secondParents.filter((parent) => isFxiedGraphNode(parent.id, clientContext))

  return lodash.intersection(firstParents, secondParents).length > 0
}

export function getNodeById(nodeId: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  return node
}
export function getNodeByName(name: string, clientContext: FreeLayoutPluginContext) {
  const allNodes = clientContext.document.getAllNodes()
  const find = allNodes.find(node => {
    return node.form?.getValueIn('name_') == name
  })

  return find
}

export function getAllInnerNodes(node: FlowNodeEntity) {
  let allChildren: FlowNodeEntity[] = []
  if (node.blocks && node.blocks.length > 0) {
    node.blocks.forEach((child) => {
      allChildren.push(child);
      allChildren = allChildren.concat(...getAllInnerNodes(child));
    });
  }
  return allChildren;
}

export function getNodeNameByNodeId(nodeId: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  let name = node?.form?.getValueIn('name_')
  return name
}

export function getNodeNamFieldValue(nodeId: string, fieldName: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  let result = node?.form?.getValueIn(fieldName)
  return result
}

export function getNodeExpandInfo(nodeId: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  let expandInfo: IExpandInfo | undefined = node?.getNodeMeta().expandInfo
  return expandInfo
}

export function nodeIterate<Node>(
  node: Node,
  childFieldName: string,
  process: (node: Node, parents: Node[]) => void,

  parents: Node[] = []
) {
  process(node, parents);
  if (node[childFieldName as keyof Node] && (node[childFieldName as keyof Node] as Node[])?.length > 0) {
    (node[childFieldName as keyof Node] as Node[]).forEach((child) => {
      nodeIterate(child, childFieldName, process, [...parents, child]);
    });
  }
}


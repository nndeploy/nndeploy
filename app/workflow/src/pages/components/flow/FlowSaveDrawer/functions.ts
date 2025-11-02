import { FlowNodeEntity, FreeLayoutPluginContext } from "@flowgram.ai/free-layout-editor";
import {
  IBusinessNode,
} from "../../../Layout/Design/WorkFlow/entity";
import { ILineEntity } from "../entity";
import { getAllInnerNodeIds, getNodeById, getNodeExpandInfo, getNodeParents } from "../functions";
import { getSubcavasInputLines, getSubcavasOutputLines } from "../form-header/function";


export function businessNodeIterate(
  node: IBusinessNode,
  process: (node: IBusinessNode, parents: IBusinessNode[]) => void,
  parents: IBusinessNode[] = []
) {
  process(node, parents);
  if (node.node_repository_ && node.node_repository_.length > 0) {
    node.node_repository_.forEach((respoitory) => {
      businessNodeIterate(respoitory, process, [...parents, node]);
    });
  }
}

export function businessNodeNormalize(businessNode: IBusinessNode) {
  businessNode.id = businessNode.id
    ? businessNode.id
    : "node_" + Math.random().toString(36).substr(2, 9);

  let inputs_ = businessNode.inputs_ ?? [];

  let inputCollectionNameIdMap: { [key: string]: string } = {};
  inputs_.map((item) => {
    if (!item.id) {
      let portId = "";
      if (inputCollectionNameIdMap[item.desc_]) {
        portId = inputCollectionNameIdMap[item.desc_];
      } else {
        portId = "port_" + Math.random().toString(36).substr(2, 9);
        inputCollectionNameIdMap[item.desc_] = portId;
      }

      item.id = portId;
    }
  });

  let outputCollectionNameIdMap: { [key: string]: string } = {};

  let outputs_ = businessNode.outputs_ ?? [];
  outputs_.map((item) => {
    if (!item.id) {
      let portId = "";
      if (outputCollectionNameIdMap[item.desc_]) {
        portId = outputCollectionNameIdMap[item.desc_];
      } else {
        portId = "port_" + Math.random().toString(36).substr(2, 9);
        outputCollectionNameIdMap[item.desc_] = portId;
      }

      item.id = portId;
    }
  });

  const nodeRepositories_ = businessNode.node_repository_ ?? [];
  nodeRepositories_.map((item) => {
    businessNodeNormalize(item);
  });
}
export function getParentInputLineToInnerNodes(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {

  let result: ILineEntity[] = []

  const innerNodeIds = getAllInnerNodeIds(getNodeById(node.id, clientContext)!)

  const parents = getNodeParents(getNodeById(node.id, clientContext)!)

  parents.map(parent => {

    const parentExpandInfo = getNodeExpandInfo(parent.id, clientContext)
    const parentInputLinesToInnerNodes = parentExpandInfo?.inputLines.filter(parentInputLIne => {
      return innerNodeIds.includes(parentInputLIne.to) || parentInputLIne.oldTo && innerNodeIds.includes(parentInputLIne.oldTo)
    }) ?? []

    result = [...result, ...parentInputLinesToInnerNodes]

  })

  result.map(inputLine=>{
    return {
      ...inputLine, 
    }
  })

  return result
}

export function getParentInputLineToNode(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {

  let result: ILineEntity[] = []
  
  const parents = getNodeParents(getNodeById(node.id, clientContext)!)

  parents.map(parent => {

    const parentExpandInfo = getNodeExpandInfo(parent.id, clientContext)
    const parentInputLinesToInnerNodes = parentExpandInfo?.inputLines.filter(parentInputLIne => {
      return node.id == parentInputLIne.to || node.id == parentInputLIne.oldTo 
    }) ?? []

    result = [...result, ...parentInputLinesToInnerNodes]

  })

  result.map(inputLine=>{
    return {
      ...inputLine, 
    }
  })

  return result
}


export function getParentOutputlineFromInnerNodes(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {

  let result: ILineEntity[] = []

  const innerNodeIds = getAllInnerNodeIds(getNodeById(node.id, clientContext)!)

  const parents = getNodeParents(getNodeById(node.id, clientContext)!)

  parents.map(parent => {

     const parentExpandInfo = getNodeExpandInfo(parent.id, clientContext)

    const parentOutputLinesFromInnerNodes = parentExpandInfo?.outputLines.filter(parentOutputLIne => {
      return innerNodeIds.includes(parentOutputLIne.from) || parentOutputLIne.oldFrom && innerNodeIds.includes(parentOutputLIne.oldFrom)
    }) ?? []

    result = [...result, ...parentOutputLinesFromInnerNodes]

  })

  return result
}






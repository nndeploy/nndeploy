
import {
  IPoint,
  WorkflowEdgeJSON,
} from "@flowgram.ai/free-layout-editor";
import { FlowDocumentJSON, FlowNodeJSON, FlowNodeRegistry } from "../../../../typings";
import {
  IBusinessNode,
  Inndeploy_ui_layout,
} from "../../../Layout/Design/WorkFlow/entity";
import { businessNodeIterate, businessNodeNormalize } from "./functions";
import { ILineEntity, INodeUiExtraInfo } from "../entity";


let iShrimpDelayIndex = 0

function isInCompositeNode(parents: IBusinessNode[]) {
  return parents.some((parent) => parent.is_composite_node_)
}


function getAllOutputPortNameToNodePortIdMap(businessContent: IBusinessNode, layout: Inndeploy_ui_layout['layout']) {
  const outputMap: { [key: string]: { nodeId: string; portId: string } } = {};

  function processNode(businessNode: IBusinessNode, parents: IBusinessNode[]) {


    if (businessNode.is_graph_ || businessNode.is_loop_) { // subcavas dont't make line when loaded,, relevent lines connectd to or from it's children  
      return
    }

    const isInComposite = isInCompositeNode(parents)

    if (isInComposite) {
      return
    }


    // if (businessNode.is_graph_) { 
    //   return
    // }
    if (businessNode.outputs_) {
      businessNode.outputs_?.map((output) => {
        if (output.name_) {


          if (businessNode.is_graph_) {
            outputMap[output.name_] = {
              nodeId: businessNode.id,
              portId: output.id!,
            };
          } else {
            if (outputMap[output.name_]) {
              return
            }
            outputMap[output.name_] = {
              nodeId: businessNode.id,
              portId: output.id!,
            };
          }
        }
      });
    }

  }

  // const children = businessContent.node_repository_ ?? [];
  // children.map((childNode) => {
  //   businessNodeIterate(childNode, processNode, [businessContent])
  // });
  businessNodeIterate(businessContent, processNode)

  // businessContent.node_repository_?.map((businessNode) => {
  //   processNode(businessNode, []);
  // });

  return outputMap;
}
/** 获取所有node的 输入节点的 名字->[{nodeId, portId}]映射表 */
function getAllInputPortNameToNodePortIdMap(businessContent: IBusinessNode) {
  const inputMap: { [key: string]: { nodeId: string; portId: string }[] } = {};


  function processNode(businessNode: IBusinessNode,
    parents: IBusinessNode[]
  ) {

    if (businessNode.is_graph_ || businessNode.is_loop_) {// subcavas dont't make line when loaded,, relevent lines connectd to or from it's children  
      return
    }

    const isInComposite = isInCompositeNode(parents)

    if (isInComposite) {
      return
    }

    if (businessNode.inputs_) {
      businessNode.inputs_?.map((input) => {
        if (input.name_) {


          if (inputMap[input.name_]) {
            inputMap[input.name_] = [...inputMap[input.name_], {
              nodeId: businessNode.id,
              portId: input.id!,
            }];
          } else {
            inputMap[input.name_] = [{
              nodeId: businessNode.id,
              portId: input.id!,
            }];
          }

          // if (businessNode.is_graph_) {
          //   inputMap[input.name_] = {
          //     nodeId: businessNode.id,
          //     portId: input.id!,
          //   };
          // } else {
          //   if (inputMap[input.name_]) {
          //     return
          //   }
          // }



        }
      });
    }

  }

  const children = businessContent.node_repository_ ?? [];
  children.map((childNode) => {
    businessNodeIterate(childNode, processNode, [])
  });


  return inputMap;
}

function getBussinessNodeFieldValue(businessContent: IBusinessNode, nodeId: string, fieldName: string) {

  let find = false
  let value: any
  businessNodeIterate(businessContent, (node) => {

    if (node.id == nodeId) {
      find = true
      value = node[fieldName]
      return

    }

  })

  return value
}

export function buildEdges(businessContent: IBusinessNode, layout: Inndeploy_ui_layout['layout']) {

  const outputMap = getAllOutputPortNameToNodePortIdMap(businessContent, layout);

  console.log('outputMap', outputMap)

  const inputMap = getAllInputPortNameToNodePortIdMap(businessContent);

  console.log('inputMap', inputMap)

  var edges: WorkflowEdgeJSON[] = [];

  //function isNode

  for (let connectionName in outputMap) {
    var outputInfo = outputMap[connectionName];
    var sourceInfos = inputMap[connectionName];

    if (!sourceInfos) {
      continue
    }

    sourceInfos.map(sourceInfo => {
      var connection: WorkflowEdgeJSON = {
        sourceNodeID: outputInfo.nodeId,
        targetNodeID: sourceInfo.nodeId,
        sourcePortID: outputInfo.portId,
        targetPortID: sourceInfo.portId,
      }

      //const sourceNodeName = getBussinessNodeFieldValue(businessContent, sourceInfo.nodeId, "name_")
      // if(layout?.[sourceNodeName].expanded == false){
      //   return
      // }

      edges.push(connection);
    })



  }
  return edges


}



export function transferBusinessNodeToDesignNodeIterate(
  businessNode: IBusinessNode,
  layout?: Inndeploy_ui_layout['layout'],
  parentNodes: IBusinessNode[] = []
): FlowNodeJSON {

  if (businessNode.name_ == 'Prefill_1') {
    // debugger
  }

  var type = businessNode.key_

  const needInitShrip = layout?.[businessNode.name_]?.expanded === false // layout?.[businessNode.name_]?.expanded === undefined ||
  if (needInitShrip) {
    iShrimpDelayIndex++
  }

  function getNodeUiExtraInfo(businessNode: IBusinessNode) {

    let nodeUiExtraInfo: INodeUiExtraInfo = { position: { x: 0, y: 0 }, size: { width: 200, height: 80 } }

    if (parentNodes.length > 0) {

      let current: INodeUiExtraInfo

      for (let i = 0; i < parentNodes.length; i++) {
        if (i == 0) {
          const parentNodeName = parentNodes[i].name_
          current = layout?.[parentNodeName]!
        } else {
          const parentNodeName = parentNodes[i].name_
          ///@ts-ignore
          current = current?.children?.[parentNodeName]!
        }
      }
      ///@ts-ignore
      nodeUiExtraInfo = current?.children?.[businessNode.name_] ?? nodeUiExtraInfo


    } else {
      nodeUiExtraInfo = layout?.[businessNode.name_] ?? nodeUiExtraInfo
    }
    return nodeUiExtraInfo
  }

  const nodeUiExtraInfo = getNodeUiExtraInfo(businessNode)

  function isContainerNode(businessNode: IBusinessNode) {
      return businessNode.is_graph_ || businessNode.is_loop_ || businessNode.is_composite_node_
  }

  const designNode: FlowNodeJSON = {
    id: `${businessNode.id}`,
    //type: businessNode.key_.split("::").pop() + "",
    type,

    meta: {
      //position: layout[businessNode.name_] ?? { x: 0, y: 0 },
      position: nodeUiExtraInfo.position,    //layout?.[businessNode.name_]?.position ?? { x: 0, y: 0 },
      size:   isContainerNode(businessNode) && false ? undefined : nodeUiExtraInfo.size,     //layout?.[businessNode.name_]?.size ?? { width: 200, height: 80 },
      //...nodeExtra[businessNode.name_], 
      //expandInfo: nodeExtra[businessNode.name_],
      defaultExpanded: true,
      //needInitShrip,
      needInitShrip: false,

      iShrimpDelayIndex: needInitShrip ? iShrimpDelayIndex : 0,
      //defaultExpanded: layout?.[businessNode.name_]?.expanded === undefined || layout?.[businessNode.name_]?.expanded === true
    },
    data: {
      // ...businessNode,
    },
  };

  function uniqCollectionPoint(businessNode: IBusinessNode, type: string) {
    let collectionPoints: any[] = [];

    businessNode[type].map((collectionPoint: any) => {
      const find = collectionPoints.find((item) => {
        return item.desc_ == collectionPoint.desc_;
      });

      if (!find) {
        collectionPoints = [
          ...collectionPoints,
          {
            id: collectionPoint.id,
            desc_: collectionPoint.desc_,
            name_: collectionPoint.name_,
            type_: collectionPoint.type_,
          },
        ];
      }
    });

    return collectionPoints;
  }

  const inputs_ = uniqCollectionPoint(businessNode, "inputs_");
  const outputs_ = uniqCollectionPoint(businessNode, "outputs_");

  businessNode.inputs_ = inputs_;
  businessNode.outputs_ = outputs_;

  const node_repository_ = businessNode.node_repository_ ? businessNode.node_repository_ : []
  delete businessNode["node_repository_"];

  designNode.data = { ...businessNode };

  const blocks = node_repository_.map((childNode) => {
    return transferBusinessNodeToDesignNodeIterate(childNode, layout, [...parentNodes, businessNode]);
  })

  designNode.blocks = blocks;


  return designNode;
}


export function transferBusinessContentToDesignContent(
  businessContent: IBusinessNode,
  nodeRegistries: FlowNodeRegistry[]
): FlowDocumentJSON {

  const { layout, groups = [] } = businessContent.nndeploy_ui_layout ?? {}


  /** 获取所有node的 输出节点的 name_->{nodeId, portId}映射表 */

  /** 遍历业务内容节点库 → 标准化每个节点 → 标准化端口 → 递归处理子节点 → 确保整个业务节点树的ID系统一致性 */
  function businessContentNormalize(businessContent: IBusinessNode) {

    /** 给node与其相关的port加上id */

    const nodeRepository = businessContent.node_repository_ ?? [];

    for (let i = 0; i < nodeRepository.length; i++) {
      let businessNode = nodeRepository[i];
      businessNodeNormalize(businessNode);
    }
  }

  businessContentNormalize(businessContent);

  let designData: FlowDocumentJSON = {
    nodes: [],
    edges: [],
  };




  const edges = buildEdges(businessContent, layout!)

  designData.edges = edges;


  const nodeRepository = businessContent.node_repository_ ?? [];

  for (let i = 0; i < nodeRepository.length; i++) {
    let businessNode = nodeRepository[i];
    let designNode = transferBusinessNodeToDesignNodeIterate(businessNode, layout, []);
    designData.nodes = [...designData.nodes, designNode];
  }


  let flowGroups = groups.map(group => {
    let groupName = group.name

    let flowGroup: FlowNodeJSON = {
      id: "group_" + Math.random().toString(36).substr(2, 9),
      type: "group",
      meta: {
        "position": {
          "x": 0,
          "y": 0
        }
      },
      data: {
        "name_": groupName
      },
      blocks: []

    }

    for (let i = 0; i < group.children.length; i++) {
      let childNodeName = group.children[i]

      let childNodeIndex = designData.nodes.findIndex(node => node.data.name_ == childNodeName)
      if (childNodeIndex != -1) {

        flowGroup.blocks = [...flowGroup.blocks!, ...designData.nodes.splice(childNodeIndex, 1)]
      }
    }

    return flowGroup
  })

  designData.nodes = [...designData.nodes, ...flowGroups]


  iShrimpDelayIndex = 0

  return designData;
}

// function buildSubcavasNodeExpandInfo(businessContent: IBusinessNode, layout: Inndeploy_ui_layout['layout']) {


//   let subcavasNodes: IBusinessNode[] = []
//   businessContent.node_repository_?.map(item => {
//     businessNodeIterate(item, (node) => {
//       if (node.is_graph_ === true && layout[node.name_ as string]?.expanded == false) {
//         subcavasNodes.push(node)
//       }
//     })
//   })

//   function getAllInnnerNodes(businessNode: IBusinessNode) {
//     let innerNodes: IBusinessNode[] = []
//     businessNodeIterate(businessNode, (node) => {
//       innerNodes.push(node)
//     })
//     return innerNodes
//   }

//   function getAllInnnerNodeName(businessNode: IBusinessNode) {
//     let innerNodes = getAllInnnerNodes(businessNode)
//     let innerNodeNames = innerNodes.map(node => node.name_)

//     return innerNodeNames

//   }





//   subcavasNodes.map(subcavasNode => {


//     let innerNodeNames = getAllInnnerNodeName(subcavasNode)



//     subcavasNode.node_repository_?.map(child => {
//       child.inputs_?.map(input => {

//         const nameParts = input.name_.split('@')
//         const topName = nameParts[0]
//         if (innerNodeNames.includes(topName)) {
//           return
//         }

//         let inputLine : ILineEntity = {
//           from:
//         }




//       })
//     })
//   })

// }

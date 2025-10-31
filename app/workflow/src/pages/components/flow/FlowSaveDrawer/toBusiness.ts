import {
  FlowNodeRenderData,
  FreeLayoutPluginContext,
  WorkflowEdgeJSON,
  WorkflowNodeJSON,
} from "@flowgram.ai/free-layout-editor";
import { FlowDocumentJSON, FlowNodeJSON } from "../../../../typings";
import lodash from 'lodash'
import {
  IBusinessNode,
} from "../../../Layout/Design/WorkFlow/entity";
import { IExpandInfo, ILineEntity, INodeUiExtraInfo } from "../entity";
import { getNodeById, getNodeNamFieldValue, isCompositeNode, isGraphNode, isNodeExpanded, isNodeOffspringOfCompositeNode } from "../functions";
import { getOriginFrom, getSubcavasInputLines, getSubcavasOutputLines } from "../form-header/function";
import { getParentInputLineToInnerNodes, getParentInputLineToNode, getParentOutputlineFromInnerNodes } from "./functions";

function designNodeIterate(
  node: WorkflowNodeJSON,
  process: (node: WorkflowNodeJSON, parrentNodes: WorkflowNodeJSON[]) => void,
  parentNodes: WorkflowNodeJSON[] = []
) {
  process(node, parentNodes);
  if (node.blocks && node.blocks.length > 0) {
    node.blocks.forEach((block) => {
      designNodeIterate(block, process, [...parentNodes, node]);
    });
  }
}

function getNodeExpandInfo(nodeId: string, clientContext: FreeLayoutPluginContext) {
  let node = clientContext.document.getNode(nodeId)
  let expandInfo: IExpandInfo = node?.getNodeMeta().expandInfo
  return expandInfo
}

export function getPortNameByPortId(portId: string, nodes: FlowNodeJSON[]) {
  let portName = ''
  nodes.map(node => {
    designNodeIterate(node, (node, parentNodes) => {
      const find = node.data.outputs_?.find(((port: any) => port.id === portId))
      if (find) {
        portName = [...parentNodes.map(item => item.data.name_), node.data.name_, find.desc_].join('@')
      }
    })
  })
  return portName

}



export function getAllEdges(designData: FlowDocumentJSON, clientContext: FreeLayoutPluginContext) {
  let edges = designData.edges;

  designData.nodes.map((node) => {
    designNodeIterate(node, (node) => {
      if (node.edges && node.edges.length > 0)
        edges = [...edges, ...node.edges];
    });
  });

  function edgesUnify(edges: WorkflowEdgeJSON[]) {
    let results: WorkflowEdgeJSON[] = [];

    edges.map((edge) => {
      var find = false;
      results.map((result) => {
        if (
          result.sourceNodeID == edge.sourceNodeID &&
          result.targetNodeID == edge.targetNodeID &&
          result.sourcePortID == edge.sourcePortID &&
          result.targetPortID == edge.targetPortID
        ) {
          find = true;
        }
      });
      if (!find) {
        results = [...results, edge];
      }
    });
    return results

  }

  let results = edgesUnify(edges)


  // function buildSubCavasEdges(lines: WorkflowEdgeJSON[]) {
  //   const subcavasEdges: WorkflowEdgeJSON[] = []


  //   lines.map(line => {

  //     let isSourceNodeContainer = isGraphNode(line.sourceNodeID, clientContext)

  //     if (isSourceNodeContainer) {

  //       let sourceNodeExpandInfo: IExpandInfo = getNodeExpandInfo(line.sourceNodeID, clientContext)

  //       sourceNodeExpandInfo?.outputLines?.map(outputLine => {
  //         let subcavasEdge: WorkflowEdgeJSON = {

  //           sourceNodeID: outputLine.oldFrom!,

  //           sourcePortID: outputLine.oldFromPort,

  //           targetNodeID: outputLine.to,
  //           targetPortID: outputLine.toPort,
  //         }
  //         const isToNodeContainer = isGraphNode(outputLine.to, clientContext)
  //         if (isToNodeContainer) {
  //           // const toNodeExpandInfo = getNodeExpandInfo(subcavasEdge.targetNodeID)
  //           // const find = toNodeExpandInfo.inputLines.find(inputLine => {
  //           //   return inputLine.from == subcavasEdge.sourceNodeID && inputLine.fromPort == subcavasEdge.sourcePortID
  //           // })
  //           // if (find) {
  //           //   subcavasEdge.targetNodeID = find.to
  //           //   subcavasEdge.targetPortID = find.oldToPort
  //           // }
  //           subcavasEdge.targetNodeID = outputLine.oldTo!
  //           subcavasEdge.targetPortID = outputLine.oldToPort

  //         }

  //         const find = subcavasEdges.find(item => {
  //           return item.sourceNodeID == subcavasEdge.sourceNodeID && item.targetNodeID == subcavasEdge.targetNodeID && item.sourcePortID == subcavasEdge.sourcePortID && item.targetPortID == subcavasEdge.targetPortID
  //         })

  //         if (!find) {
  //           subcavasEdges.push(subcavasEdge)
  //         }

  //       })
  //     }

  //     let isTargetNodeContainer = isGraphNode(line.targetNodeID, clientContext)

  //     if (isTargetNodeContainer) {


  //       let targetNodeExpandInfo: IExpandInfo = getNodeExpandInfo(line.targetNodeID, clientContext)

  //       targetNodeExpandInfo.inputLines?.map(inputLine => {
  //         let subcavasEdge: WorkflowEdgeJSON = {

  //           sourceNodeID: inputLine.from,

  //           sourcePortID: inputLine.fromPort,

  //           targetNodeID: inputLine.oldTo!,
  //           targetPortID: inputLine.oldToPort,
  //         }

  //         const isFromNodeContainer = isGraphNode(inputLine.from, clientContext)
  //         if (isFromNodeContainer) {
  //           // const fromNodeExpandInfo = getNodeExpandInfo(subcavasEdge.sourceNodeID)
  //           // const find = fromNodeExpandInfo.outputLines.find(outputLine => {
  //           //   return outputLine.to == subcavasEdge.sourceNodeID && outputLine.fromPort == subcavasEdge.sourcePortID
  //           // })
  //           // if (find) {
  //           //   subcavasEdge.sourceNodeID = find.from
  //           //   subcavasEdge.targetPortID = find.oldFromPort
  //           // }
  //           let lineTemp = line
  //           if (!inputLine.oldFrom) {
  //             let j = 0
  //           }
  //           subcavasEdge.sourceNodeID = inputLine.oldFrom!
  //           subcavasEdge.sourcePortID = inputLine.oldFromPort

  //         }


  //         const find = subcavasEdges.find(item => {
  //           return item.sourceNodeID == subcavasEdge.sourceNodeID && item.targetNodeID == subcavasEdge.targetNodeID && item.sourcePortID == subcavasEdge.sourcePortID && item.targetPortID == subcavasEdge.targetPortID
  //         })

  //         if (!find) {
  //           subcavasEdges.push(subcavasEdge)
  //         }

  //       })
  //     }
  //   })

  //   return subcavasEdges

  // }

  // const subcavasEdges = buildSubCavasEdges(results)

  // function substractDynamicContainerLines(lines: WorkflowEdgeJSON[]) {
  //   const result = lines.filter(item => {


  //     //let sourceNode = clientContext.document.getNode(item.sourceNodeID)
  //     // let sourceForm = sourceNode?.form
  //     // let isSourceNodeContainer =  sourceForm?.getValueIn('is_graph') ?? false
  //     let isSourceDynamicNode = isGraphNode(item.sourceNodeID, clientContext)

  //     //let targetNode = clientContext.document.getNode(item.targetNodeID)
  //     // let targetForm = targetNode?.form
  //     // let isTargetNodeContainer = targetForm?.getValueIn('is_graph') ?? false
  //     let isTargetDynamicNode = isGraphNode(item.targetNodeID, clientContext)


  //     return (isTargetDynamicNode || isSourceDynamicNode) ? false : true

  //   })
  //   return result
  // }

  // results = substractDynamicContainerLines(results)

  return [...results,
    //...subcavasEdges
  ];
}
/** 获取所有线条的 映射表:  
 *   name: edge.sourceNodeID + "@" + edge.sourcePortID -> value: source_node_name + "@" + source_port.desc_ 
 *   name: edge.edge.targetNodeID + "@" + edge.targetPortID -> value: source_node_name + "@" + source_port.desc_
 * 
 * */
export function getEdgeToNameMaps(allNodes: FlowNodeJSON[], allEdges: WorkflowEdgeJSON[], clientContext: FreeLayoutPluginContext) {
  function getEdgeNameByIterate(nodeId: string, portId: string, node: WorkflowNodeJSON, parentNodeNames: string[]): string {
    if (node.id === nodeId) {
      let node_name = node.data.name_ as string;
      for (let edge of node.data.outputs_ ?? []) {
        if (edge.id === portId) {

          let parts = [...parentNodeNames, node_name, edge.desc_]

          let edge_name = parts.join('@');

          //return "fuck you"
          return edge_name;
        }
      }
      return "";
    }
    if (node.blocks && node.blocks.length > 0) {
      for (let childNode of node.blocks) {
        let result = getEdgeNameByIterate(nodeId, portId, childNode, [...parentNodeNames, node.data.name_]);
        if (result) {
          return result;
        }
      }
    }

    return "";
  }

  function getEdgeNameById(nodeId: string, portId: string, allNodes: FlowNodeJSON[]) {
    try {
      for (let i = 0; i < allNodes.length; i++) {
        let node = allNodes[i];
        let result = getEdgeNameByIterate(nodeId, portId, node, []);
        if (result) {
          return result;
        }
      }
    } catch (e) {
      console.log("getEdgeNameById", e);
    }

    return "";
  }

  let edge_map: { [key: string]: string } = {};
  allEdges.map((edge) => {

    const nodeName = getNodeNamFieldValue(edge.targetNodeID, 'name_', clientContext)
    if (nodeName == 'decode_sampler') {
      let k = 0
    }

    const fromNodeName = getNodeNamFieldValue(edge.sourceNodeID, 'name_', clientContext)
    // if(fromNodeName == 'decode_sampler'){
    //   let k = 0
    // }

    const origin = getOriginFrom({
      ...edge, from: edge.sourceNodeID, fromPort: edge.sourcePortID,
      //from_name: edge.sourceNodeID,
      to: edge.targetNodeID,
      toPort: edge.targetPortID,
      //to_name: edge.targetNodeID,
    }, clientContext)



    var name = getEdgeNameById(origin.originFrom as string, origin.originFromPort as string, allNodes);
    //let name = getPortNameByPortId(origin.originFromPort as string, allNodes)
    if (!name) {
      let j = 0;
    }
    edge_map[edge.sourceNodeID + "@" + edge.sourcePortID] = name
    edge_map[edge.targetNodeID + "@" + edge.targetPortID] = name;
  });

  return edge_map
}

export function designDataToBusinessData(designData: FlowDocumentJSON, graphTopNode: IBusinessNode, allNodes: FlowNodeJSON[], clientContext: FreeLayoutPluginContext) {

  let allEdges = getAllEdges(designData, clientContext);


  function buildNodesLayout() {

    let nodesExtraInfo: { [nodeName: string]: INodeUiExtraInfo } = {}

    function processNode(node: WorkflowNodeJSON, parrentNodes: WorkflowNodeJSON[]) {

      let tempNode = clientContext?.document?.getNode(node.id as string)
      let expanded = tempNode?.getData(FlowNodeRenderData).expanded

      let nodeMeta = clientContext?.document?.getNode(node.id as string)?.getNodeMeta();
      //const expandInfo: IExpandInfo = nodeMeta?.expandInfo

      const nodeExpandInfo: INodeUiExtraInfo = {
        position: node.meta?.position || { x: 0, y: 0 },
        size: nodeMeta?.size || { width: 200, height: 60 },
        expanded: expanded,

      }


      let current: INodeUiExtraInfo

      if (parrentNodes.length < 1) {
        nodesExtraInfo[node.data.name_] = nodeExpandInfo
      } else {
        for (let i = 0; i < parrentNodes.length; i++) {

          const parentNodeName: string = parrentNodes[i].data.name_;

          if (i == 0) {
            current = nodesExtraInfo[parentNodeName]
          } else {
            ///@ts-ignore
            current = current?.children?.[parentNodeName]

          }
          current.children = current.children || {}
          current.children[node.data.name_] = nodeExpandInfo
        }

      }


    }
    designData.nodes.map(node => {

      designNodeIterate(node, processNode)

    })

    return nodesExtraInfo
  }

  const nodesLayout = buildNodesLayout()

  function getGroupInfos() {
    let groups = designData.nodes.filter(node => node.type == 'group')
    let groupInfos = groups.map(group => {
      let groupName = group.data.name_
      let childrenNodes: string[] = group.blocks?.map(blockNode => {
        return blockNode.data.name_
      }) ?? []

      return { name: groupName, children: childrenNodes }

    })

    return groupInfos


  }

  const groups = getGroupInfos()

  // function buildGraphNodeExpandInfo() {


  //   function getAllContainerNodes(designData: FlowDocumentJSON) {

  //     let containerNodes: string[] = []

  //     for (let i = 0; i < designData.nodes.length; i++) {
  //       const node = designData.nodes[i]

  //       designNodeIterate(node, (node) => {
  //         if (node.data.is_graph_) {
  //           containerNodes.push(node.id)
  //         }
  //       })
  //     }

  //     return containerNodes
  //   }


  //   const containerNodes = getAllContainerNodes(designData)


  //   let newExpandInfo: [containerNodeName: string]=>IExpandInfo[] = {}

  //    containerNodes.forEach(containerNode => {
  //     let expandInfo: IExpandInfo = getNodeExpandInfo(containerNode, clientContext)

  //     const inputLines = expandInfo.inputLines?.map(inputLine => {

  //       let newLine :  ILineEntity  = {
  //         ...inputLine, 

  //       }
  //       const ports =  ['oldFromPort', 'fromPort', 'oldToPort', 'toPort']

  //       ports.map( portField=>{
  //         if(inputLine[portField as keyof ILineEntity]) {
  //           newLine[portField as keyof ILineEntity] = getPortNameByPortId(inputLine[portField as keyof ILineEntity] as string)
  //         }
  //       })

  //      return newLine


  //     })

  //     const outputLines = expandInfo.outputLines?.map(outputLine => {

  //       let newLine :  ILineEntity  = {
  //         ...outputLine, 

  //       }
  //       const ports =  ['oldFromPort', 'fromPort', 'oldToPort', 'toPort']

  //       ports.map( portField=>{
  //         if(outputLine[portField as keyof ILineEntity]) {
  //           newLine[portField as keyof ILineEntity] = getPortNameByPortId(outputLine[portField as keyof ILineEntity] as string)
  //         }
  //       })

  //      return newLine


  //     })

  //     return {
  //       inputLines,
  //       outputLines
  //     }

  //   })

  // }

  let businessData: IBusinessNode = {
    // key_: "nndeploy::dag::Graph",
    // name_: "demo",
    // device_type_: "kDeviceTypeCodeCpu:0",
    // inputs_: [],
    // outputs_: [],
    // is_external_stream_: false,
    // is_inner_: false,
    // is_time_profile_: false,
    // is_debug_: false,
    // is_graph_node_share_stream_: true,
    // queue_max_size_: 16,
    // node_repository_: [],
    ...graphTopNode,
    node_repository_: [],
    nndeploy_ui_layout: {
      layout: nodesLayout,
      groups
    }
  };

  const edgeToNameMap = getEdgeToNameMaps(allNodes, allEdges, clientContext)

  function transferDesignNodeToBusinessNode(
    node: WorkflowNodeJSON
  ): IBusinessNode {

    function changeNodeDescendConnecitonName(node: WorkflowNodeJSON, originName: string, name_: string) {
      if (originName == name_) {
        return
      }

      ///@ts-ignore
      //const respositories: any[] = node?.data.node_repository_ ?? []
      const childrenblocks: WorkflowNodeJSON[] = node?.blocks ?? []

      childrenblocks.map(node => {

        designNodeIterate(node, function (node: WorkflowNodeJSON) {
          ///@ts-ignore
          const inputs = node.inputs_ ?? []
          inputs.map((item: any) => {
            if (item.name_ == originName) {
              item.name_ = name_
            }
          })

          ///@ts-ignore
          const outputs_ = node.outputs_ ?? []
          outputs_.map((item: any) => {
            if (item.name_ == originName) {
              item.name_ = name_
            }
          })

        })
      })
    }


    let inputArray_: any[] = []

    let inputs_: any[] = []

    let outputArray_: any[] = []
    let outputs_: any[] = []

    const flowNode = getNodeById(node.id, clientContext)!
    const parentInputLineToInnerNodes = getParentInputLineToInnerNodes(flowNode, clientContext)

    if (isNodeOffspringOfCompositeNode(flowNode, clientContext)) {
      inputs_ = node.data.inputs_ ?? []
      outputs_ = node.data.outputs_ ?? []
    } else {

      if (isGraphNode(node.id, clientContext)) {


        let inputLines: ILineEntity[] = []

        if (isNodeExpanded(node.id, clientContext)) {

          inputLines = getSubcavasInputLines(flowNode, clientContext)

          inputLines = [...inputLines, ...parentInputLineToInnerNodes]

        } else {

          const expandInfo = getNodeExpandInfo(node.id, clientContext)

          inputLines = expandInfo.inputLines ?? []

          inputLines = [...inputLines, ...parentInputLineToInnerNodes]

        }

        inputArray_ = inputLines.map(item => {


          const name_ = getPortNameByPortId(item.originFromPort as string, designData.nodes)

          return {
            type_: item.type_,
            name_: name_,
            desc_: item.desc_
          }

        })

        inputArray_ = lodash.uniqBy(inputArray_, 'name_')
      } else {
        inputArray_ = (node.data.inputs_ ?? []).map((input: any) => {

          const nodeName = getNodeNamFieldValue(node.id, 'name_', clientContext)

          // const originName = input.name_
          let name_ = edgeToNameMap[node.id + "@" + input.id]

          if (!name_) {

            const parentInputLines = getParentInputLineToNode(flowNode, clientContext)
            if (parentInputLines.length > 0) {
              const originFromPort = parentInputLines[0].originFromPort
              name_ = getPortNameByPortId(originFromPort as string, designData.nodes)
            }

          }
          return {
            ...input,
            name_,
          };
        });

        inputArray_ = lodash.uniqBy(inputArray_, 'name_')
      }

      inputs_ = inputArray_.flat();



      if (isGraphNode(node.id, clientContext)) {

        let outputLines: ILineEntity[] = []
        const flowNode = getNodeById(node.id, clientContext)!
        const parentOutputLineFromInnerNodes = getParentOutputlineFromInnerNodes(flowNode, clientContext)


        if (isNodeExpanded(node.id, clientContext)) {

          outputLines = getSubcavasOutputLines(getNodeById(node.id, clientContext)!, clientContext)
          outputLines = [...outputLines, ...parentOutputLineFromInnerNodes]

        } else {

          const expandInfo = getNodeExpandInfo(node.id, clientContext)
          outputLines = expandInfo.outputLines ?? []
          outputLines = [...outputLines, ...parentOutputLineFromInnerNodes]

        }

        outputArray_ = outputLines.map(outputLine => {


          let name_ = getPortNameByPortId(outputLine.originFromPort as string, designData.nodes)

          return {
            type_: outputLine.type_,
            name_: name_,
            desc_: outputLine.desc_
          }
        })

        outputArray_ = lodash.uniqBy(outputArray_, 'name_')

      } else {

        outputArray_ = (node.data.outputs_ ?? []).map((output: any) => {


          let name_ = getPortNameByPortId(output.id as string, designData.nodes)

          return {
            ...output,
            name_,
          };

        })
      }
      outputs_ = outputArray_.flat();
    }

   

    let node_repository_: IBusinessNode[] = [];

    if (node.blocks && node.blocks.length > 0) {
      node_repository_ = node.blocks.map((childNode) => {
        return transferDesignNodeToBusinessNode(childNode);
      });
    }

    let businessNode: IBusinessNode = {
      ...node.data,
      inputs_,

      outputs_,
      node_repository_,
    };

    function normalize() {
      delete businessNode.id
      inputs_.map((collectionPoint: any) => {
        delete collectionPoint.id
      })

      outputs_.map((collectionPoint: any) => {
        delete collectionPoint.id
      })
    }
    normalize()

    return businessNode;
  }

  let node_repository_: IBusinessNode[] = []

  designData.nodes.map((node) => {
    if (node.type == 'group') {

      for (let i = 0; i < node.blocks!.length; i++) {
        let blockNode = node.blocks![i]
        let nodeItem = transferDesignNodeToBusinessNode(blockNode)
        node_repository_.push(nodeItem)
      }

    } else {
      let nodeItem = transferDesignNodeToBusinessNode(node);
      node_repository_.push(nodeItem)
    }

  });

  businessData.node_repository_ = node_repository_;

  return businessData;
}



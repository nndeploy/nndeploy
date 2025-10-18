import {
  FlowNodeRenderData,
  FreeLayoutPluginContext,
  WorkflowEdgeJSON,
  WorkflowNodeJSON,
} from "@flowgram.ai/free-layout-editor";
import { FlowDocumentJSON, FlowNodeJSON, FlowNodeRegistry } from "../../../../typings";
import {
  IBusinessNode,
  Inndeploy_ui_layout,
  IWorkFlowEntity,
} from "../../../Layout/Design/WorkFlow/entity";
import { resourceLimits } from "worker_threads";
import { random } from "lodash";
import { IExpandInfo, INodeUiExtraInfo } from "../entity";

// export function buildBusinessDataFromDesignData(designData: FlowDocumentJSON) {
//   let businessData: IBusinessNode = {
//     key_: "nndeploy::dag::Graph",
//     name_: "demo",
//     device_type_: "kDeviceTypeCodeX86:0",
//     inputs_: [],
//     outputs_: [
//       {
//         name_: "detect_out",
//         type_: "kNotSet",
//       },
//     ],
//     is_external_stream_: false,
//     is_inner_: false,
//     "is_time_profile_": true,
//     is_debug_: false,
//     is_graph_node_share_stream_: true,
//     queue_max_size_: 16,
//     node_repository_: [],
//   };

//   function getNodeNameByIterate(id: string, node: WorkflowNodeJSON): string {
//     if (node.id === id) {
//       return node.data.name_ as string;
//     }
//     if (node.blocks && node.blocks.length > 0) {
//       for (let childNode of node.blocks) {
//         let result = getNodeNameByIterate(id, childNode);
//         if (result) {
//           return result;
//         }
//       }
//     }

//     return "";
//   }

//   function getNodeNameById(id: string) {
//     try {
//       for (let i = 0; i < designData.nodes.length; i++) {
//         let node = designData.nodes[i];
//         let result = getNodeNameByIterate(id, node);
//         if (result) {
//           return result;
//         }
//       }
//     } catch (e) {
//       console.log("getNodeNameById", e);
//     }

//     return "";
//   }

//   function transferDesignNodeToBusinessNode(
//     node: WorkflowNodeJSON
//   ): IBusinessNode {
//     var inputs_ = designData.edges
//       .filter((edge) => {
//         return edge.targetNodeID === node.id;
//       })
//       .map((item, index) => {
//         var sourceNodeName = getNodeNameById(item.sourceNodeID as string);
//         var targetNodeName = getNodeNameById(item.targetNodeID as string);

//         if(!node.data.inputs_[index]){
//           debugger;
//         }
//         var temp1 = document.querySelectorAll(`[data-port-id=${item.sourcePortID}]`);
//         var temp2 = temp1.item(0)
//         let sourcePortDesc = document.querySelectorAll(`[data-port-id='${item.sourcePortID}']`).item(0).getAttribute('data-port-desc')
//         let descPortDesc = document.querySelectorAll(`[data-port-id='${item.targetPortID}']`).item(0).getAttribute('data-port-desc')

//         let find = node.data.inputs_.find( input=>{
//           return input.id == item.targetPortID
//         })

//         return {
//           name: [
//             sourceNodeName,
//             targetNodeName,
//             // item.sourcePortID,
//             // item.targetPortID,
//             sourcePortDesc,
//             descPortDesc

//           ].join("!"),
//           type_:find.type_,
//         };
//       });

//     var outputs_ = designData.edges
//       .filter((edge) => {
//         return edge.sourceNodeID === node.id;
//       })
//       .map((item, index) => {
//         var sourceNodeName = getNodeNameById(item.sourceNodeID as string);
//         var targetNodeName = getNodeNameById(item.targetNodeID as string);

//         if(!node.data.outputs_[index]){
//           debugger;
//         }

//         let find = node.data.outputs_.find( output=>{
//           return output.id == item.sourcePortID
//         })

//         var temp3 = document.querySelectorAll(`[data-port-id=${item.sourcePortID}]`);
//         var temp4 = temp3.item(0)

//           let sourcePortDesc = document.querySelectorAll(`[data-port-id='${item.sourcePortID}']`).item(0).getAttribute('data-port-desc')
//         let descPortDesc = document.querySelectorAll(`[data-port-id='${item.targetPortID}']`).item(0).getAttribute('data-port-desc')

//         return {
//           name: [
//             sourceNodeName,
//             targetNodeName,
//             // item.sourcePortID,
//             // item.targetPortID,
//              sourcePortDesc,
//             descPortDesc
//           ].join("!"),
//           type_: find.type_,
//         };
//       });

//     let node_repository_: IBusinessNode[] = [];

//     if (node.blocks && node.blocks.length > 0) {
//       node_repository_ = node.blocks.map((childNode) => {
//         return transferDesignNodeToBusinessNode(childNode);
//       });
//     }

//     let businessNode: IBusinessNode = {
//       ...node.data,
//       inputs_,

//       outputs_,
//       node_repository_,
//     };
//     return businessNode;
//   }

//   let node_repository_: IBusinessNode[] = designData.nodes.map((node) => {
//     return transferDesignNodeToBusinessNode(node);
//   });

//   businessData.node_repository_ = node_repository_;

//   return businessData;
// }


function designNodeIterate(
  node: WorkflowNodeJSON,
  process: (node: WorkflowNodeJSON) => void
) {
  process(node);
  if (node.blocks && node.blocks.length > 0) {
    node.blocks.forEach((block) => {
      designNodeIterate(block, process);
    });
  }
}

function businessNodeIterate(
  node: IBusinessNode,
  process: (node: IBusinessNode) => void
) {
  process(node);
  if (node.node_repository_ && node.node_repository_.length > 0) {
    node.node_repository_.forEach((respoitory) => {
      businessNodeIterate(respoitory, process);
    });
  }
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


  function buildSubCavasEdges(lines: WorkflowEdgeJSON[]) {
    const subcavasEdges: WorkflowEdgeJSON[] = []


    function isContainerNode(nodeId: string) {
      let node = clientContext.document.getNode(nodeId)
      let form = node?.form
      let isContainer = form?.getValueIn('is_graph') ?? false
      return isContainer
    }
    function getNodeExpandInfo(nodeId: string) {
      let node = clientContext.document.getNode(nodeId)
      let expandInfo: IExpandInfo = node?.getNodeMeta().expandInfo
      return expandInfo
    }

    lines.map(line => {

      let isSourceNodeContainer = isContainerNode(line.sourceNodeID)

      if (isSourceNodeContainer) {

        let sourceNodeExpandInfo: IExpandInfo = getNodeExpandInfo(line.sourceNodeID)

        sourceNodeExpandInfo?.outputLines?.map(outputLine => {
          let subcavasEdge: WorkflowEdgeJSON = {

            sourceNodeID: outputLine.oldFrom!,

            sourcePortID: outputLine.oldFromPort,

            targetNodeID: outputLine.to,
            targetPortID: outputLine.toPort,
          }
          const isToNodeContainer = isContainerNode(outputLine.to)
          if (isToNodeContainer) {
            // const toNodeExpandInfo = getNodeExpandInfo(subcavasEdge.targetNodeID)
            // const find = toNodeExpandInfo.inputLines.find(inputLine => {
            //   return inputLine.from == subcavasEdge.sourceNodeID && inputLine.fromPort == subcavasEdge.sourcePortID
            // })
            // if (find) {
            //   subcavasEdge.targetNodeID = find.to
            //   subcavasEdge.targetPortID = find.oldToPort
            // }
            subcavasEdge.targetNodeID = outputLine.oldTo!
            subcavasEdge.targetPortID = outputLine.oldToPort

          }

          const find = subcavasEdges.find(item => {
            return item.sourceNodeID == subcavasEdge.sourceNodeID && item.targetNodeID == subcavasEdge.targetNodeID && item.sourcePortID == subcavasEdge.sourcePortID && item.targetPortID == subcavasEdge.targetPortID
          })

          if (!find) {
            subcavasEdges.push(subcavasEdge)
          }

        })
      }

      let isTargetNodeContainer = isContainerNode(line.targetNodeID)

      if (isTargetNodeContainer) {


        let targetNodeExpandInfo: IExpandInfo = getNodeExpandInfo(line.targetNodeID)

        targetNodeExpandInfo.inputLines?.map(inputLine => {
          let subcavasEdge: WorkflowEdgeJSON = {

            sourceNodeID: inputLine.from,

            sourcePortID: inputLine.fromPort,

            targetNodeID: inputLine.oldTo!,
            targetPortID: inputLine.oldToPort,
          }

          const isFromNodeContainer = isContainerNode(inputLine.from)
          if (isFromNodeContainer) {
            // const fromNodeExpandInfo = getNodeExpandInfo(subcavasEdge.sourceNodeID)
            // const find = fromNodeExpandInfo.outputLines.find(outputLine => {
            //   return outputLine.to == subcavasEdge.sourceNodeID && outputLine.fromPort == subcavasEdge.sourcePortID
            // })
            // if (find) {
            //   subcavasEdge.sourceNodeID = find.from
            //   subcavasEdge.targetPortID = find.oldFromPort
            // }
            subcavasEdge.sourceNodeID = inputLine.oldFrom!
            subcavasEdge.sourcePortID = inputLine.oldFromPort

          }


          const find = subcavasEdges.find(item => {
            return item.sourceNodeID == subcavasEdge.sourceNodeID && item.targetNodeID == subcavasEdge.targetNodeID && item.sourcePortID == subcavasEdge.sourcePortID && item.targetPortID == subcavasEdge.targetPortID
          })

          if (!find) {
            subcavasEdges.push(subcavasEdge)
          }

        })
      }
    })

    return subcavasEdges

  }

  const subcavasEdges = buildSubCavasEdges(results)

  function substractContainerLines(lines: WorkflowEdgeJSON[]) {
    const result = lines.filter(item => {

      // inputLines
      let targetNode = clientContext.document.getNode(item.targetNodeID)
      let targetForm = targetNode?.form
      let isTargetNodeContainer = targetForm?.getValueIn('is_graph') ?? false

      // inputLines
      let sourceNode = clientContext.document.getNode(item.sourceNodeID)
      let sourceForm = sourceNode?.form
      let isSourceNodeContainer = sourceForm?.getValueIn('is_graph') ?? false

      return !isTargetNodeContainer && !isSourceNodeContainer

    })
    return result
  }

  results = substractContainerLines(results)

  return [...results, ...subcavasEdges];
}
/** 获取所有线条的 映射表:  
 *   name: edge.sourceNodeID + "@" + edge.sourcePortID -> value: source_node_name + "@" + source_port.desc_ 
 *   name: edge.edge.targetNodeID + "@" + edge.targetPortID -> value: source_node_name + "@" + source_port.desc_
 * 
 * */
export function getEdgeMaps(allNodes: FlowNodeJSON[], allEdges: WorkflowEdgeJSON[]) {
  function getEdgeNameByIterate(nodeId: string, portId: string, node: WorkflowNodeJSON): string {
    if (node.id === nodeId) {
      let node_name = node.data.name_ as string;
      for (let edge of node.data.outputs_ ?? []) {
        if (edge.id === portId) {
          let edge_name = node_name + "@" + edge.desc_;
          return edge_name;
        }
      }
      return "";
    }
    if (node.blocks && node.blocks.length > 0) {
      for (let childNode of node.blocks) {
        let result = getEdgeNameByIterate(nodeId, portId, childNode);
        if (result) {
          return result;
        }
      }
    }

    return "";
  }

  function getEdgeNameById(nodeId: string, portId: string) {
    try {
      for (let i = 0; i < allNodes.length; i++) {
        let node = allNodes[i];
        let result = getEdgeNameByIterate(nodeId, portId, node);
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

    var name = getEdgeNameById(edge.sourceNodeID as string, edge.sourcePortID as string);
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


  function getNodesLayout() {

    let nodesExtraInfo: { [nodeName: string]: INodeUiExtraInfo } = {}

    function processNode(node: WorkflowNodeJSON) {

      let tempNode = clientContext?.document?.getNode(node.id as string)
      let expanded = tempNode?.getData(FlowNodeRenderData).expanded

      let nodeMeta = clientContext?.document?.getNode(node.id as string)?.getNodeMeta();
      //const expandInfo: IExpandInfo = nodeMeta?.expandInfo

      nodesExtraInfo[node.data.name_] = {
        position: node.meta?.position || { x: 0, y: 0 },
        size: nodeMeta?.size || { width: 180, height: 48 },
        expanded: expanded,

        // inputLines: expandInfo?.inputLines || [],
        // outputLines: expandInfo?.outputLines || []

      }
    }
    designData.nodes.map(node => {

      designNodeIterate(node, processNode)

    })


    return nodesExtraInfo
  }

  const nodesLayout = getNodesLayout()

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

  function getNodeNameByIterate(id: string, node: WorkflowNodeJSON): string {
    if (node.id === id) {
      return node.data.name_ as string;
    }
    if (node.blocks && node.blocks.length > 0) {
      for (let childNode of node.blocks) {
        let result = getNodeNameByIterate(id, childNode);
        if (result) {
          return result;
        }
      }
    }

    return "";
  }

  function getNodeNameById(id: string) {
    try {
      for (let i = 0; i < designData.nodes.length; i++) {
        let node = designData.nodes[i];
        let result = getNodeNameByIterate(id, node);
        if (result) {
          return result;
        }
      }
    } catch (e) {
      console.log("getNodeNameById", e);
    }

    return "";
  }

  const edge_map = getEdgeMaps(allNodes, allEdges)



  function transferDesignNodeToBusinessNode(
    node: WorkflowNodeJSON
  ): IBusinessNode {


    // function nodeIterate(
    //   node: any,
    //   process: (node: any) => void,
    // ) {
    //   process(node);
    //   if (node?.blocks && node?.blocks.length > 0) {
    //     node.blocks.forEach((itemNode: WorkflowNodeJSON) => {
    //       nodeIterate(itemNode, process);
    //     });
    //   }
    // }

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

    const inputArray_ = (node.data.inputs_ ?? []).map((input: any) => {

      const originName = input.name_
      const name_ = edge_map[node.id + "@" + input.id]

      if (!name_) {
        //debugger
      }

      changeNodeDescendConnecitonName(node, originName, name_)
      return {
        ...input,
        name_,
      };
    });

    const inputs_ = inputArray_.flat();

    const outputArray_ = (node.data.outputs_ ?? []).map((output: any) => {


      const originName = output.name_
      const name_ = edge_map[node.id + "@" + output.id]
      changeNodeDescendConnecitonName(node, originName, name_)

      return {
        ...output,
        name_,
      };


    })

    const outputs_ = outputArray_.flat();

    let node_repository_: IBusinessNode[] = [];

    if (node.blocks && node.blocks.length > 0) {
      node_repository_ = node.blocks.map((childNode) => {
        return transferDesignNodeToBusinessNode(childNode);
      });
    }

    // if (node.data.node_repository_) {
    //   node_repository_ = node.data.node_repository_
    // }

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

export function transferBusinessNodeToDesignNodeIterate(
  businessNode: IBusinessNode,
  layout?: Inndeploy_ui_layout['layout']
): FlowNodeJSON {

  if (businessNode.name_ == 'Prefill_1') {
    // debugger
  }



  // var type = businessNode.is_graph_ ? 'group':  businessNode.key_

  var type = businessNode.key_

  const designNode: FlowNodeJSON = {
    id: `${businessNode.id}`,
    //type: businessNode.key_.split("::").pop() + "",
    type,

    meta: {
      //position: layout[businessNode.name_] ?? { x: 0, y: 0 },
      position: layout?.[businessNode.name_]?.position ?? { x: 0, y: 0 },
      size: layout?.[businessNode.name_]?.size ?? { width: 200, height: 80 },
      //...nodeExtra[businessNode.name_], 
      //expandInfo: nodeExtra[businessNode.name_],
      defaultExpanded: layout?.[businessNode.name_]?.expanded === undefined || layout?.[businessNode.name_]?.expanded === true
    },
    data: {
      // ...businessNode,
    },
  };

  // if (
  //   businessNode.node_repository_ &&
  //   businessNode.node_repository_.length > 0
  // ) {
  //   const nodeRepositories = businessNode.node_repository_;
  //   delete businessNode["node_repository_"];
  //   const blocks = nodeRepositories.map((childNode) => {
  //     return transferBusinessNodeToDesignNode(childNode);
  //   });

  //   designNode.blocks = blocks;
  // }

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

  designNode.data = { ...businessNode };

  if (businessNode.node_repository_ && businessNode.node_repository_.length > 0) {
    const nodeRepositories = businessNode.node_repository_;
    delete businessNode["node_repository_"];
    const blocks = nodeRepositories.map((childNode) => {
      return transferBusinessNodeToDesignNodeIterate(childNode);
    });

    designNode.blocks = blocks;
  }

  return designNode;
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

export function transferBusinessContentToDesignContent(
  businessContent: IBusinessNode,
  nodeRegistries: FlowNodeRegistry[]
): FlowDocumentJSON {

  const { layout, groups = [] } = businessContent.nndeploy_ui_layout

  function businessNodeIterate(
    businessNode: IBusinessNode,
    process: (businessNode: IBusinessNode) => void
  ) {
    process(businessNode);

  }
  /** 获取所有node的 输出节点的 name_->{nodeId, portId}映射表 */
  function getAllOutputPortNameToNodePortIdMap() {
    const outputMap: { [key: string]: { nodeId: string; portId: string } } = {};

    function processNode(businessNode: IBusinessNode, parentNodePath: string[]) {
      if (businessNode.outputs_) {
        businessNode.outputs_?.map((output) => {
          if (output.name_) {

            const outputFullName = [
              //...parentNodePath, 
              output.name_].join("!")
            outputMap[outputFullName] = {
              nodeId: businessNode.id,
              portId: output.id,
            };
          }
        });
      }
      const children = businessNode.node_repository_ ?? [];
      children.map((childNode) => {
        processNode(childNode, [...parentNodePath, businessNode.name_]);
      });
    }

    businessContent.node_repository_?.map((businessNode) => {
      processNode(businessNode, []);
    });

    return outputMap;
  }
  /** 获取所有node的 输入节点的 名字->[{nodeId, portId}]映射表 */
  function getAllInputPortNameToNodePortIdMap() {
    const inputMap: { [key: string]: { nodeId: string; portId: string }[] } = {};


    function processNode(businessNode: IBusinessNode, parentNodePath: string[]) {
      if (businessNode.inputs_) {
        businessNode.inputs_?.map((input) => {
          if (input.name_) {

            const outputFullName = [
              //...parentNodePath, 
              input.name_].join("!")
            if (inputMap[outputFullName]) {
              inputMap[outputFullName] = [...inputMap[outputFullName], {
                nodeId: businessNode.id,
                portId: input.id,
              }];
            } else {
              inputMap[outputFullName] = [{
                nodeId: businessNode.id,
                portId: input.id,
              }];
            }

          }
        });
      }
      const children = businessNode.node_repository_ ?? [];
      children.map((childNode) => {
        processNode(childNode, [...parentNodePath, businessNode.name_]);
      });
    }

    businessContent.node_repository_?.map((businessNode) => {
      processNode(businessNode, []);
    });

    return inputMap;
  }
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





  const outputMap = getAllOutputPortNameToNodePortIdMap();

  console.log('outputMap', outputMap)

  const inputMap = getAllInputPortNameToNodePortIdMap();

  console.log('inputMap', inputMap)

  const nodeRepository = businessContent.node_repository_ ?? [];

  for (let i = 0; i < nodeRepository.length; i++) {
    let businessNode = nodeRepository[i];
    let designNode = transferBusinessNodeToDesignNodeIterate(businessNode);
    designData.nodes = [...designData.nodes, designNode];
  }

  function buildEdges() {
    var edges: WorkflowEdgeJSON[] = [];

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
        };

        edges.push(connection);
      })



    }
    return edges


  }

  const edges = buildEdges()

  designData.edges = edges;



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




  return designData;
}

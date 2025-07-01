import {
  WorkflowEdgeJSON,
  WorkflowNodeJSON,
} from "@flowgram.ai/free-layout-editor";
import { FlowDocumentJSON, FlowNodeJSON, FlowNodeRegistry } from "../../../../typings";
import {
  IBusinessNode,
  IWorkFlowEntity,
} from "../../../Layout/Design/WorkFlow/entity";
import { resourceLimits } from "worker_threads";
import { random } from "lodash";

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

function getAllEdges(designData: FlowDocumentJSON) {
  let edges = designData.edges;

  function nodeIterate(
    node: WorkflowNodeJSON,
    process: (node: WorkflowNodeJSON) => void
  ) {
    process(node);
    if (node.blocks && node.blocks.length > 0) {
      node.blocks.forEach((block) => {
        nodeIterate(block, process);
      });
    }
  }

  designData.nodes.map((node) => {
    nodeIterate(node, (node) => {
      if (node.edges && node.edges.length > 0)
        edges = [...edges, ...node.edges];
    });
  });

  var results: WorkflowEdgeJSON[] = [];

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

  return results;
}

export function designDataToBusinessData(designData: FlowDocumentJSON) {
  let allEdges = getAllEdges(designData);

  let businessData: IBusinessNode = {
    key_: "nndeploy::dag::Graph",
    name_: "demo",
    device_type_: "kDeviceTypeCodeCpu:0",
    inputs_: [],
    outputs_: [],
    is_external_stream_: false,
    is_inner_: false,
    is_time_profile_: false,
    is_debug_: false,
    is_graph_node_share_stream_: true,
    queue_max_size_: 16,
    node_repository_: [],
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
      for (let i = 0; i < designData.nodes.length; i++) {
        let node = designData.nodes[i];
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
    edge_map[edge.sourceNodeID + "@" + edge.sourcePortID] = getEdgeNameById(edge.sourceNodeID as string, edge.sourcePortID as string);
    edge_map[edge.targetNodeID + "@" + edge.targetPortID] = getEdgeNameById(edge.sourceNodeID as string, edge.sourcePortID as string);
  });

  function transferDesignNodeToBusinessNode(
    node: WorkflowNodeJSON
  ): IBusinessNode {
    function buildCollectionName(findEdge: WorkflowEdgeJSON) {
      var sourceNodeName = getNodeNameById(findEdge.sourceNodeID as string);
      var targetNodeName = getNodeNameById(findEdge.targetNodeID as string);

      var temp1 = document.querySelectorAll(
        `[data-port-id=${findEdge.sourcePortID}]`
      );
      var temp2 = temp1.item(0);

      let sourcePortDesc = document
        .querySelectorAll(`[data-port-id='${findEdge.sourcePortID}']`)
        .item(0)
        .getAttribute("data-port-desc");
      let descPortDesc = document
        .querySelectorAll(`[data-port-id='${findEdge.targetPortID}']`)
        .item(0)
        .getAttribute("data-port-desc");

      const name = [
        sourceNodeName,
        targetNodeName,
        // item.sourcePortID,
        // item.targetPortID,
        sourcePortDesc,
        descPortDesc,
      ].join("@");

      return name;
    }

    // const inputArray_ = (node.data.inputs_ ?? []).map((input) => {
    //   let results = allEdges
    //     .filter((edge) => {
    //       return edge.targetNodeID === node.id && edge.targetPortID == input.id;
    //     })
    //     .map((edge) => {
    //       const name_ = buildCollectionName(edge);
    //       return {
    //         ...input,
    //         name_,
    //       };
    //     });
    //   if (results.length < 1) {
    //     return [input];
    //   } else {
    //     return results;
    //   }
    // });

    // const inputs_ = inputArray_.flat();

    const inputArray_ = (node.data.inputs_ ?? []).map((input:any) => {
      const name_ = edge_map[node.id + "@" + input.id]
        return {
          ...input,
          name_,
        };
    });
    
    const inputs_ = inputArray_.flat();

    // const inputs_ = inputArray_.flat();

    // const outputArray_ = (node.data.outputs_ ?? []).map((output) => {
    //   let results = allEdges
    //     .filter((edge) => {
    //       return (
    //         edge.sourceNodeID === node.id && edge.sourcePortID == output.id
    //       );
    //     })
    //     .map((edge) => {
    //       const name_ = buildCollectionName(edge);
    //       return {
    //         ...output,
    //         name_,
    //       };
    //     });
    //   if (results.length < 1) {
    //     return [output];
    //   } else {
    //     return results;
    //   }
    // });

    // const outputs_ = outputArray_.flat();

    const outputArray_ = (node.data.outputs_ ?? []).map((output:any) => {
      const name_ = edge_map[node.id + "@" + output.id]
        return {
          ...output,
          name_,
        };
    });

    const outputs_ = outputArray_.flat();

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
    return businessNode;
  }

  

  let node_repository_: IBusinessNode[] = designData.nodes.map((node) => {
    return transferDesignNodeToBusinessNode(node);
  });

  businessData.node_repository_ = node_repository_;

  return businessData;
}

export function transferBusinessContentToDesignContent(
  businessContent: IBusinessNode, 
  nodeRegistries: FlowNodeRegistry[]
): FlowDocumentJSON {
  function businessNodeIterate(
    businessNode: IBusinessNode,
    process: (businessNode: IBusinessNode) => void
  ) {
    process(businessNode);
    if (
      businessNode.node_repository_ &&
      businessNode.node_repository_.length > 0
    ) {
      let nodeRepositories = businessNode.node_repository_
        ? businessNode.node_repository_
        : [];
      nodeRepositories.forEach((childBusinssNode) => {
        businessNodeIterate(childBusinssNode, process);
      });
    }
  }

  function getAllOutputNameMap() {
    const outputMap: { [key: string]: { nodeId: string; portId: string } } = {};

    businessContent.node_repository_?.map((businessNode) => {
      businessNodeIterate(businessNode, function (businessNode) {
        if (businessNode.outputs_) {
          businessNode.outputs_?.map((output) => {
            if (output.name_) {
              outputMap[output.name_] = {
                nodeId: businessNode.id,
                portId: output.id,
              };
            }
          });
        }
      });
    });

    return outputMap;
  }

  function getAllInputNameMap() {
    const inputMap: { [key: string]: { nodeId: string; portId: string }[] } = {};

    businessContent.node_repository_?.map((businessNode) => {
      businessNodeIterate(businessNode, function (businessNode) {
        if (businessNode.inputs_) {
          businessNode.inputs_?.map((input) => {
            if (input.name_) {

              if(inputMap[input.name_]){
                inputMap[input.name_] = [ ...inputMap[input.name_], {
                  nodeId: businessNode.id,
                  portId: input.id,
                }];
              }else{
                inputMap[input.name_] = [ {
                  nodeId: businessNode.id,
                  portId: input.id,
                }];
              }
             
            }
          });
        }
      });
    });

    return inputMap;
  }

  function businessContentNormalize(businessContent: IBusinessNode) {
    function businessNodeNormalize(businessNode: IBusinessNode) {
      businessNode.id = businessNode.id
        ? businessNode.id
        : "node" + Math.random().toString(36).substr(2, 9);

      let inputs_ = businessNode.inputs_ ?? [];

      let inputCollectionNameIdMap: { [key: string]: string } = {};
      inputs_.map((item) => {
        if (!item.id) {
          let portId = "";
          if (inputCollectionNameIdMap[item.desc_]) {
            portId = inputCollectionNameIdMap[item.desc_];
          } else {
            portId = "port" + Math.random().toString(36).substr(2, 9);
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
            portId = "port" + Math.random().toString(36).substr(2, 9);
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

  function transferBusinessNodeToDesignNode(
    businessNode: IBusinessNode
  ): FlowNodeJSON {

    if(businessNode.is_graph_){
      debugger
    }

    var type = businessNode.is_graph_ ? 'group':  businessNode.key_
    const designNode: FlowNodeJSON = {
      id: `${businessNode.id}`,
      //type: businessNode.key_.split("::").pop() + "",
      type, 

      meta: {
        position: {
          x: 0,
          y: 0,
        },
      },
      data: {
        // ...businessNode,
      },
    };

    if (
      businessNode.node_repository_ &&
      businessNode.node_repository_.length > 0
    ) {
      const nodeRepositories = businessNode.node_repository_;
      delete businessNode["node_repository_"];
      const blocks = nodeRepositories.map((childNode) => {
        return transferBusinessNodeToDesignNode(childNode);
      });

      designNode.blocks = blocks;
    }

    function uniqCollectionPoint(businessNode: IBusinessNode, type: string) {
      let collectionPoints: any[] = [];

      businessNode[type].map((collectionPoint :any) => {
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

    return designNode;
  }

  const outputMap = getAllOutputNameMap();

  const inputMap = getAllInputNameMap();

  const nodeRepository = businessContent.node_repository_ ?? [];

  for (let i = 0; i < nodeRepository.length; i++) {
    let businessNode = nodeRepository[i];
    let designNode = transferBusinessNodeToDesignNode(businessNode);
    designData.nodes = [...designData.nodes, designNode];
  }

  var edges: WorkflowEdgeJSON[] = [];

  for (let connectionName in outputMap) {
    var outputInfo = outputMap[connectionName];
    var sourceInfos = inputMap[connectionName];

    sourceInfos.map(sourceInfo=>{
       var connection: WorkflowEdgeJSON = {
      sourceNodeID: outputInfo.nodeId,
      targetNodeID: sourceInfo.nodeId,
      sourcePortID: outputInfo.portId,
      targetPortID: sourceInfo.portId,
    };

    edges.push(connection);
    })

   

    designData.edges = edges;
  }
  return designData;
}

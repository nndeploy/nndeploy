import { WorkflowNodeJSON } from "@flowgram.ai/free-layout-editor";
import { FlowDocumentJSON } from "../../../../typings";
import { IBusinessNode } from "../../../Layout/Design/WorkFlow/entity";

export function buildBusinessDataFromDesignData(designData: FlowDocumentJSON) {
  let businessData: IBusinessNode = {
    key_: "nndeploy::dag::Graph",
    name_: "demo",
    device_type_: "kDeviceTypeCodeX86:0",
    inputs_: [],
    outputs_: [
      {
        name_: "detect_out",
        type_: "kNotSet",
      },
    ],
    is_external_stream_: false,
    is_inner_: false,
    "is_time_p  rofile_": true,
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
        } else {
          return "";
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

  function transferDesignNodeToBusinessNode(
    node: WorkflowNodeJSON
  ): IBusinessNode {
    var inputs_ = designData.edges
      .filter((edge) => {
        return edge.targetNodeID === node.id;
      })
      .map((item, index) => {
        var sourceNodeName = getNodeNameById(item.sourceNodeID as string);
        var targetNodeName = getNodeNameById(item.targetNodeID as string);

        if(!node.data.inputs_[index]){
          debugger;
        }
        var temp1 = document.querySelectorAll(`[data-port-id=${item.sourcePortID}]`); 
        var temp2 = temp1.item(0)
        let sourcePortDesc = document.querySelectorAll(`[data-port-id='${item.sourcePortID}']`).item(0).getAttribute('data-port-desc')
        let descPortDesc = document.querySelectorAll(`[data-port-id='${item.targetPortID}']`).item(0).getAttribute('data-port-desc')

        let find = node.data.inputs_.find( input=>{
          return input.id == item.targetPortID
        })

        return {
          name: [
            sourceNodeName,
            targetNodeName,
            // item.sourcePortID,
            // item.targetPortID,
            sourcePortDesc, 
            descPortDesc

          ].join("_"),
          type_:find.type_,
        };
      });

    var outputs_ = designData.edges
      .filter((edge) => {
        return edge.sourceNodeID === node.id;
      })
      .map((item, index) => {
        var sourceNodeName = getNodeNameById(item.sourceNodeID as string);
        var targetNodeName = getNodeNameById(item.targetNodeID as string);

        if(!node.data.outputs_[index]){
          debugger;
        }

        let find = node.data.outputs_.find( output=>{
          return output.id == item.sourcePortID
        })

        var temp3 = document.querySelectorAll(`[data-port-id=${item.sourcePortID}]`); 
        var temp4 = temp3.item(0)

          let sourcePortDesc = document.querySelectorAll(`[data-port-id='${item.sourcePortID}']`).item(0).getAttribute('data-port-desc')
        let descPortDesc = document.querySelectorAll(`[data-port-id='${item.targetPortID}']`).item(0).getAttribute('data-port-desc')



        return {
          name: [
            sourceNodeName,
            targetNodeName,
            // item.sourcePortID,
            // item.targetPortID,
             sourcePortDesc, 
            descPortDesc
          ].join("_"),
          type_: find.type_,
        };
      });

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

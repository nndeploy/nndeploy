import {
  IBusinessNode,
} from "../../../Layout/Design/WorkFlow/entity";


export function businessNodeIterate(
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





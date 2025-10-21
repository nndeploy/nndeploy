import { nanoid } from "nanoid";
import { FlowNodeRegistry } from "../../../../typings";
import { INodeEntity } from "../../../Node/entity";
import { formMeta } from "./form-meta";
import iconCondition from '../../../../assets/icon-condition.svg';


export function buildNodeRegistry(nodeEntity: INodeEntity) {


  //let formMetaTemp =  ['nndeploy::detect::YoloGraph', ''].includes( nodeEntity.key_) ? groupFormMeta: formMeta

  //let type = ['nndeploy::detect::YoloGraph'].includes( nodeEntity.key_) ? 'group': nodeEntity.key_
  const nodeRegistry: FlowNodeRegistry = {
    type: nodeEntity.key_,
    info: {
      icon: iconCondition,
      description:
        "Connect multiple downstream branches. Only the corresponding branch will be executed if the set conditions are met.",
    },
    meta: {
      //defaultPorts: [{ type: "input" }],
      // Condition Outputs use dynamic port
      isContainer: !!nodeEntity.is_graph_ ,
      useDynamicPort: true,
      dynamicPort: true,
      disableSideBar: !!nodeEntity.is_graph_,
      expandable: !!nodeEntity.is_graph_, // disable expanded

      padding: () => ({
        top: 25,
        bottom: 5,
        left: 15,
        right: 15,
      }),
    },
    formMeta: formMeta,
    onAdd() {
      return {
        id: `${nodeEntity.key_}_${nanoid(5)}`,
        type: nodeEntity.key_,
        data: nodeEntity

      };
    },
  };

  return nodeRegistry;
}

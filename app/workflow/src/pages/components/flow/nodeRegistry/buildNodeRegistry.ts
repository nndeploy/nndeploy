import { nanoid } from "nanoid";
import { FlowNodeRegistry } from "../../../../typings";
import { INodeEntity } from "../../../Node/entity";
import { formMeta } from "./form-meta";
import iconCondition from '../../../../assets/icon-condition.svg';
import iconLoop from '../../../../assets/icon-loop.jpg';
import iconComposite from '../../../../assets/compose-line.png';
import iconGraph from '../../../../assets/photography.png';
import { isNodeOffspringOfCompositeNode, isNodeOffSpringOfFixedGraph } from "../functions";


export function buildNodeRegistry(nodeEntity: INodeEntity) {


  //let formMetaTemp =  ['nndeploy::detect::YoloGraph', ''].includes( nodeEntity.key_) ? groupFormMeta: formMeta

  //let type = ['nndeploy::detect::YoloGraph'].includes( nodeEntity.key_) ? 'group': nodeEntity.key_

  const isContainer = nodeEntity.is_loop_ || nodeEntity.is_graph_ || nodeEntity.is_composite_node_

  function getIcon() {

    if (nodeEntity.is_loop_) {
      return iconLoop
    }
    else if (nodeEntity.is_composite_node_) {
      return iconComposite
    }
    else if (nodeEntity.is_graph_) {
      return iconGraph //
    } else {
      return undefined
    }
  }

  const nodeRegistry: FlowNodeRegistry = {
    type: nodeEntity.key_,
    info: {
      icon: getIcon(),
      description:
        "Connect multiple downstream branches. Only the corresponding branch will be executed if the set conditions are met.",
    },
    meta: {
      //defaultPorts: [{ type: "input" }],
      // Condition Outputs use dynamic port
      isContainer: isContainer,
      useDynamicPort: true,
      dynamicPort: true,
      //disableSideBar: true,
      //disableSideBar: !!nodeEntity.is_graph_,
      expandable: isContainer, // disable expanded

      padding: () => ({
        top: 45, //25
        bottom: 25, //5
        left: 15,
        right: 15,
      }),
    },
    formMeta: formMeta,
    canDelete(ctx, node) {
      return  isNodeOffspringOfCompositeNode(node, ctx) || isNodeOffSpringOfFixedGraph(node, ctx) ? false: true
    },
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

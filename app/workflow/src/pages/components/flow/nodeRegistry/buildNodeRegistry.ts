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
      size: {
        width: isContainer ? 400 : 200,
        height: isContainer ? 160 : 80,
      },

      // padding: (transform) => {
      //   // if (!transform.isContainer) {
      //   //   return {
      //   //     top: 0,
      //   //     bottom: 0,
      //   //     left: 0,
      //   //     right: 0,
      //   //   };
      //   // }
      //   return {
      //     top: 55, //25
      //     bottom: 45, //5
      //     left: 35,
      //     right: 35,
      //   };
      // },

      padding: () => ({
        top: 55, //25
        bottom: 45, //5
        left: 35,
        right: 35,
      }),
    },
    formMeta: formMeta,
    canDelete(ctx, node) {
      return isNodeOffspringOfCompositeNode(node, ctx) || isNodeOffSpringOfFixedGraph(node, ctx) ? false : true
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

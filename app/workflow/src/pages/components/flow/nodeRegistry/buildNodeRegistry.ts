import { nanoid } from "nanoid";
import { FlowNodeRegistry } from "../../../../typings";
import { INodeEntity } from "../../../Node/entity";
import { formMeta } from "./form-meta";
import iconCondition from '../../../../assets/icon-condition.svg';


export function buildNodeRegistry(nodeEntity: INodeEntity) {
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
      useDynamicPort: true,
      dynamicPort: true, 
      expandable: true, // disable expanded
    },
    formMeta,
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

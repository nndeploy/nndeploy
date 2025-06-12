import { FlowNodeRegistry } from "../../typings";
import iconStart from "../../assets/icon-start.jpg";
import { formMeta } from "./form-meta";
import { WorkflowNodeType } from "../constants";
import { nanoid } from "nanoid";

let index = 0;
export const MyNodeRegistry: FlowNodeRegistry = {
  type: WorkflowNodeType.My,
  meta: {
    isStart: false,
    deleteDisable: true,
    copyDisable: true,
    //defaultPorts: [{ type: 'output' }],
    size: {
      width: 360,
      height: 211,
    },
  },
  info: {
    icon: iconStart,
    description:
      "The starting node of the workflow, used to set the information needed to initiate the workflow.",
  },
  /**
   * Render node via formMeta
   */
  formMeta,
  /**
   * Start Node cannot be added
   */
  canAdd() {
    return true;
  },
  onAdd() {
    return {
      id: `llm_${nanoid(5)}`,
      type: WorkflowNodeType.My,
      data: {
        title: `My_${++index}`,
        inputsValues: {},
        // inputs: {
        //   type: 'object',
        //   required: ['modelType', 'temperature', 'prompt'],
        //   properties: {
        //     modelType: {
        //       type: 'string',
        //     },
        //     temperature: {
        //       type: 'number',
        //     },
        //     systemPrompt: {
        //       type: 'string',
        //     },
        //     prompt: {
        //       type: 'string',
        //     },
        //   },
        // },
        outputs: {
          type: "object",
          properties: {
            query: {
              type: "string",
              default: "Hello Flow.",
            },
          },
        },
      },
    };
  },
};

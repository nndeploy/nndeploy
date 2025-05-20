import {
  WorkflowNodeJSON as FlowNodeJSONDefault,
  WorkflowNodeRegistry as FlowNodeRegistryDefault,
  FreeLayoutPluginContext,
  FlowNodeEntity,
  type WorkflowEdgeJSON,
  WorkflowNodeMeta,
} from '@flowgram.ai/free-layout-editor';

import { type JsonSchema } from './json-schema';

export type FlowLiteralValueSchema = string | number | boolean;
export type FlowObjectValueSchema = Record<string, FlowLiteralValueSchema | FlowRefValueSchema>;
export type FlowArrayValueSchema = FlowObjectValueSchema[];
export type FlowRefValueSchema =
  | { type: 'ref'; content?: string }
  | { type: 'expression'; content?: string }
  | { type: 'template'; content?: string };
export type FlowValueSchema = FlowLiteralValueSchema | FlowRefValueSchema | FlowArrayValueSchema;
/**
 * You can customize the data of the node, and here you can use JsonSchema to define the input and output of the node
 * 你可以自定义节点的 data 业务数据, 这里演示 通过 JsonSchema 来定义节点的输入/输出
 */
export interface FlowNodeJSON extends FlowNodeJSONDefault {
  data: {
    /**
     * Node title
     */
    title?: string;
    /**
     * Inputs data values
     */
    inputsValues?: Record<string, FlowValueSchema>;
    /**
     * Define the inputs data of the node by JsonSchema
     */
    inputs?: JsonSchema;
    /**
     * Define the outputs data of the node by JsonSchema
     */
    outputs?: JsonSchema;
    /**
     * Rest properties
     */
    [key: string]: any;
  };
}

/**
 * You can customize your own node meta
 * 你可以自定义节点的meta
 */
export interface FlowNodeMeta extends WorkflowNodeMeta {
  disableSideBar?: boolean;
}

/**
 * You can customize your own node registry
 * 你可以自定义节点的注册器
 */
export interface FlowNodeRegistry extends FlowNodeRegistryDefault {
  meta: FlowNodeMeta;
  info?: {
    icon: string;
    description: string;
  };
  canAdd?: (ctx: FreeLayoutPluginContext) => boolean;
  canDelete?: (ctx: FreeLayoutPluginContext, from: FlowNodeEntity) => boolean;
  onAdd?: (ctx: FreeLayoutPluginContext) => FlowNodeJSON;
}

export interface FlowDocumentJSON {
  nodes: FlowNodeJSON[];
  edges: WorkflowEdgeJSON[];
}

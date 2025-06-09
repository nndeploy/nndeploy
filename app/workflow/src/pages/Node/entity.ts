import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { JsonSchema } from "../components/type-selector/types";


export interface INodeTreeNodeEntity{
  id: string; 
  name: string; 
  parentId: string; 
  type: 'branch' | 'leaf'
  //isLeaf: boolean;
  
}

export interface INodeBranchEntity{
  id: string; 
  name: string; 
  parentId: string; 
  
}


export interface NodeTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity:INodeTreeNodeEntity, 
  children?: NodeTreeNodeData[];
}


// export interface INodeEntityConfigField{
//   name:string; 
//   label: string; 
//   type: 'number' | 'string' | 'boolean' | 'file' | 'object';
//   children?: INodeEntityConfigField[]
// }

export interface IConnectionPoint{
  name_: string;
  type_: string; 
}

export interface INodeEntity{
  key_: string;
  name_: string;
  device_type_: string; 
  inputs_: IConnectionPoint[];
  outputs_: IConnectionPoint[]
  param_: {
    [key:string]: any
  }, 
   [key:string]: any
  //schema: JsonSchema
}

export interface INodeInstance{
  id: string; 
  key_: string;
  name_: string;
  device_type_: string; 
  inputs_: IConnectionPoint[];
  outputs_: IConnectionPoint[]
  param_: {
    [key:string]: any
  }, 
  [key:string]: any
  //schema: JsonSchema
}





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

export interface INodeEntity{
  id: string;
  name: string;
  parentId: string;
  schema: JsonSchema
}




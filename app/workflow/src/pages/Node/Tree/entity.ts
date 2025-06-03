import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";


export interface INodeBranchEntity{
  id: string; 
  name: string; 
  parentId: string; 
  
}


export interface NodeBranchTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity:INodeBranchEntity, 
  children?: NodeBranchTreeNodeData[];
}


export interface INodeEntityConfigField{
  name:string; 
  label: string; 
  type: 'number' | 'string' | 'boolean' | 'file' | 'object';
  children?: INodeEntityConfigField[]
}

export interface INodeEntity{
  id: string;
  name: string;
  parentId: string;
  config: INodeEntityConfigField[]
}




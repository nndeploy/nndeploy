import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { IBusinessNode } from "../Layout/Design/WorkFlow/entity";


export interface IDagGraphInfo{
  accepted_edge_types: {[nodeKey:string]:string[]}
  graph: IBusinessNode, 
  nodes: INodeTreeNodeEntity[]
}

export interface INodeTreeNodeEntity{
  id: string;  //可以是目录的名字
  name: string;  //节点名字 , 
  desc: string;  //气泡提示
  parentId: string;   //父节点id, 可以是目录的名字
  type: 'branch' | 'leaf'   //branch: 目录, leaf: 节点
  nodeEntity?: INodeEntity  //如果是目录为空; 如果是node, 以前的的node内容

}

export interface INodeBranchEntity{
  id: string; 
  name: string; 
  parentId: string; 
  
}


export interface NodeTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  nodeEntity:INodeTreeNodeEntity, 
  children?: NodeTreeNodeData[];
}


// export interface INodeEntityConfigField{
//   name:string; 
//   label: string; 
//   type: 'number' | 'string' | 'boolean' | 'file' | 'object';
//   children?: INodeEntityConfigField[]
// }

export interface IConnectionPoint{
  desc_: string;
  type_: string; 
}

export interface INodeEntity{
  key_: string;
  name_: string;
  device_type_: string; 
  inputs_: IConnectionPoint[];
  outputs_: IConnectionPoint[]
  param_?: {
    [key:string]: any
  }, 
  is_graph_?: boolean;

  node_repository_?: INodeEntity[]; 
  
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





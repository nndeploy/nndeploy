import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { FlowDocumentJSON } from "../../../../typings";

export interface IWorkFlowTreeNodeEntity{
  id: string; 
  name: string; 
  parentId: string; 
  type: 'branch' | 'leaf'
  //isLeaf: boolean;
  
}

export interface IWorkFlowBranchEntity{
  id: string; 
  name: string; 
  parentId: string; 
  
}

export interface IBusinessNode{
  key_: string; 
  name_: string; 
  device_type_: string; 
  inputs_:any[], 
  outputs_:any[], 
  node_repository_?: IBusinessNode[], 
  [key: string]: any;
}

export interface IWorkFlowEntity{
  id: string; 
  name: string; 
  parentId: string; 
  designContent: FlowDocumentJSON, 
  businessContent: IBusinessNode, 
  
  
}

export interface IWorkFlowRunResult{
  task_id: string;
}

export interface WorkFlowTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity:IWorkFlowTreeNodeEntity, 
  children?: WorkFlowTreeNodeData[];
}

export interface IParamTypes{
  [key:string]: string[]
}

export interface IFieldType{
   isArray: boolean, 
   componentType: string; 
   primateType: string; 
   selectOptions: string[]; 
   selectKey: string; 
}
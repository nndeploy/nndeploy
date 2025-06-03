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

export interface IWorkFlowEntity{
  id: string; 
  name: string; 
  parentId: string; 
  content: FlowDocumentJSON
  
}

export interface WorkFlowTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity:IWorkFlowTreeNodeEntity, 
  children?: WorkFlowTreeNodeData[];
}

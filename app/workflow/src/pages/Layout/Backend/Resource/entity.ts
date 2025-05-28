import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";

export interface ResourceTreeNodeData extends TreeNodeData {
  type: "leaf" | "branch";
  mime?: string;
  url?: string;
}

export interface ResourceBranchSaveEntity  {
  id: string; 
  parentId: string; 
  name: string
}



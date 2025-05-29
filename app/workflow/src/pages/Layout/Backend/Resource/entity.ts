import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";

export interface IResourceEntity{
  id: string; 
  name: string; 

  parentId: string; 
  children?: IResourceEntity[];

  isLeaf?: boolean;

  mime?: string;
  url?: string;

}

export interface ResourceTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity:IResourceEntity, 
  children?: ResourceTreeNodeData[];
}

export function resourceToTreeNodeData(entity: IResourceEntity): ResourceTreeNodeData {
  return {
    key: entity.id,
    label: entity.name,
    isLeaf: entity.isLeaf,
    children: entity.children? entity.children.map(resourceToTreeNodeData) : undefined,
    entity: entity
  };
}

// export interface ResourceBranchSaveEntity  {
//   id: string; 
//   parentId: string; 
//   name: string
// }



import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";


export interface IResourceTreeNodeEntity{
  id: string; 
  name: string; 
  parentId: string; 
  type: 'branch' | 'leaf'
  //isLeaf: boolean;
  
}

export interface IResourceBranchEntity{
  id: string; 
  name: string; 
  parentId: string; 
  
}

export interface IResourceEntity{
  id: string; 
  name: string; 

  parentId: string; 
  //children?: IResourceEntity[];
  mime: string;
  url: string;

}

export interface ResourceTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity:IResourceTreeNodeEntity, 
  children?: ResourceTreeNodeData[];
}

// export function resourceToTreeNodeData(entity: IResourceEntity): ResourceTreeNodeData {
//   return {
//     key: entity.id,
//     label: entity.name,
//     isLeaf: entity.type,
//     children: entity.children? entity.children.map(resourceToTreeNodeData) : undefined,
//     entity: entity
//   };
// }

// export interface ResourceBranchSaveEntity  {
//   id: string; 
//   parentId: string; 
//   name: string
// }



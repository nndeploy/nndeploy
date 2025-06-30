import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";


export interface IResourceTreeNodeEntity {
  id: string;
  name: string;
  parentId: string;
  type: 'branch' | 'leaf', 
  entity?: IServerResourceFileEntity

  
  //isLeaf: boolean;

}

export interface IServerResourceFileEntity {
  filename: string; //"sample.jpg",
  saved_path: string; //"/home/lds/nndeploy/server/resources/images/sample.jpg",
  size: number; 
  uploaded_at: string; //"2025-06-28 17:46:55",
  extension: string; //"jpg"
}
export interface IServerResourceResonseData {
  [resourceType: string]: IServerResourceFileEntity[]
}

export interface IResourceBranchEntity {
  id: string;
  name: string;
  parentId: string;

}

export interface IResourceEntity {
  id: string;
  name: string;

  parentId: string;
  //children?: IResourceEntity[];
 // mime: string;
 // url: string;

  saved_path: string; 
  size: number; 
  uploaded_at:string

}

export interface ResourceTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity: IResourceTreeNodeEntity,
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


// export interface IFileUploadResponseData {
//   filename: string; //"1665212274726.png",
//   saved_path: string; // "images/1665212274726.png",
//   size: number; //93505,
//   uploaded_at: string; //"2025-06-28T11:53:46.478764"
// }



import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { ResourceBranchSaveEntity } from "./entity";

export async function apiGetResourceTree(){
 var response = await request.get<TreeNodeData[]>('/resource/tree', {});

  return response;
}


export async function apiResourceBranchSave(entity: ResourceBranchSaveEntity){
 var response = await request.post<ResourceBranchSaveEntity>('/resource/branch/save', entity);

  return response;
}
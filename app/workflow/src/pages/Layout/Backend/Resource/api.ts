import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { IResourceEntity } from "./entity";

export async function apiGetResourceTree(){
 var response = await request.get<IResourceEntity[]>('/resource/tree', {});

  return response;
}


export async function apiResourceBranchSave(entity: IResourceEntity){
 var response = await request.post<IResourceEntity>('/resource/branch/save', entity);

  return response;
}


export async function apiResourceSave(entity: IResourceEntity){
 var response = await request.post<IResourceEntity>('/resource/save', entity);

  return response;
}

export async function apiResourceUpload(formData: FormData|any){
 var response = await request.post<IResourceEntity>('/resource/upload', formData);

  return response;
}

export async function apiResourceDelete(id: string){
 var response = await request.post<any>('/resource/delete', {id});

  return response;
}
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { IResourceBranchEntity, IResourceEntity, IResourceTreeNodeEntity, IServerResourceFileEntity, IServerResourceResonseData } from "./entity";

export async function apiGetResourceTree(){
 var response = await request.get<IServerResourceResonseData>('/api/files', {});

  return response;
}

export async function apiGetResource(id: string){
 var response = await request.post<IResourceEntity>('/resource/get', {id});

  return response;
}


export async function apiResourceBranchSave(entity: IResourceBranchEntity){
 var response = await request.post<IResourceBranchEntity>('/resource/branch/save', entity);

  return response;
}


export async function apiResourceSave(url:string, formData: FormData){
 var response = await request.upload<IServerResourceFileEntity>(url, formData);

  return response;
}

export async function apiResourceUpload(formData: FormData|any){
 var response = await request.post<IResourceEntity>('/resource/upload', formData);

  return response;
}

export async function apiResourceDelete(url: string){
 var response = await request.post<any>(url, {});

  return response;
}
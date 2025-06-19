import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { IBusinessNode, IWorkFlowEntity, IWorkFlowTreeNodeEntity } from "./entity";

export async function apiGetWorkFlowTree(){
 var response = await request.get<IWorkFlowTreeNodeEntity[]>('/workflow/tree', {});

  return response;
}

export async function apiGetWorkFlowBranch(){
 var response = await request.get<IWorkFlowTreeNodeEntity[]>('/workflow/branch', {});

  return response;
}



export async function apiWorkFlowBranchSave(entity: IWorkFlowTreeNodeEntity){
 var response = await request.post<IWorkFlowTreeNodeEntity>('/workflow/branch/save', entity);

  return response;
}


export async function apiWorkFlowSave(entity: IBusinessNode){
 var response = await request.post<IBusinessNode>('/workflow/save', entity);

  return response;
}

export async function apiGetWorkFlow(id: string){
 var response = await request.post<IWorkFlowEntity>('/workflow/get', {id});

  return response;
}


// export async function apiworkFlowUpload(formData: FormData|any){
//  var response = await request.post<IWorkFlowTreeNodeEntity>('/workFlow/upload', formData);

//   return response;
// }

export async function apiWorkFlowDelete(id: string){
 var response = await request.post<any>('/workflow/delete', {id});

  return response;
}
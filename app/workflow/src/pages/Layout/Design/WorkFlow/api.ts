import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { IBusinessNode, ITreeWorkFlowResponseData, IWorkFlowEntity, IWorkFlowRunResult, IWorkFlowTreeNodeEntity } from "./entity";

export async function apiGetWorkFlowTree(){
 var response = await request.get<ITreeWorkFlowResponseData>('/api/workflow', {});

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
 var response = await request.post<IBusinessNode>('/api/workflow/save', entity);

  return response;
}

export async function apiWorkFlowRun(entity: IBusinessNode){
 var response = await request.post<IWorkFlowRunResult>('/api/queue', entity);

  return response;
}

export async function apiGetWorkFlow(name: string){
 var response = await request.post<IWorkFlowEntity>(`aip/workflow/${name}`, {});

  return response;
}


// export async function apiworkFlowUpload(formData: FormData|any){
//  var response = await request.post<IWorkFlowTreeNodeEntity>('/workFlow/upload', formData);

//   return response;
// }

export async function apiWorkFlowDelete(id: string){
 var response = await request.post<any>(`/api/workflow/delete/${id}`, {});

  return response;
}
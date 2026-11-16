import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { IBusinessNode, ITreeWorkFlowResponseData, IWorkFlowEntity, IWorkFlowRunResult, IWorkFlowTreeNodeEntity } from "./entity";
import { IWorkFlowShortEntity } from "../../../../entity";
import type { RunTab } from '../../../../components/tools/run-session';
import { submitRunTask } from '../../../../components/tools/run-connection';
import type { RunTaskState } from '../../../../components/tools/run-state';

// export async function apiGetWorkFlowTree(){
//  var response = await request.get<IWorkFlowShortEntity>('/api/workflows', {});

//   return response;
// }

export async function apiGetWorkFlowBranch(){
 var response = await request.get<IWorkFlowTreeNodeEntity[]>('/workflow/branch', {});

  return response;
}



export async function apiWorkFlowBranchSave(entity: IWorkFlowTreeNodeEntity){
 var response = await request.post<IWorkFlowTreeNodeEntity>('/workflow/branch/save', entity);

  return response;
}


export async function apiWorkFlowSave(id: string, entity: IBusinessNode){
 var response = await request.post<IBusinessNode>('/api/workflow/save', { id,  businessContent: entity});

  return response;
}

export async function apiWorkFlowRun(entity: IBusinessNode, tab?: RunTab){
  return submitRunTask(entity, tab);
}

export async function apiWorkFlowRunStop(taskId:string){
 var response = await request.post<RunTaskState>(`/api/queue/cancel/${taskId}`, {});

  return response;
}

export async function apiModelsRunDownload(entity: IBusinessNode){

 var response = await request.post<IWorkFlowRunResult>('/api/models/download', entity);

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

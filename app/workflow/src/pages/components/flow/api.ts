import request from "../../../request";
import { IWorkFlowEntity } from "../../Layout/Backend/WorkFlow/entity";

export async function apiGetNodeType  (nodeTypeId: string) {
   var response = await request.post<any>('/nodeType/get', {id: nodeTypeId});

  return response;
}

export async function apiGetWorkFlow(id: string){
 var response = await request.post<IWorkFlowEntity>('/workflow/get', {id});

  return response;
}
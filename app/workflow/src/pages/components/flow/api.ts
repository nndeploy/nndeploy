import request from "../../../request";
import { FlowNodeRegistry } from "../../../typings";
import { apiGetNodeList } from "../../Layout/Design/Node/api";
import { IBusinessNode, IWorkFlowEntity } from "../../Layout/Design/WorkFlow/entity";
import { INodeEntity } from "../../Node/entity";
import { buildNodeRegistry } from "./nodeRegistry/buildNodeRegistry";

export async function apiGetNodeById  (key_: string) {
   var response = await request.post<INodeEntity>('/node/get', {key_});

  return response;
}

export async function apiGetWorkFlow(flowName: string){
 var response = await request.get<IBusinessNode>(`/api/workflow/get/${flowName}`, {});

  return response;
}

export async function getNodeRegistry(){
  
      const response = await apiGetNodeList()

      const nodeRegistry : FlowNodeRegistry[]  = response.result.map((item) => {
          return buildNodeRegistry(item)
      })

      return nodeRegistry
         
      
}
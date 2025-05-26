import request from "../../../request";

export async function apiGetNodeType  (nodeTypeId: string) {
   var response = await request.post<any>('/nodeType/get', {id: nodeTypeId});

  return response;
}
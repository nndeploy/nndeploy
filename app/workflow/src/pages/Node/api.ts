import { INodeBranchEntity, INodeEntity, INodeTreeNodeEntity} from "./entity";
import request from "../../request";
import { contentType } from "../../request/types";


export async function apiGetNodeTree(){
 var response = await request.get<INodeTreeNodeEntity[]>('/node/tree', {});

  return response;
}

export async function apiGetNodeBranch() {
  var response = await request.post<INodeBranchEntity[]>(
    "/node/branch",
    {}
  );

  return response;
}

export async function apiNodeBranchSave(entity: INodeBranchEntity) {
  var response = await request.post<INodeBranchEntity>(
    "/node/branch/save",
    entity
  );

  return response;
}



export async function apiNodeBranchDelete(id: string) {
  var response = await request.post<any>("/node/branch/delete", { id });

  return response;
}



export async function apiNodeSave(entity: INodeEntity) {
  var response = await request.post<INodeEntity>(
    "/node/save",
    entity
  );

  return response;
}






export async function apiNodeDelete(id: string) {
  var response = await request.post<any>("/node/delete", { id });

  return response;
}


export async function apiGetNodeById(id: string) {
  var response = await request.post<INodeEntity>(
    "/node/get",
    {id}
  );

  return response;
}



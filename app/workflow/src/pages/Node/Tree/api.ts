import { INodeBranchEntity, INodeEntity} from "../entity";
import request from "../../../request";

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



export async function apiGetNodeById(id: string) {
  var response = await request.get<INodeEntity>(
    "/node/get",
    {id}
  );

  return response;
}



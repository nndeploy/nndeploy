import { INodeBranchEntity, INodeEntity} from "./entity";
import request from "../../request";

export async function apiNodeSave(entity: INodeEntity) {
  var response = await request.post<INodeEntity>(
    "/node/save",
    entity
  );

  return response;
}


export async function apiGetNodePage() {
  var response = await request.post<INodeBranchEntity[]>(
    "/node/page",
    {}
  );

  return response;
}




export async function apiNodeDelete(id: string) {
  var response = await request.post<any>("/node/delete", { id });

  return response;
}


export async function apiGetNodeById(id: string) {
  var response = await request.get<INodeEntity>(
    "/node/get",
    {id}
  );

  return response;
}



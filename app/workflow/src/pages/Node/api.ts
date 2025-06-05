import { INodeBranchEntity, INodeEntity} from "./entity";
import request from "../../request";
import { AnyColor } from "@douyinfe/semi-ui/lib/es/colorPicker";
import { contentType } from "../../request/types";

export async function apiNodeSave(entity: INodeEntity) {
  var response = await request.post<INodeEntity>(
    "/node/save",
    entity
  );

  return response;
}


export async function apiGetNodePage(query:any) {
  query = {...query}
  var response = await request.getList<INodeEntity>(
    "/node/page",
    query, 

    {
      method: 'post', 
      headers: contentType.json
    }
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



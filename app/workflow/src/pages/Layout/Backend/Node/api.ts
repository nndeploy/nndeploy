import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";

export async function apiGetNodeTree(){
 var response = await request.post<TreeNodeData[]>('/node/tree', {hobby: 'zhang'});

  return response;
}
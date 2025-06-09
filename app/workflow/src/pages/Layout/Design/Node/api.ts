import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { INodeEntity, INodeTreeNodeEntity } from "../../../Node/entity";

export async function apiGetNodeTree(){
 var response = await request.post<INodeTreeNodeEntity[]>('/node/tree', {hobby: 'zhang'});

  return response;
}

export async function apiGetNodeList(){
 var response = await request.post<INodeEntity[]>('/node/list', {hobby: 'zhang'});

  return response;
}
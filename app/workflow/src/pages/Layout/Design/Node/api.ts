import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import request from "../../../../request";
import { INodeEntity, INodeTreeNodeEntity } from "../../../Node/entity";

export async function apiGetNodeTree(){
 var response = await request.post<INodeTreeNodeEntity[]>('/node/node', {hobby: 'zhang'});

  return response;
}

export async function apiGetNodeList(){
 var response = await request.get<INodeEntity[]>('/api/nodes', {hobby: 'zhang'});

  return response;
}
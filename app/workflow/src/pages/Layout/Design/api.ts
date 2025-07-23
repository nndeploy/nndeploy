import { AxiosRequestConfig } from "axios";
import request from "../../../request";
import { IDagGraphInfo } from "../../Node/entity";

export async function apiGetFiles(url: string) {


  var response = await request.get(url, {});

  return response;
}

export async function apiGetDagInfo(){
 var response = await request.get<IDagGraphInfo>('/api/dag/info', {});

  return response;
}


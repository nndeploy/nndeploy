import { AxiosRequestConfig } from "axios";
import request from "../../../request";

export async function apiGetFiles(url: string) {


  var response = await request.get(url, {});

  return response;
}

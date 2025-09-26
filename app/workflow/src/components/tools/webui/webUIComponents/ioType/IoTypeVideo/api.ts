import request from "../../../../../../request";
import { IServerResourceFileEntity } from "../../../../../Layout/Design/Resource/entity";

export async function apiVideoSave(formData: FormData) {

  const url = '/api/files/upload?file_path=resources/videos';
  var response = await request.upload<any>(url, formData);

  return response;
}


export async function apiGetFileInfo(filePath: string) {
  const url = `/api/files/info`
  const response = await request.get<IServerResourceFileEntity>(url, {
    file_path: filePath
  });
  return response;
}
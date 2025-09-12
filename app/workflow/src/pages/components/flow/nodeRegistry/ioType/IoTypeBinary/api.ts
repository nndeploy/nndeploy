import request from "../../../../../../request";
import { IServerResourceFileEntity } from "../../../../../Layout/Design/Resource/entity";

export async function apiGetFileContent(fileName: string) {

  const url = `/api/preview?file_path=${fileName}`
  const response = await request.get<string>(url,
      {
      //responseType: 'blob'
      returnMimeType: 'text'
    }
  );

  return response

  // return new Promise((resolve, reject) => {
  //   const reader = new FileReader();
  //   reader.onload = () => resolve(reader.result as string);
  //   reader.onerror = reject;
  //   reader.readAsText(response as any); // 转换Blob为文本
  // });
}



export async function apiOtherFileSave(formData: FormData) {

  const url = '/api/files/upload?file_path=resources/others';
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

//IServerResourceFileEntity
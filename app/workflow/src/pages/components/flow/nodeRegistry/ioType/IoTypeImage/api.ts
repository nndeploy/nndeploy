import request from "../../../../../../request";

export async function apiImageSave(formData: FormData) {

  const url = '/api/files/upload?file_path=resources/images';
  var response = await request.upload<any>(url, formData);

  return response;
}
/** Existing Graph declarations identify static model resources. */
export type FileReference = { file_id: string };
export interface ResourceField { path: string[]; value: unknown; directory: boolean }

export function isFileReference(value: unknown): value is FileReference {
  return !!value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).length === 1 && 'file_id' in value
    && typeof value.file_id === 'string' && /^file_[A-Za-z0-9_-]{1,59}$/.test(value.file_id);
}

export function resourceLabel(value: unknown, directory = false): string {
  if (isFileReference(value)) return directory ? 'Uploaded model directory' : 'Uploaded model file';
  if (typeof value === 'string' && value) return value.split(/[\\/]/).pop() || value;
  return 'No file selected';
}

export function getResourceFields(node: any): ResourceField[] {
  if (['Output', 'kNodeTypeOutput'].includes(node?.node_type_)) return [];
  const result: ResourceField[] = [];
  for (const parent of ['', 'param_', 'param']) {
    const target = parent ? node?.[parent] : node;
    const type = target?.io_type_ || node?.io_type_;
    if (!target || !['Model', 'Dir'].includes(type)) continue;
    const fields = target.io_param_?.length ? target.io_param_ : target.io_params_;
    if (!Array.isArray(fields)) continue;
    for (const field of new Set(fields)) {
      if (typeof field !== 'string' || !Object.prototype.hasOwnProperty.call(target, field)
          || (field === 'model_value_' && target.is_path_ === false)) continue;
      result.push({ path: parent ? [parent, field] : [field], value: target[field], directory: type === 'Dir' });
    }
  }
  return result;
}

export function setResourceValue(node: any, path: string[], value: unknown): any {
  const [key, ...rest] = path;
  return { ...node, [key]: rest.length ? setResourceValue(node[key], rest, value) : value };
}

export function resourceFieldName(path: string[]): string {
  return path.map(part => `[${JSON.stringify(part)}]`).join('');
}

export async function uploadResource(file: File, directory: boolean, signal?: AbortSignal): Promise<FileReference> {
  if (directory && !file.name.toLowerCase().endsWith('.zip')) throw new Error('Select a ZIP file for a model directory.');
  const body = new FormData();
  body.append('file', new Blob([file], { type: directory ? 'application/zip' : 'application/octet-stream' }), file.name);
  const response = await fetch('/api/files/upload', { method: 'POST', body, signal });
  const data = await response.json();
  if (!response.ok || data.flag !== 'success') {
    throw new Error(data.error?.message || (typeof data.detail === 'string' ? data.detail : data.message)
      || 'Model upload failed. Please try again.');
  }
  const reference = { file_id: data.result?.file_id };
  if (!isFileReference(reference)) throw new Error('The server returned an invalid file reference.');
  return reference;
}

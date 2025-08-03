/**
 * @author 集团事业-张赵红
 * @description HTTP内容请求头
 */
export const contentType = {
  json: { 'Content-Type': 'application/json;charset=UTF-8' },
  form: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
  multipart: { 'Content-Type': 'multipart/form-data' },
}

/**
 * @description 后端返回的分页信息
 */
export interface IPage {
  total: number
}

// export interface IResponse<
//   Entity,
//   Result extends (IPage & { records: Entity[] }) | Entity = (IPage & { records: Entity[] }) | Entity,
// > {
//   version?: string
//   flag: 'success' | 'error'
//   status: number
//   error: number
//   message: string
//   result: Result
// }

/**
 * @description 通用返回实体-单条记录
 */
export interface IResponse<Entity> {
  version?: string

  /**结果标识 */
  flag: 'success' | 'error'

  /**状态信息 */
  status: number
  /**错误编码 */
  error: number
  /**信息 */
  message: string

  /**实体信息-当条记录 */
  result: Entity
}

/**
 * @description 通用返回实体-多条记录
 */
export interface IResponseList<Entity> {
  version?: string

  /**结果标识 */
  flag: 'success' | 'error'

  /**状态信息 */
  status: number
  /**错误编码 */
  error: number
  /**信息 */
  message: string

  /**实体数组信息-包含后端分页 */
  result: IPage & { records: Entity[] }
}

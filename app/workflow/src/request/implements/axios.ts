/***
 * @author zzh
 * @description 通用请求axios实现版本
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import qs from 'qs'
import lodash, { isArray } from 'lodash'

import { IPage, IResponse, IResponseList, contentType } from '../types'
import { IRequest } from '../IRequest'
//import { message } from 'antd'
import { Toast as message } from '@douyinfe/semi-ui';

export class AxiosRequest implements IRequest {
  defaultConfig: AxiosRequestConfig
  axiosInstance: AxiosInstance

  constructor(options?: AxiosRequestConfig) {
    this.defaultConfig = {
      transformRequest: [
        function (data, headers) {
          try {
            if (!headers['Content-Type'] || headers['Content-Type'] && headers['Content-Type']?.includes('application/x-www-form-urlencoded')) {
              return qs.stringify(data)
            } else if (headers['Content-Type'] && headers['Content-Type']?.includes('multipart/form-data')) {
              return data
            } else {
              return JSON.stringify(data)
            }
          } catch (e) {
            console.error(e)
          }
        },
      ],
      timeout: 1000 * 120, // 默认超时时间，120秒
    }
    this.axiosInstance = axios.create({ ...this.defaultConfig, ...options })

    this.axiosInstance.interceptors.response.use(
      (response) => {
        if (response.config.responseType == 'blob') {
          return response
        }
        const businessData = response.data

        const { flag } = businessData

        var error = businessData.error ? businessData.error : businessData.msg ? businessData.msg : ''

        if (flag && flag == 'error' && businessData.statusCode && businessData.statusCode == '403') {
          message.warning('您的登录已失效, 正在退出, 请稍后!')
          setTimeout(function () {
            top!.window.location.href = '/page/login/index.html' + '?' + new Date().getTime()
          }, 3000)
        }

        // 账户禁用
        if (businessData.errorCode && businessData.errorCode == 'JR-300005') {
          message.warning('您的账户已被禁用或删除，系统将退出，请联系管理员！')

          setTimeout(function () {
            top!.window.location.href = '/page/login/index.html' + '?' + new Date().getTime()
          }, 3000)
        }

        if (flag == 'error') {
          message.warning(error)
        }

        if (!businessData.result) {
          businessData.result = {}
        }

        if (isArray(businessData.data)) {
          businessData.result.records = businessData.data
          delete businessData.data
        }

        if (isArray(businessData.jsonArray)) { //jreap组织机构树查询结果使用这个字段
          //debugger;
          businessData.result.records = businessData.jsonArray
          delete businessData.jsonArray
        }

        if (Reflect.has(businessData, 'rows')) {
          businessData.result.records = businessData.rows
          delete businessData.rows
        }
        if (Reflect.has(businessData, 'totalrecords')) {
          ; (businessData.result as IPage).total = parseInt(businessData.totalrecords)
        }
        return response
      },
      (err) => {
        let config = err.config
        if (!config || !config.retry) {
          return new Promise((resolve, reject) => {
            resolve({
              flag: 'error',
              msg: err,
            })
          })
        }
        config.__retryCount = config.__retryCount || 0
        if (config.__retryCount >= config.retry) {
          return new Promise((resolve, reject) => {
            resolve({
              msg: '响应超时',
              flag: 'error',
              success: false,
            })
          })
        }
        config.__retryCount += 1
        var backoff = new Promise<void>((resolve) => {
          setTimeout(() => {
            resolve()
          }, config.retryDelay || 1)
        })
        return backoff.then(() => {
          return this.axiosInstance(config)
        })
      },
    )
  }
  async getList<Entity, BusinessResponse extends IResponseList<Entity> = IResponseList<Entity>>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig,
  ) {
    var method = config?.method ? config?.method : 'GET'

    var tempConfig: AxiosRequestConfig | undefined = config

    if (method == 'GET') {
      tempConfig = {
        method,
        headers: contentType.form,
        ...config,
        params: data,
        url,
      }
    } else {
      tempConfig = {
        method,
        headers: contentType.form,
        ...config,
        data,
        url,
      }
    }
    var response = await this.axiosInstance.request<BusinessResponse>(tempConfig)
    return response.data
  }

  async get<Entity, BusinessResponse extends IResponse<Entity> = IResponse<Entity>>(
    url: string,
    data: any,
    config?: AxiosRequestConfig,
  ) {
    var method = config?.method ? config?.method : 'GET'

    var tempConfig: AxiosRequestConfig | undefined = config

    if (method == 'GET') {
      tempConfig = {
        method,
        headers: contentType.form,
        ...config,
        params: data,
        url,
      }
    } else {
      tempConfig = {
        method,
        headers: contentType.form,
        ...config,
        data,
        url,
      }
    }
    var response = await this.axiosInstance.request<BusinessResponse>(tempConfig)
    return response.data
  }
  async post<Entity, BusinessResponse extends IResponse<Entity> = IResponse<Entity>>(
    url: string,
    data?: any,
    config?: AxiosRequestConfig,
  ) {
    var response = await this.axiosInstance.post<BusinessResponse>(url, data, {
      headers: contentType.json,
      ...config,
    })
    return response.data
  }

  async postJ<Entity, BusinessResponse extends IResponse<Entity> = IResponse<Entity>>(
    url: string,
    data: any,
    config?: AxiosRequestConfig,
  ) {
    var response = await this.axiosInstance.post<BusinessResponse>(url, data, {
      headers: contentType.json,
      ...config,
    })
    return response.data
  }

  async upload<Entity, BusinessResponse extends IResponse<Entity> = IResponse<Entity>>(
    url: string,
    data: any,
    config?: AxiosRequestConfig,
  ) {
    var response = await this.axiosInstance.post<BusinessResponse>(url, data, {
      headers: contentType.multipart,
      ...config,
    })
    return response.data
  }

  async download(url: string, params: any, config?: AxiosRequestConfig, fileName?: string) {

    var method: AxiosRequestConfig['method'] = 'GET'



    if (config && config.method == 'POST') {
      method = 'POST'
    }

    if (method == 'GET') {
      var tempConfig: AxiosRequestConfig = {
        url,
        method,

        params,

        headers: contentType.form,
        responseType: 'blob',
        ...config,
      }
    } else {
      var tempConfig: AxiosRequestConfig = {
        url,
        method,
        data: params,

        headers: contentType.form,
        responseType: 'blob',
        ...config,
      }
    }



    var response = await this.axiosInstance.request<any>(tempConfig)

    const { data, headers } = response
    //   var fileName:string = headers['content-disposition'].replace(/\w+;filename=(.*)/, '$1')
    //   var filename2 = decodeURI(fileName);

    fileName = fileName ?? response.headers['content-disposition'].split(';')[1].split('=')[1]

    //fileName = fileName.substring("utf-8''".length);

    function trim(str: string, char: string): string {
      if (char === '^' || char === '$' || char === '\\' || char === '.' || char === '*' || char === '+' || char === '?' || char === '(' || char === ')' || char === '[' || char === ']' || char === '{' || char === '}' || char === '|') {
        char = '\\' + char; // 转义正则表达式中的特殊字符
      }
      return str.replace(new RegExp(`^${char}+|${char}+$`, 'g'), '');
    }
    fileName = trim(fileName!, "\"")

    //fileName = fileName.trim("_")

    // 此处当返回json文件时需要先对data进行JSON.stringify处理，其他类型文件不用做处理
    //const blob = new Blob([JSON.stringify(data)], ...)
    const blob = new Blob([data], { type: headers['content-type'] })
    var dom = document.createElement('a')
    var href = window.URL.createObjectURL(blob)
    dom.href = href
    dom.download = fileName
    dom.style.display = 'none'
    document.body.appendChild(dom)
    dom.click()
    // dom.parentNode.removeChild(dom)
    window.URL.revokeObjectURL(url)
  }
}

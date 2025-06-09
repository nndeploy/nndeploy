/***
 * @author 集团事业-张赵红
 * @description 通用请求接口
 */


import { AxiosRequestConfig } from "axios";
import { IResponse, IResponseList } from "./types";

/**
 * @description 通用请求接口
 */
export interface IRequest {
    /**
     * 
     * @param url url地址
     * @param data  请求数据
     * @param config  axios请求配置对象
     */
    get<Entity, BusinessResponse extends IResponse<Entity> = IResponse<Entity>>(url:string ,data?:any,  config?:AxiosRequestConfig):Promise<BusinessResponse>
    getList<Entity, BusinessResponse extends IResponseList<Entity> = IResponseList<Entity>>(url:string, data?:any,  config?:AxiosRequestConfig):Promise<BusinessResponse>
    post<Entity, BusinessResponse extends IResponse<Entity> = IResponse<Entity>>(url:string, data:any, config?:AxiosRequestConfig):Promise<BusinessResponse>; 

}
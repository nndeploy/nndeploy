
export const INIT_JSON:string = 'INIT_JSON'  

export function initJson(payload:any){
    return {
        type: INIT_JSON, 
        payload
    }
}




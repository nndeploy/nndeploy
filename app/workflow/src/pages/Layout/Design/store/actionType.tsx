export const INIT_DAG_GRAPH_INFO:string = 'INIT_DAG_GRAPH_INFO'  

export function initDagGraphInfo(payload:any){

    //debugger;
    return {
        type: INIT_DAG_GRAPH_INFO, 
        payload
    }
}


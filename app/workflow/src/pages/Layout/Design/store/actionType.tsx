export const INIT_DAG_GRAPH_INFO:string = 'INIT_DAG_GRAPH_INFO'  

export function initDagGraphInfo(payload:any){

    //debugger;
    return {
        type: INIT_DAG_GRAPH_INFO, 
        payload
    }
}


export const INIT_FRESH_FLOW_TREE:string = 'INIT_FRESH_FLOW_TREE'  

export function initFreshFlowTree(payload:any){

    //debugger;
    return {
        type: INIT_FRESH_FLOW_TREE, 
        payload
    }
}



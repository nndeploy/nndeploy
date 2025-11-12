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



export const INIT_FRESH_RESOURCE_TREE:string = 'INIT_FRESH_RESOURCE_TREE'  

export function initFreshResourceTree(payload:any){

    //debugger;
    return {
        type: INIT_FRESH_RESOURCE_TREE, 
        payload
    }
}


export const INIT_RESOURCE_DIR:string = 'INIT_RESOURCE_DIR'  

export function initResourceDir(payload:any){

    //debugger;
    return {
        type: INIT_RESOURCE_DIR, 
        payload
    }
}




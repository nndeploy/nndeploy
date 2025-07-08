
export const INIT_ARTICLE_DIGETSTS:string = 'INIT_ARTICLE_DIGETSTS'  
export function initArticleDigests(payload:any){
    return {
        type: INIT_ARTICLE_DIGETSTS, 
        payload
    }
}


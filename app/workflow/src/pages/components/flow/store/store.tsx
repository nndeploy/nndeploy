import * as React from 'react';
import { INIT_ARTICLE_DIGETSTS } from './actionType'

interface state {
  
    articleDigests: any[],
  

}

export const initialState: state = {
  
    articleDigests: []
}

type ContextType = {
    state: state
    dispatch?: any
}

const store = React.createContext<ContextType>({ state: initialState });


export function reducer(state: state, action: any): state {

    const { payload } = action

    //console.log('action-type', action.type, 'payload', payload, 'state', state);
   
    switch (action.type) {
       
        case INIT_ARTICLE_DIGETSTS:
            //debugger;
            return { ...state, ...payload }

        default:
            throw new Error();
    }
}

export default store
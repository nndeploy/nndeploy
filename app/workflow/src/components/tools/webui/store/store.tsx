import * as React from 'react';
import { INIT_JSON, } from './actionType'
import { FlowDocumentJSON, } from '../../../../typings';

interface state {
  json: FlowDocumentJSON,

}

export const initialState: state = {
  json: {
    nodes: [],
    edges: [],
  }

}

type ContextType = {
  state: state
  dispatch?: any
}

const store = React.createContext<ContextType>({ state: initialState });


export function reducer(state: state, action: any): state {

  const json: FlowDocumentJSON = action.payload

  switch (action.type) {

    case INIT_JSON:

      return { json }


    default:
      throw new Error();
  }
}

export default store
import React from 'react';
import { FlowDocumentJSON } from '../typings';

export const FlowEnviromentContext = React.createContext<
  {
    element?: React.MutableRefObject<HTMLDivElement | null>;
    onSave?: (flowJson: FlowDocumentJSON) => void;
  }

>({});

export function useFlowEnviromentContext(){
  const context = React.useContext(FlowEnviromentContext);
  return context;
}


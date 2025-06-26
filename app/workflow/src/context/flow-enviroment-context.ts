import React from 'react';
import { FlowDocumentJSON } from '../typings';
import { INodeEntity } from '../pages/Node/entity';

export const FlowEnviromentContext = React.createContext<
  {
    nodeList?: INodeEntity[], 
    element?: React.MutableRefObject<HTMLDivElement | null>;
    onSave?: (flowJson: FlowDocumentJSON) => void;
    onRun?: (flowJson: FlowDocumentJSON) => void;
  }

>({});

export function useFlowEnviromentContext(){
  const context = React.useContext(FlowEnviromentContext);
  return context;
}


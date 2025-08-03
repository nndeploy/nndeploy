import React from 'react';
import { FlowDocumentJSON } from '../typings';
import { INodeEntity } from '../pages/Node/entity';
import { IBusinessNode, IParamTypes } from '../pages/Layout/Design/WorkFlow/entity';
import { IFlowNodesRunningStatus, IOutputResource } from '../pages/components/flow/entity';

export const FlowEnviromentContext = React.createContext<
  {
    nodeList?: INodeEntity[],
    element?: React.MutableRefObject<HTMLDivElement | null>;
    onSave?: (flowJson: FlowDocumentJSON) => void;
    onRun?: (flowJson: FlowDocumentJSON) => void;
    onConfig: ()=>void; 
    graphTopNode: IBusinessNode;
    paramTypes: IParamTypes;
    outputResources: IOutputResource;
    flowNodesRunningStatus: IFlowNodesRunningStatus
  }

>({
  outputResources: {path: [], text: []}, 
  flowNodesRunningStatus: {}, 
  graphTopNode: {}
} as any);

export function useFlowEnviromentContext() {
  const context = React.useContext(FlowEnviromentContext);
  return context;
}


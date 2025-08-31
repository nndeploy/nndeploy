import React from 'react';
import { FlowDocumentJSON } from '../typings';
import { INodeEntity } from '../pages/Node/entity';
import { IBusinessNode, IParamTypes } from '../pages/Layout/Design/WorkFlow/entity';
import { IFlowNodesRunningStatus, ILog, IOutputResource } from '../pages/components/flow/entity';
import { run } from 'node:test';

export const FlowEnviromentContext = React.createContext<
  {
    nodeList?: INodeEntity[],
    element?: React.MutableRefObject<HTMLDivElement | null>;
    onSave?: (flowJson: FlowDocumentJSON) => void;
    onRun?: (flowJson: FlowDocumentJSON) => void;
    onDownload?: (flowJson: FlowDocumentJSON) => void;
    downloading: boolean, 
    onConfig: ()=>void; 
    graphTopNode: IBusinessNode;
    paramTypes: IParamTypes;
    outputResource: IOutputResource;
    flowNodesRunningStatus: IFlowNodesRunningStatus, 
    log: ILog, 
    runResult: string; 
    downloadModalVisible:boolean, 
    setDownloadModalVisible:React.Dispatch<React.SetStateAction<boolean>>,
    downloadModalList:string[]
  }

>({
  outputResources: {path: [], text: []}, 
  flowNodesRunningStatus: {}, 
  downloading: false, 
  graphTopNode: {},
  log: {
    items: [],
    time_profile: {
      init_time: undefined,
      run_time: undefined
    }
  },
  runResult: '',
  downloadModalVisible: false, 
  setDownloadModalList: []

} as any);

export function useFlowEnviromentContext() {
  const context = React.useContext(FlowEnviromentContext);
  return context;
}


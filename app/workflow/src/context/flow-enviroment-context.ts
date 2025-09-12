import React from 'react';
import { FlowDocumentJSON } from '../typings';
import { INodeEntity } from '../pages/Node/entity';
import { IBusinessNode, IParamTypes } from '../pages/Layout/Design/WorkFlow/entity';
import { IFlowNodesRunningStatus, ILog, IOutputResource, IRunInfo } from '../pages/components/flow/entity';
import { run } from 'node:test';

export const FlowEnviromentContext = React.createContext<
  {
    nodeList?: INodeEntity[],
    element?: React.MutableRefObject<HTMLDivElement | null>;
    onSave?: (flowJson: FlowDocumentJSON) => void;
    onRun?: (flowJson: FlowDocumentJSON) => void;
    onDownload?: (flowJson: FlowDocumentJSON) => void;
    downloading: boolean,
    onConfig: () => void;
    graphTopNode: IBusinessNode;
    paramTypes: IParamTypes;


    // isRunning: boolean, 
    // runResult: string; 
    // runningTaskId: string, 
    // outputResource: IOutputResource;
    // flowNodesRunningStatus: IFlowNodesRunningStatus, 
    // log: ILog, 
    runInfo: IRunInfo

    downloadModalVisible: boolean,
    setDownloadModalVisible: React.Dispatch<React.SetStateAction<boolean>>,
    downloadModalList: string[],
  }


>({

  flowNodesRunningStatus: {},
  downloading: false,
  graphTopNode: {},

  // isRunning:false, 
  // runningTaskId: '', 
  // runResult: '',
  // outputResources: {path: [], text: []}, 
  // log: {
  //   items: [],
  //   time_profile: {
  //     init_time: undefined,
  //     run_time: undefined
  //   }
  // },
  runInfo: {
    isRunning: false,
    result: '',
    runningTaskId: '',
    log: {
      items: [],
      time_profile: {
        init_time: undefined,
        run_time: undefined

      }
    },
    outputResource: {
      type: 'memory',
      content: {},
      time: Date.now()
    },
    flowNodesRunningStatus: {}
  }, 

  downloadModalVisible: false,
  setDownloadModalList: [],


} as any);

export function useFlowEnviromentContext() {
  const context = React.useContext(FlowEnviromentContext);
  return context;
}


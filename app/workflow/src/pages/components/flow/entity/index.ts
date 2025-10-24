import { WorkflowNodeStatus } from "../../../../components/base-node/node-status-bar/type";

export interface IFlowNodeItemRunningStatus {
  time: number;
  status: WorkflowNodeStatus // "IDLE" | "INITING" | "INITED" | "RUNNING" | "DONE" |"ERROR"
}

export interface IFlowNodesRunningStatus {
  [nodeName: string]: IFlowNodeItemRunningStatus
}


export interface IOutputResource {
  // path: { name: string, path: string }[],
  // text: { name: string, text: string }[]
  type: 'memory',
  content: {
    [nodeName: string]: string
  }
  time: number

}


export interface ILog {
  items: string[],
  time_profile: {
    init_time?: number,
    run_time?: number
  }

}

export interface IDownloadProgress {
  filename: string,
  percent: number,
  downloaded: number,
  elapsed: number,
  total: number
}
export interface IRunInfo {
  isRunning: boolean
  result: '' | 'error' | 'success',
  runningTaskId: string;
  downloadProgress: {
    [fileName: string]: IDownloadProgress
  }
  log: ILog;
  outputResource: IOutputResource,
  flowNodesRunningStatus: IFlowNodesRunningStatus,
  time: number,

}

export interface ILineEntity {

  oldFrom?: string;
  oldFrom_name?: string;

  from: string,
  from_name:string; 

  oldFromPort?: string | number,
  fromPort: string | number,


  oldTo?: string;
   oldTo_name?: string;
  to: string,
  to_name:string; 

  oldToPort?: string | number,
  toPort: string | number,

  type_: string,
  desc_: string,
  
 // [key:string]:any
}


export interface IExpandInfo {
  //expanded: boolean,

  inputLines: ILineEntity[],
  outputLines: ILineEntity[]
}

export interface INodeUiExtraInfo {
  expanded?: boolean,
  position?: { x: number, y: number },
  size?: { width: number, height: number },
  children?: {[nodeName:string]: INodeUiExtraInfo}

  // inputLines?: ILineEntity[],
  // outputLines?: ILineEntity[]
}
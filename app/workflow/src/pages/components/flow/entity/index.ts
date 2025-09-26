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
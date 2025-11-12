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

  originFrom?: string, //origin node ID
  originFrom_name?: string, //origin node name : for debug
  originFromPort?: string | number,//origin node port

  oldFrom?: string; //old from node ID
  oldFrom_name?: string; //old from node name

  from: string,  //from node ID, when collapse subcavas node, outputline connnect to other node through subcavas
  from_name: string; //from node name

  oldFromPort?: string | number, //old from port ID
  fromPort: string | number, //from port ID, when collapse subcavas node, outputline connnect to other node through subcavas dynamic output port


  oldTo?: string;  //old to node ID
  oldTo_name?: string; //old to node name
  
  to: string,  //when collapse subcavas node, inputline reconnect to subcavas 
  to_name: string; //to node name

  oldToPort?: string | number, //old to port ID
  toPort: string | number, //to port ID, when collapse subcavas node, inputline reconnect to subcavas input dynamic port


  originTo?: string,
  originTo_name?: string;
  originToPort?: string | number,

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
  children?: { [nodeName: string]: INodeUiExtraInfo }

  // inputLines?: ILineEntity[],
  // outputLines?: ILineEntity[]
}
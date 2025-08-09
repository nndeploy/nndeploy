export interface IFlowNodeItemRunningStatus {
  time: number;
  status: "IDLE" | "INITING" | "INITED" | "RUNNING" | "DONE"
}

export interface IFlowNodesRunningStatus {
  [nodeName: string]: IFlowNodeItemRunningStatus
}


export interface IOutputResource {
  path: { name: string, path: string }[],
  text: { name: string, text: string }[]
}


export interface ILog {
  items: string[],
  time_profile: {
    init_time?: number,
    run_time?: number
  }
}
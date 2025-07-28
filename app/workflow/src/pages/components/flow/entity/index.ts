export interface IFlowNodeItemRunningStatus {
  time: number;
  status: "IDLE" | "INITING" | "INITED" | "RUNNING" | "DONE"
}

export interface IFlowNodesRunningStatus {
  [nodeName: string]: IFlowNodeItemRunningStatus
}


export interface IOutputResource {
  path: { name: string }[],
  text: { name: string, text: string }[]
}
export interface IFlowNodeItemRunningStatus{
  time: number; 
  status: "IDLE" |  "PENDING"| "RUNNING" | "DONE"
}

export interface  IFlowNodesRunningStatus{
  [nodeName:string]: IFlowNodeItemRunningStatus
}
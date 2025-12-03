import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { FlowDocumentJSON } from "../../../../typings";
import { INodeUiExtraInfo } from "../../../components/flow/entity";

export interface IWorkFlowTreeNodeEntity {
  id: string;
  name: string;
  parentId: string;
  type: 'branch' | 'leaf'
  //isLeaf: boolean;

}

export interface IWorkFlowBranchEntity {
  id: string;
  name: string;
  parentId: string;

}

export interface ITreeWorkFlowResponseData {
  fileNames: string[],
  workflows: IBusinessNode[]
}

export interface Inndeploy_ui_layout {
  // position?: { x: number, y: number },
  // size?: { width: number, height: number },
  layout: {
    [nodeName: string]: {
      //config: INodeUiExtraInfo, 
      expanded?: boolean,
      position?: { x: number, y: number },
      size?: { width: number, height: number },
      children?: { [nodeName: string]: INodeUiExtraInfo }


    }
  },
  //nodeExtra: { [nodeName: string]: INodeUiExtraInfo }
  groups: { name_: string, blockIDs: string[], parentID: string,  }[]
}

export interface IConnectinPoint {
  id?: string,
  name_: string,
  type_: string,
  desc_: string
}
export interface IBusinessNode {
  key_: string;
  name_: string;
  desc_: string;
  device_type_: string;
  inputs_: IConnectinPoint[],
  outputs_: IConnectinPoint[],
  node_repository_?: IBusinessNode[],
  [key: string]: any;
  nndeploy_ui_layout?: Inndeploy_ui_layout
}

export interface IWorkFlowEntity {
  id: string;
  name: string;
  parentId: string;
  designContent: FlowDocumentJSON,
  businessContent: IBusinessNode,


}

export interface IWorkFlowRunResult {
  task_id: string;
}

export interface WorkFlowTreeNodeData extends TreeNodeData {
  //type: "leaf" | "branch";
  entity: IWorkFlowTreeNodeEntity,
  children?: WorkFlowTreeNodeData[];
}

export interface IParamTypes {
  [key: string]: string[]
}

export interface IFieldType {
  isArray: boolean,
  componentType?: string;
  primateType: string;
  selectOptions?: string[];
  selectKey?: string;
  originValue: any
}


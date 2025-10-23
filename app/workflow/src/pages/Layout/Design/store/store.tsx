import * as React from 'react';
import { INIT_DAG_GRAPH_INFO, INIT_FRESH_FLOW_TREE, INIT_FRESH_RESOURCE_TREE, } from './actionType'
import { IDagGraphInfo, INodeEntity, INodeTreeNodeEntity, NodeTreeNodeData } from '../../../Node/entity';
import { FlowNodeRegistry } from '../../../../typings';
import { buildNodeRegistry } from '../../../components/flow/nodeRegistry/buildNodeRegistry';
import { CommentNodeRegistry } from '../../../../nodes/comment';
import { GroupNodeRegistry } from '../../../../nodes/group';

interface state {
  dagGraphInfo: IDagGraphInfo,
  nodeList: INodeEntity[],
  nodeRegistries: FlowNodeRegistry[],
  treeData: NodeTreeNodeData[],
  freshFlowTreeCnt: number,
  freshResourceTreeCnt: number,
}

export const initialState: state = {
  dagGraphInfo: {
    graph: {
      key_: "",
      name_: "",
      desc_: "", 
      device_type_: "",
      inputs_: [],
      outputs_: [],
      is_external_stream_: false,
      is_inner_: false,
      is_time_profile_: false,
      is_debug_: false,
      is_graph_node_share_stream_: true,
      queue_max_size_: 16,
      node_repository_: [],
    },
    accepted_edge_types: {},
    nodes: []

  },
  nodeList: [],
  nodeRegistries: [],
  treeData: [],
  freshFlowTreeCnt: 0,
  freshResourceTreeCnt: 0

}

type ContextType = {
  state: state
  dispatch?: any
}

const store = React.createContext<ContextType>({ state: initialState });

function buildTreeFromArray(
  data: INodeTreeNodeEntity[],
  parentId: string = ""
): NodeTreeNodeData[] {
  return data
    .filter((item) => {
      if (item.parentId == parentId) {
        return true;
      }
      return false;
    })
    .map((item) => {
      const children = buildTreeFromArray(data, item.id);

      return {
        key: item.id,
        label: item.name,

        nodeEntity: item,
        children: children.length > 0 ? children : undefined,
      };
    });
}


export function reducer(state: state, action: any): state {

  const dagGraphInfo: IDagGraphInfo = action.payload

  switch (action.type) {
    case INIT_DAG_GRAPH_INFO: {



      const leafNodes = dagGraphInfo.nodes.filter((item) => {
        return item.type == 'leaf'
      }).map(item => {
        return item.nodeEntity! // 这里的nodeEntity是INodeEntity
      })

      let nodeRegistries: FlowNodeRegistry[] = leafNodes.map((item) => {
        return buildNodeRegistry(item)
      })

      nodeRegistries.push(CommentNodeRegistry)
       nodeRegistries.push(GroupNodeRegistry)
       


      const treeData = buildTreeFromArray(dagGraphInfo.nodes)

      const temp: state = { ...state, dagGraphInfo, nodeList: [dagGraphInfo.graph, ...leafNodes], nodeRegistries, treeData }
      return temp
    }


    case INIT_FRESH_FLOW_TREE: {


      const temp: state = { ...state, freshFlowTreeCnt: state.freshFlowTreeCnt + 1 }
      return temp
    }
    case INIT_FRESH_RESOURCE_TREE: {
      const temp: state = { ...state, freshResourceTreeCnt: state.freshResourceTreeCnt + 1 }
      return temp
    }
    default:
      throw new Error();
  }
}

export default store
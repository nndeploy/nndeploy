import { FunctionComponent, useContext, useState } from 'react';

import { FlowNodeEntity, SelectorBoxPopoverProps, useClientContext, WorkflowEdgeJSON, WorkflowNodeLinesData } from '@flowgram.ai/free-layout-editor';
import { WorkflowGroupCommand } from '@flowgram.ai/free-group-plugin';
import { Button, ButtonGroup, SideSheet, Toast, Tooltip } from '@douyinfe/semi-ui';
import { IconCopy, IconDeleteStroked, IconExpand, IconSave, IconShrink } from '@douyinfe/semi-icons';

import { IconGroup } from '../group';
import { FlowCommandId } from '../../shortcuts/constants';
import {  } from '../../pages/components/flow/FlowSaveDrawer/functions';
import { FlowDocumentJSON, FlowNodeJSON } from '../../typings';
import FlowSaveDrawer from '../../pages/components/flow/FlowSaveDrawer';
import { IWorkFlowEntity } from '../../pages/Layout/Design/WorkFlow/entity';
import store from '../../pages/Layout/Design/store/store';
import { initFreshFlowTree } from '../../pages/Layout/Design/store/actionType';
import { useFlowEnviromentContext } from '../../context/flow-enviroment-context';
import { apiWorkFlowSave } from '../../pages/Layout/Design/WorkFlow/api';
import { designDataToBusinessData, getEdgeToNameMaps } from '../../pages/components/flow/FlowSaveDrawer/toBusiness';
import { EnumFlowType } from '../../enum';

const BUTTON_HEIGHT = 24;

export const SelectorBoxPopover: FunctionComponent<SelectorBoxPopoverProps> = ({
  bounds,
  children,
  flowSelectConfig,
  commandRegistry,
}) => {

  const ctx = useClientContext()

  let allNodes = ctx.document.toJSON().nodes

  const flowEnviroment = useFlowEnviromentContext();


  const { state, dispatch } = useContext(store)



  const [entity, setEntity] = useState<IWorkFlowEntity>({
    id: "",
    name: "",
    parentId: "",
    designContent: {
      nodes: [],
      edges: [],
    },
    businessContent: {
      key_: "nndeploy::dag::Graph",
      name_: "demo",
      desc_: "", 
      device_type_: "kDeviceTypeCodeX86:0",
      inputs_: [],
      outputs_: [
        {
          name_: "detect_out",
          type_: "kNotSet",
          desc_: ""
        },
      ],
      is_external_stream_: false,
      is_inner_: false,
      is_time_profile_: true,
      is_debug_: false,
      is_graph_node_share_stream_: true,
      queue_max_size_: 16,
      node_repository_: [],
    },
  });

  const [saveDrawerVisible, setSaveDrawerVisible] = useState(false);

  function handleSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }

  function onflowSaveDrawrSure(entity: IWorkFlowEntity) {

    dispatch(initFreshFlowTree({}))
    setSaveDrawerVisible(false);

  }
  function onFlowSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }


  // function buildDesignData(selectedNodes: FlowNodeEntity[]) {

  //   function buildSelectedNodes() {
  //     const nodes = selectedNodes.map(node => {
  //       return node.toJSON()
  //     })

  //     return nodes
  //   }

  //   function buildEdges() {

  //     let edges: WorkflowEdgeJSON[] = []


  //     selectedNodes.map(node => {
  //       let inputLines = node.getData(WorkflowNodeLinesData).inputLines

  //       inputLines.map(line => {
  //         // if (!nodeIds.includes(line.from.id)) {
  //         //   return
  //         // }

  //         let edge = {
  //           sourceNodeID: line.from!.id,
  //           targetNodeID: line.to!.id,
  //           sourcePortID: line.fromPort!.portID,
  //           targetPortID: line.toPort!.portID
  //         }

  //         if (edges.find(item => {
  //           return item.sourceNodeID == edge.sourceNodeID &&
  //             item.targetNodeID == edge.targetNodeID &&
  //             item.sourcePortID == edge.sourcePortID &&
  //             item.targetPortID == edge.targetPortID
  //         })) {
  //           return
  //         }
  //         edges.push(edge)
  //       })

  //       // 输出线条
  //       let outputLines = node.getData(WorkflowNodeLinesData).outputLines
  //       outputLines.map(line => {
  //         // if (!nodeIds.includes(line.to!.id)) {
  //         //   return
  //         // }

  //         let edge = {
  //           sourceNodeID: line.from!.id,
  //           targetNodeID: line.to!.id,
  //           sourcePortID: line.fromPort!.portID,
  //           targetPortID: line.toPort!.portID
  //         }

  //         if (edges.find(item => {
  //           return item.sourceNodeID == edge.sourceNodeID &&
  //             item.targetNodeID == edge.targetNodeID &&
  //             item.sourcePortID == edge.sourcePortID &&
  //             item.targetPortID == edge.targetPortID
  //         })) {
  //           return
  //         }
  //         edges.push(edge)
  //       })

  //     })

  //     return edges
  //   }

  //   ///@ts-ignore
  //   let nodes: FlowNodeJSON[] = buildSelectedNodes()


  //   let edges = buildEdges()

  //   return {
  //     nodes,
  //     edges
  //   }

  // }

  // async function onSubGraphSave() {
  //   var selectedNodes = flowSelectConfig.selectedNodes
  //   // selectedNodes.map(node => {
  //   //   let inputLines = node.getData(WorkflowNodeLinesData).inputLines
  //   //   // 输出线条
  //   //   let outputLines = node.getData(WorkflowNodeLinesData).outputLines


  //   //   let json1 = node.toJSON()

  //   //   let jsonData = node.getJSONData()
  //   //   let j = 0;
  //   // })

  //   // var json = flowSelectConfig.toJSON()
  //   // var i = 0;

  //   let designContent: FlowDocumentJSON = buildDesignData(selectedNodes)

  //   let businessContent = designDataToBusinessData(designContent, flowEnviroment.graphTopNode, allNodes as any,  ctx)

  //   let edgeMaps = getEdgeToNameMaps(allNodes as any, designContent.edges)

  //   let selectedNodeIds = selectedNodes.map(node => {
  //     return node.id
  //   })

  //   let subFlowInputEdges = designContent.edges.filter(edge => !selectedNodeIds.includes(edge.sourceNodeID))

  //   let inputs_ = subFlowInputEdges.map(edge => {

  //     let soureNode = allNodes.find(item => item.id == edge.sourceNodeID)!

  //     let outputs = soureNode.data.outputs_ ?? []
  //     let output = outputs.find((item : any) => item.id == edge.sourcePortID)

  //     let name_ = edgeMaps[edge.sourceNodeID + "@" + edge.sourcePortID]
  //     return {
  //       ...output,
  //       name_
  //     }

  //   })


  //   let subFlowOutputEdges = designContent.edges.filter(edge => !selectedNodeIds.includes(edge.targetNodeID))

  //   let outputs_ = subFlowOutputEdges.map(edge => {

  //     let outputNode = allNodes.find(item => item.id == edge.targetNodeID)!

  //     let inputs = outputNode.data.inputs_ ?? []
  //     let input = inputs.find((item : any) => item.id == edge.targetPortID)

  //     let name_ = edgeMaps[edge.sourceNodeID + "@" + edge.sourcePortID]
  //     return {
  //       ...input,
  //       name_
  //     }

  //   })

  //   businessContent.inputs_ = inputs_
  //   businessContent.outputs_ = outputs_

  //   let temp = businessContent
  //   let i = 0;

  //   const response = await apiWorkFlowSave("", businessContent);
  //   if (response.flag == "success") {
  //     Toast.success('save subflow successed')
  //   }else{
  //     Toast.error('save subflow failed')
  //   }
  // }

  return <>
    <div
      style={{
        position: 'absolute',
        left: bounds.right,
        top: bounds.top,
        transform: 'translate(-100%, -100%)',
      }}
      onMouseDown={(e) => {
        e.stopPropagation();
      }}
    >
      <ButtonGroup
        size="small"
        style={{ display: 'flex', flexWrap: 'nowrap', height: BUTTON_HEIGHT }}
      >

        <Tooltip content={'Create Group'}>
          <Button
            icon={<IconGroup size={14} />}
            style={{ height: BUTTON_HEIGHT }}
            type="primary"
            theme="solid"
            onClick={() => {
              commandRegistry.executeCommand(WorkflowGroupCommand.Group);
            }}
          />
        </Tooltip>

        <Tooltip content={'Delete'}>
          <Button
            type="primary"
            theme="solid"
            icon={<IconDeleteStroked />}
            style={{ height: BUTTON_HEIGHT }}
            onClick={() => {
              commandRegistry.executeCommand(FlowCommandId.DELETE);
            }}
          />
        </Tooltip>
      </ButtonGroup>
    </div>
    <div>{children}</div>
    <SideSheet
      width={"30%"}
      visible={saveDrawerVisible}
      onCancel={handleSaveDrawerClose}
      title={"save flow"}
    >
      <FlowSaveDrawer
        entity={entity!}
        flowType={EnumFlowType.workspace}
        onSure={onflowSaveDrawrSure}
        onClose={onFlowSaveDrawerClose}
      />
    </SideSheet>
  </ >
}

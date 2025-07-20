import { FunctionComponent, useState } from 'react';

import { FlowNodeEntity, SelectorBoxPopoverProps, useClientContext, WorkflowEdgeJSON, WorkflowNodeLinesData } from '@flowgram.ai/free-layout-editor';
import { WorkflowGroupCommand } from '@flowgram.ai/free-group-plugin';
import { Button, ButtonGroup, SideSheet, Tooltip } from '@douyinfe/semi-ui';
import { IconCopy, IconDeleteStroked, IconExpand, IconSave, IconShrink } from '@douyinfe/semi-icons';

import { IconGroup } from '../group';
import { FlowCommandId } from '../../shortcuts/constants';
import { designDataToBusinessData } from '../../pages/components/flow/FlowSaveDrawer/functions';
import { FlowDocumentJSON, FlowNodeJSON } from '../../typings';
import FlowSaveDrawer from '../../pages/components/flow/FlowSaveDrawer';
import { IWorkFlowEntity } from '../../pages/Layout/Design/WorkFlow/entity';

const BUTTON_HEIGHT = 24;

export const SelectorBoxPopover: FunctionComponent<SelectorBoxPopoverProps> = ({
  bounds,
  children,
  flowSelectConfig,
  commandRegistry,
}) => {

  const ctx = useClientContext()

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
        device_type_: "kDeviceTypeCodeX86:0",
        inputs_: [],
        outputs_: [
          {
            name_: "detect_out",
            type_: "kNotSet",
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
    setSaveDrawerVisible(false);

  }
  function onFlowSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }


  function buildDesignData(selectedNodes: FlowNodeEntity[]) {

    function buildSelectedNodes() {
      const nodes = selectedNodes.map(node => {
        return node.toJSON()
      })

      return nodes
    }

    function buildEdges() {

      let edges: WorkflowEdgeJSON[] = []

      let nodeIds = selectedNodes.map(node => {
        return node.id
      })
      selectedNodes.map(node => {
        let inputLines = node.getData(WorkflowNodeLinesData).inputLines

        inputLines.map(line => {
          if (!nodeIds.includes(line.from.id)) {
            return
          }

          let edge = {
            sourceNodeID: line.from.id,
            targetNodeID: line.to!.id,
            sourcePortID: line.fromPort.portID,
            targetPortID: line.toPort!.portID
          }

          if (edges.find(item => {
            return item.sourceNodeID == edge.sourceNodeID &&
              item.targetNodeID == edge.targetNodeID &&
              item.sourcePortID == edge.sourcePortID &&
              item.targetPortID == edge.targetPortID
          })) {
            return
          }
          edges.push(edge)
        })

        // 输出线条
        let outputLines = node.getData(WorkflowNodeLinesData).outputLines
        outputLines.map(line => {
          if (!nodeIds.includes(line.to!.id)) {
            return
          }

          let edge = {
            sourceNodeID: line.from.id,
            targetNodeID: line.to!.id,
            sourcePortID: line.fromPort.portID,
            targetPortID: line.toPort!.portID
          }

          if (edges.find(item => {
            return item.sourceNodeID == edge.sourceNodeID &&
              item.targetNodeID == edge.targetNodeID &&
              item.sourcePortID == edge.sourcePortID &&
              item.targetPortID == edge.targetPortID
          })) {
            return
          }
          edges.push(edge)
        })

      })

      return edges
    }

    ///@ts-ignore
    let nodes: FlowNodeJSON[] = buildSelectedNodes()


    let edges = buildEdges()

    return {
      nodes,
      edges
    }

  }

  function onSubGraphSave() {
    var selectedNodes = flowSelectConfig.selectedNodes
    // selectedNodes.map(node => {
    //   let inputLines = node.getData(WorkflowNodeLinesData).inputLines
    //   // 输出线条
    //   let outputLines = node.getData(WorkflowNodeLinesData).outputLines


    //   let json1 = node.toJSON()

    //   let jsonData = node.getJSONData()
    //   let j = 0;
    // })

    // var json = flowSelectConfig.toJSON()
    // var i = 0;

    let designContent: FlowDocumentJSON = buildDesignData(selectedNodes)
   // let businessContent = designDataToBusinessData(designData)

    setEntity({...entity, designContent})
    setSaveDrawerVisible(true)
  }

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
        <Tooltip content={'Collapse'}>
          <Button
            icon={<IconShrink />}
            style={{ height: BUTTON_HEIGHT }}
            type="primary"
            theme="solid"
            onMouseDown={(e) => {
              commandRegistry.executeCommand(FlowCommandId.COLLAPSE);
            }}
          />
        </Tooltip>

        <Tooltip content={'Expand'}>
          <Button
            icon={<IconExpand />}
            style={{ height: BUTTON_HEIGHT }}
            type="primary"
            theme="solid"
            onMouseDown={(e) => {
              commandRegistry.executeCommand(FlowCommandId.EXPAND);
            }}
          />
        </Tooltip>

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

        {/* <Tooltip content={'Copy'}>
          <Button
            icon={<IconCopy />}
            style={{ height: BUTTON_HEIGHT }}
            type="primary"
            theme="solid"
            onClick={() => {
              commandRegistry.executeCommand(FlowCommandId.COPY);
            }}
          />
        </Tooltip> */}

        <Tooltip content={'Save'}>
          <Button
            icon={<IconSave />}
            style={{ height: BUTTON_HEIGHT }}
            type="primary"
            theme="solid"
            onClick={() => {
              //commandRegistry.executeCommand(FlowCommandId.COPY);
              console.log('保存........')
              onSubGraphSave()
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
        onSure={onflowSaveDrawrSure}
        onClose={onFlowSaveDrawerClose}
      />
    </SideSheet>
  </ >
}

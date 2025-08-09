import { FC, useState } from 'react';

import { Field, FlowNodeEntity, useClientContext, WorkflowEdgeJSON, WorkflowNodeLinesData } from '@flowgram.ai/free-layout-editor';
import { Button, Input, Toast } from '@douyinfe/semi-ui';

import { GroupField } from '../constant';
import lodash from 'lodash';
import { FlowDocumentJSON, FlowNodeJSON } from '../../../typings';
import { designDataToBusinessData, getAllEdges, getEdgeMaps } from '../../../pages/components/flow/FlowSaveDrawer/functions';
import { useFlowEnviromentContext } from '../../../context/flow-enviroment-context';
import { apiWorkFlowSave } from '../../../pages/Layout/Design/WorkFlow/api';
import { ReadableStreamDefaultController } from 'node:stream/web';

export const GroupTitle: FC = () => {
  const [inputting, setInputting] = useState(false);

  const ctx = useClientContext()

  const flowEnviroment = useFlowEnviromentContext();



  function buildDesignData(selectedNodes: FlowNodeJSON[], allEdges: WorkflowEdgeJSON[]) {

    // function buildSelectedNodes() {
    //   const nodes = selectedNodes.map(node => {
    //     return node.toJSON()
    //   })

    //   return nodes
    // }

    let allNodeIds = selectedNodes.map(node => {
      return node.id
    })

    // function buildEdges() {

    //   let edges: WorkflowEdgeJSON[] = []

    //   selectedNodes.map(node=>{

    //   })
    

    //   return edges
    // }

    ///@ts-ignore
    //let nodes: FlowNodeJSON[] = buildSelectedNodes()


    //let edges = buildEdge.s()

    let edgges = allEdges.filter(edge=>{
        return  lodash.intersection(allNodeIds, [edge.sourceNodeID, edge.targetNodeID]).length > 0
    })

    return {
      nodes: selectedNodes,
      edges : edgges
    }

  }




  async function hanleSubFlowSave(groupName: string) {

    if(!groupName.endsWith('.json')){
      Toast.warning('sub flow name should end with .json')
      
    }

    let workFlowJson:any= ctx.document.toJSON()
    let allNodes = workFlowJson.nodes

      let allEdges = getAllEdges(workFlowJson);


    let selectedNodes: any = allNodes.find(item => item.type == 'group' && item.data.name_ == groupName)?.blocks!

    var j = 0

    let designContent: FlowDocumentJSON = buildDesignData(selectedNodes, allEdges)
  
    //designContent.edges = allEdges

    let businessContent = designDataToBusinessData(designContent, flowEnviroment.graphTopNode, allNodes)

    let edgeMaps = getEdgeMaps(allNodes, allEdges)

    let selectedNodeIds = selectedNodes.map(node => {
      return node.id
    })

    let subFlowInputEdges = designContent.edges.filter(edge => !selectedNodeIds.includes(edge.sourceNodeID))

    let inputs_ = lodash.uniqBy(subFlowInputEdges, item=>item.sourceNodeID).map(edge => {

      let soureNode = allNodes.find(item => item.id == edge.sourceNodeID)!

      let outputs = soureNode.data.outputs_ ?? []
      let output = outputs.find(item => item.id == edge.sourcePortID)

      let name_ = edgeMaps[edge.sourceNodeID + "@" + edge.sourcePortID]
      return {
        ...lodash.omit(output, ['id']),
        name_
      }

    })


    let subFlowOutputEdges = designContent.edges.filter(edge => !selectedNodeIds.includes(edge.targetNodeID))

    let outputs_ = lodash.uniqBy(subFlowOutputEdges, item=>item.targetNodeID).map(edge => {

      let outputNode = allNodes.find(item => item.id == edge.targetNodeID)!

      let inputs = outputNode.data.inputs_ ?? []
      let input = inputs.find(item => item.id == edge.targetPortID)

      let name_ = edgeMaps[edge.sourceNodeID + "@" + edge.sourcePortID]
      return {
        ... lodash.omit(input, ['id']),
        name_
      }

    })

    businessContent.inputs_ = inputs_
    businessContent.outputs_ = outputs_
    businessContent.name_  = groupName

    let temp = businessContent
    let i = 0;


    //return;

    const response = await apiWorkFlowSave("", businessContent);
    if (response.flag == "success") {
      Toast.success('save subflow successed')
    } else {
      Toast.error('save subflow failed')
    }
  }
  return (
    <Field<string> name={GroupField.Title}>
      {({ field }) =>
        inputting ? (
          <Input
            autoFocus
            className="workflow-group-title-input"
            size="small"
            value={field.value}
            onChange={field.onChange}
            onMouseDown={(e) => e.stopPropagation()}
            onBlur={() => setInputting(false)}
            draggable={false}
            onEnterPress={() => setInputting(false)}
          />
        ) : (
          <>
            <span className="workflow-group-title" onDoubleClick={() => setInputting(true)}>
              {field.value ?? 'Group'}
            </span>
            <Button onClick={() => hanleSubFlowSave(field.value)} >save</Button>
          </>
        )
      }
    </Field>
  );
};

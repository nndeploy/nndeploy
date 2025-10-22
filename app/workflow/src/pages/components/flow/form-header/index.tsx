import { Field, FieldRenderProps, useClientContext, WorkflowLinePortInfo, WorkflowNodePortsData } from '@flowgram.ai/free-layout-editor';
import { Typography, Button } from '@douyinfe/semi-ui';
import { IconSmallTriangleDown, IconSmallTriangleLeft } from '@douyinfe/semi-icons';
import { Header, Operators, Title } from './styles';
import { useIsSidebar, useNodeRenderContext } from '../../../../hooks';
import { NodeMenu } from '../../../../components/node-menu';
import { Feedback } from '../../../../form-components';
import { toggleLoopExpanded } from '../../../../utils/toggle-loop-expanded';
import { useEffect, useState } from 'react';
import { IExpandInfo } from '../entity';
import lodash from 'lodash'
import { adjustInputLinesReleventContainerNodeExpandInfo, adjustOutputLinesReleventContainerNodeExpandInfo, destroySubcavasInputLines, destroySubcavasOutputLines, getSubcavasInputLines, getSubcavasOutputLines } from './function';
import { getNodeById, getNodeByName, getNodeExpandInfo, getNodeNameByNodeId, isContainerNode } from '../functions';
import { getIcon } from './utils';

const { Text } = Typography;

export function FormHeader() {
  const { node, expanded, toggleExpand, readonly, deleteNode, form } = useNodeRenderContext();
  const isSidebar = useIsSidebar();

  const clientContext = useClientContext();
  const linesManager = clientContext.document.linesManager

  const [diposedPort, setDiposedPort] = useState<any[]>([])


  useEffect(() => {




    if (node?.getNodeMeta?.()?.needInitShrip) {

      let temp = node?.getNodeMeta?.()

      console.log('iShrimpDelayIndex', node?.getNodeMeta?.()?.iShrimpDelayIndex * 100)
      setTimeout(() => {
        toggleExpand();
        shrimpNode()
      },

        node?.getNodeMeta?.()?.iShrimpDelayIndex * 100 + 300)

    }
  }, [node?.getNodeMeta?.()?.needInitShripIndex])



  const handleExpand = (e: React.MouseEvent) => {
    toggleExpand();
    if (expanded == true) {
      shrimpNode()
    } else {
      expandNode()
    }
    e.stopPropagation(); // Disable clicking prevents the sidebar from opening
  };

  useEffect(() => {
    // 折叠 loop 子节点
    //if (node.flowNodeType === 'loop') {

    if(isContainerNode(node.id, clientContext)){

      toggleLoopExpanded(node, expanded);
    }

    // node.updateExtInfo({ expanded });

    // node.updateExtInfo({ hoby: 'pingpang' });
    //}
  }, [expanded, isSidebar]);


  // useEffect(() => {
  //   const disposible = node.getData(WorkflowNodeLinesData).onDataChange(debounce(
  //     () => {


  //       //const renderData = node.getData<FlowNodeRenderData>(FlowNodeRenderData)
  //       return;

  //       const isContainer = isContainerNode(node.id, clientContext)

  //       if (!isContainer) {
  //         return
  //       }
  //       const isExpanded = isNodeExpaned(node.id)
  //       if (isExpanded) {
  //         return
  //       }

  //       const expandInfo: IExpandInfo = node.getNodeMeta()?.expandInfo || { inputLines: [], outputLines: [] } as IExpandInfo

  //       const expandInputLines = expandInfo.inputLines || []

  //       const inputLines = node.getData(WorkflowNodeLinesData).inputLines

  //       if (node.form?.getValueIn('name_') == 'Prefill_1') {
  //         let j = 0
  //       }

  //       let newExpandInputLines = expandInputLines.filter(expandInputLine => {
  //         let findResult = inputLines.find(inputLine => {
  //           return expandInputLine.from === inputLine.from?.id
  //             && expandInputLine.fromPort === inputLine.fromPort?.portID
  //             && expandInputLine.to === inputLine.to?.id
  //             && expandInputLine.toPort === inputLine.toPort?.portID
  //         })

  //         return findResult
  //       })


  //       const expandOutputLines = expandInfo.outputLines || []

  //       const outputLines = node.getData(WorkflowNodeLinesData).outputLines

  //       let newExpandOutputLines = expandOutputLines.filter(expandOutputLine => {
  //         let findResult = outputLines.find(outputLine => {
  //           return expandOutputLine.from === outputLine.from?.id && expandOutputLine.fromPort === outputLine.fromPort?.portID
  //             &&
  //             expandOutputLine.to === outputLine.to?.id && expandOutputLine.toPort === outputLine.toPort?.portID
  //         })

  //         return findResult
  //       })

  //       let newExpandInfo = {

  //         inputLines: newExpandInputLines,
  //         outputLines: newExpandOutputLines
  //       }
  //       //node.getNodeMeta

  //       node.getNodeMeta().expandInfo = newExpandInfo

  //       var temp = node.getNodeMeta().expandInfo

  //       let temp2 = 2

  //       //node.updateExtInfo({ expandInfo: newExpandInfo })

  //     }), 50)

  //   return () => {
  //     disposible.dispose()
  //   }
  // }, [])



  //const allInnerNodes = getAllInnerNodes(node)
  //const allInnerNodeIds = allInnerNodes.map((node) => node.id)


  // function isNodeExpaned(nodeId: string) {
  //   let node = clientContext.document.getNode(nodeId)
  //   let renderData = node?.getData(FlowNodeRenderData)
  //   return renderData?.expanded === undefined || renderData?.expanded === true


  // }



  function shrimpNode() {

    /** lines from outside to this node's children*/


    let prefillNode = getNodeByName('Prefill_2', clientContext)
    let prefillNodeExpandInfo = prefillNode?.getNodeMeta().expandInfo

    const allInputLines = getSubcavasInputLines(node, clientContext)
    destroySubcavasInputLines(node, clientContext)
    adjustInputLinesReleventContainerNodeExpandInfo(allInputLines, clientContext)


    prefillNode = getNodeByName('Prefill_2', clientContext)
    prefillNodeExpandInfo = prefillNode?.getNodeMeta().expandInfo

    const allOutputLines = getSubcavasOutputLines(node, clientContext)
    destroySubcavasOutputLines(node, clientContext)
    adjustOutputLinesReleventContainerNodeExpandInfo(allOutputLines, clientContext)



    prefillNode = getNodeByName('Prefill_2', clientContext)
    prefillNodeExpandInfo = prefillNode?.getNodeMeta().expandInfo

    let inputs = lodash.uniqBy(allInputLines, ['from', 'fromPort']).map(item => {
      return {
        //id: "new_port_" + Math.random().toString(36).substr(2, 9),
        id: item.toPort,
        type_: item.type_,
        desc_: item.desc_,
      }
    })


    let temp = lodash.uniqBy(allOutputLines, ['oldFrom', 'oldFromPort'])

    let outputs = lodash.uniqBy(allOutputLines, ['oldFrom', 'oldFromPort']).map(item => {
      return {
        //id: "new_port_" + Math.random().toString(36).substr(2, 9),
        id: item.fromPort,
        type_: item.type_,
        desc_: item.desc_,
      }
    })



    form?.setValueIn('inputs_', inputs)
    form?.setValueIn('outputs_', outputs)



    const allPorts = node.getData(WorkflowNodePortsData).allPorts

    allPorts.map(port => {
      port.dispose()
      setDiposedPort([...diposedPort, port.portID])
    })


    const expandInfo: IExpandInfo = {

      inputLines: allInputLines,
      outputLines: allOutputLines,

    }

    if (node.form?.getValueIn('name_') == 'Prefill_2') {
      let k = 0
    }

    node.getNodeMeta().expandInfo = expandInfo
    // node.updateExtInfo({ expandInfo })


    prefillNode = getNodeByName('Prefill_2', clientContext)
    prefillNodeExpandInfo = prefillNode?.getNodeMeta().expandInfo

    setTimeout(() => {


      // node.getData(WorkflowNodePortsData).updateAllPorts(
      //   [
      //     ...inputs.map(item=>({ portID: item.id, type: 'input' as const}), 
      //     ...outputs.map(item=>({ portID: item.id, type: 'output' as const}))
      //   )
      //   ])
      node.getData(WorkflowNodePortsData).updateDynamicPorts()


      setTimeout(() => {

        allInputLines.map(inputLine => {
          linesManager.createLine({
            ...inputLine,
            // to: node.id
          })
        })


        allOutputLines.map(outLine => {
          linesManager.createLine({
            ...outLine,
            // from: node.id
          })
        })


      }, 20)

    }, 20)



  }

  function expandNode() {
    const expandInfo = node.getNodeMeta().expandInfo as IExpandInfo

    const allPorts = node.getData(WorkflowNodePortsData).allPorts

    allPorts.map(port => {
      port.dispose()

    })

    setDiposedPort([...diposedPort, allPorts.map(item => item.portID)])

    form?.setValueIn('inputs_', [])
    form?.setValueIn('outputs_', [])

    node.getData(WorkflowNodePortsData).updateAllPorts([])

    setTimeout(() => {

      node.getData(WorkflowNodePortsData).updateDynamicPorts()

      expandInfo.inputLines.map(inputLine => {

        let line: WorkflowLinePortInfo = { ...inputLine, to: inputLine.oldTo, toPort: inputLine.oldToPort }

        if (line.fromPort && diposedPort.includes(line.fromPort)) {
          let i = 0
        }

        if (line.toPort && diposedPort.includes(line.toPort)) {
          let j = 0
        }

        linesManager.createLine(line)

        const isFromNodeContainer = isContainerNode(inputLine.from, clientContext)
        if (!isFromNodeContainer) {
          return
        }

        const formNodeExandInfo = getNodeExpandInfo(inputLine.from, clientContext)
        const outputLines = formNodeExandInfo?.outputLines.map(outputLine => {


          if (
            //[...allInnerNodeIds, node.id].includes(outputLine.to!)
            outputLine.from == inputLine.from && outputLine.fromPort == inputLine.fromPort
            &&
            outputLine.oldTo == inputLine.oldTo && outputLine.oldToPort == inputLine.oldToPort
          ) {
            return {
              ...outputLine,
              //to: node.id,
              oldTo: '',
              oldTo_name: '',
              to: outputLine.oldTo,
              to_name: getNodeNameByNodeId(outputLine.oldTo!, clientContext),
              //oldTo: inputLine.oldTo,
              oldToPort: '',

              toPort: outputLine.oldToPort,
              //oldToPort: inputLine.oldToPort

            }
          } else {
            return outputLine
          }
        })

        getNodeById(inputLine.from, clientContext)!.getNodeMeta().expandInfo = { ...formNodeExandInfo, outputLines }

        let temp = getNodeById(inputLine.from, clientContext)!.getNodeMeta().expandInfo
        let j = 0


      })




      expandInfo.outputLines.map(outputLine => {

        let line: WorkflowLinePortInfo = { ...outputLine, from: outputLine.oldFrom, fromPort: outputLine.oldFromPort }
        linesManager.createLine(line)

        if (line.fromPort && diposedPort.includes(line.fromPort)) {
          let i = 0
        }

        if (line.toPort && diposedPort.includes(line.toPort)) {
          let j = 0
        }


        const isToNodeContainer = isContainerNode(outputLine.to, clientContext)
        if (!isToNodeContainer) {
          return
        }

        const toNodeExandInfo = getNodeExpandInfo(outputLine.to, clientContext)
        const inputLines = toNodeExandInfo?.inputLines.map(inputLine => {


          if (
            // [...allInnerNodeIds, node.id].includes(inputLine.from!)
            inputLine.from == outputLine.from && inputLine.fromPort == outputLine.fromPort
            &&
            inputLine.oldTo == outputLine.oldTo && inputLine.oldToPort == outputLine.oldToPort
          ) {
            return {
              ...inputLine,

              //oldFrom: outputLine.oldFrom,
              oldFrom: '',
              oldFrom_name: '',
              from: outputLine.oldFrom,
              from_name: getNodeNameByNodeId(outputLine.oldFrom!, clientContext),

              oldFromPort: '',

              fromPort: outputLine.oldFromPort,
              //oldFromPort: outputLine.oldFromPort

            }
          } else {
            return inputLine
          }
        })

        getNodeById(outputLine.to, clientContext)!.getNodeMeta().expandInfo = { ...toNodeExandInfo, inputLines }

        let temp = getNodeById(outputLine.to, clientContext)!.getNodeMeta().expandInfo
        let j = 0
      })

      expandInfo.inputLines = []
      expandInfo.outputLines = []

      //node.updateExtInfo({ expandInfo })
      node.getNodeMeta().expandInfo = expandInfo
    }, 20)


  }

  return (
    <>
      <Header>
        {getIcon(node)}
        <Title>
          {/* <Field name="key_">
          {({ field: { value, onChange }, fieldState }: FieldRenderProps<string>) => (
            <div style={{ height: 24 }}>
              <Text ellipsis={{ showTooltip: true }} >{value}</Text>
              <Feedback errors={fieldState?.errors} />
            </div>
          )}
        </Field> */}
          <Field name="name_">
            {({ field: { value, onChange }, fieldState }: FieldRenderProps<string>) => (
              <div style={{ height: 24 }}>
                <Text ellipsis={{ showTooltip: false }}>{value}</Text>
                <Feedback errors={fieldState?.errors} />
              </div>
            )}
          </Field>
        </Title>
        {node.renderData.expandable && !isSidebar && (
          <Button
            type="primary"
            icon={expanded ? <IconSmallTriangleDown /> : <IconSmallTriangleLeft />}
            size="small"
            theme="borderless"
            onClick={handleExpand}
          />
        )}
        {readonly ? undefined : (
          <Operators>
            <NodeMenu node={node} deleteNode={deleteNode} />
          </Operators>
        )}
      </Header>
      {/* <SecondTitle> 
     
    </SecondTitle> */}
    </>
  );
}

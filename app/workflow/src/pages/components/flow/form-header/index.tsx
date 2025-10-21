import { Field, FieldRenderProps, FlowNodeEntity, FlowNodeRenderData, FreeLayoutPluginContext, getNodeForm, useClientContext, WorkflowLineEntity, WorkflowLinePortInfo, WorkflowNodeLinesData, WorkflowNodePortsData } from '@flowgram.ai/free-layout-editor';
import { Typography, Button } from '@douyinfe/semi-ui';
import { IconSmallTriangleDown, IconSmallTriangleLeft } from '@douyinfe/semi-icons';
import { getIcon } from './utils';
import { Header, Operators, Title, SecondTitle } from './styles';
import { useIsSidebar, useNodeRenderContext } from '../../../../hooks';
import { NodeMenu } from '../../../../components/node-menu';
import { Feedback } from '../../../../form-components';
import lodash, { debounce, throttle } from 'lodash';
import { toggleLoopExpanded } from '../../../../utils/toggle-loop-expanded';
import { useEffect, useState } from 'react';
import { IExpandInfo, ILineEntity } from '../entity';
import { getNodeByName } from './function';
import { getNodeById, isContainerNode } from '../functions';

const { Text } = Typography;

export function FormHeader() {
  const { node, expanded, toggleExpand, readonly, deleteNode, form } = useNodeRenderContext();
  const isSidebar = useIsSidebar();

  const clientContext = useClientContext();
  const linesManager = clientContext.document.linesManager

  const [diposedPort, setDiposedPort] = useState<any[]>([])


  useEffect(() => {
    if (!expanded) {
      // setTimeout(()=>{
      shrimpNode()
      // }, 3000)

    }
  }, [])



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
    toggleLoopExpanded(node, expanded);

    // node.updateExtInfo({ expanded });

    // node.updateExtInfo({ hoby: 'pingpang' });
    //}
  }, [expanded]);


  useEffect(() => {
    const disposible = node.getData(WorkflowNodeLinesData).onDataChange(debounce(
      () => {


        //const renderData = node.getData<FlowNodeRenderData>(FlowNodeRenderData)
        return;

        const isContainer = isContainerNode(node.id, clientContext)

        if (!isContainer) {
          return
        }
        const isExpanded = isNodeExpaned(node.id)
        if (isExpanded) {
          return
        }

        const expandInfo: IExpandInfo = node.getNodeMeta()?.expandInfo || { inputLines: [], outputLines: [] } as IExpandInfo

        const expandInputLines = expandInfo.inputLines || []

        const inputLines = node.getData(WorkflowNodeLinesData).inputLines

        if (node.form?.getValueIn('name_') == 'Prefill_1') {
          let j = 0
        }

        let newExpandInputLines = expandInputLines.filter(expandInputLine => {
          let findResult = inputLines.find(inputLine => {
            return expandInputLine.from === inputLine.from?.id
              && expandInputLine.fromPort === inputLine.fromPort?.portID
              && expandInputLine.to === inputLine.to?.id
              && expandInputLine.toPort === inputLine.toPort?.portID
          })

          return findResult
        })


        const expandOutputLines = expandInfo.outputLines || []

        const outputLines = node.getData(WorkflowNodeLinesData).outputLines

        let newExpandOutputLines = expandOutputLines.filter(expandOutputLine => {
          let findResult = outputLines.find(outputLine => {
            return expandOutputLine.from === outputLine.from?.id && expandOutputLine.fromPort === outputLine.fromPort?.portID
              &&
              expandOutputLine.to === outputLine.to?.id && expandOutputLine.toPort === outputLine.toPort?.portID
          })

          return findResult
        })

        let newExpandInfo = {

          inputLines: newExpandInputLines,
          outputLines: newExpandOutputLines
        }
        //node.getNodeMeta

        node.getNodeMeta().expandInfo = newExpandInfo

        var temp = node.getNodeMeta().expandInfo

        let temp2 = 2

        //node.updateExtInfo({ expandInfo: newExpandInfo })

      }), 50)

    return () => {
      disposible.dispose()
    }
  }, [])

  function getAllInnerNodes(node: FlowNodeEntity) {
    let allChildren: FlowNodeEntity[] = []
    if (node.blocks && node.blocks.length > 0) {
      node.blocks.forEach((child) => {
        allChildren.push(child);
        allChildren = allChildren.concat(...getAllInnerNodes(child));
      });
    }
    return allChildren;
  }

  const allInnerNodes = getAllInnerNodes(node)
  const allInnerNodeIds = allInnerNodes.map((node) => node.id)





  function getNodeExpandInfo(nodeId: string) {
    let node = clientContext.document.getNode(nodeId)
    let expandInfo: IExpandInfo | undefined = node?.getNodeMeta().expandInfo
    return expandInfo
  }

  function isNodeExpaned(nodeId: string) {
    let node = clientContext.document.getNode(nodeId)
    let renderData = node?.getData(FlowNodeRenderData)
    return renderData?.expanded === undefined || renderData?.expanded === true


  }

  function getNodeName(nodeId: string) {
    let node = clientContext.document.getNode(nodeId)
    let name = node?.form?.getValueIn('name_')
    return name
  }


  function shrimpNode() {

    /** lines from outside to this node's children*/
    function getInputLines() {
      const allInputLines = allInnerNodes.map((innerNode) => {
        let lines = innerNode.getData(WorkflowNodeLinesData).inputLines.filter((line) => {
          return !allInnerNodeIds.includes(line.from!.id)
        })

        return lines.map(line => {

          let form = line.to!.form
          let inputs = form?.getValueIn('inputs_')
          let port = inputs.find((input: any) => {
            return input.id == line.toPort?.portID

          })


          let result: ILineEntity = {
            oldFrom: '',
            oldFromPort: '',

            from: line.from!.id,
            from_name: getNodeName(line.from!.id),
            fromPort: line.fromPort!.portID,

            oldTo: line.to!.id,
            oldTo_name: getNodeName(line.to!.id),
            oldToPort: line.toPort!.portID,

            to: node.id,
            to_name: getNodeName(node.id),
            //toPort: line.toPort!.portID,

            toPort: "new_port_" + Math.random().toString(36).substr(2, 9),

            // toPort: "new_port_" + Math.random().toString(36).substr(2, 9),
            type_: port?.type_,
            desc_: port?.desc_,
          }


          const isFromNodeContainer = isContainerNode(line.from!.id, clientContext)
          if (isFromNodeContainer) {
            const formNodeExandInfo = getNodeExpandInfo(line.from!.id)
            formNodeExandInfo?.outputLines.map(outputLine => {

              if (outputLine.from == result.from && outputLine.fromPort == result.fromPort
                && outputLine.to == result.oldTo && outputLine.toPort == result.oldToPort) {

                result.oldFrom = outputLine.oldFrom
                result.oldFrom_name = outputLine.oldFrom_name
                result.oldFromPort = outputLine.oldFromPort

              }

            })
          }
          line.dispose()
          return result
        })
      })
      return allInputLines.flat()


    }

    let prefillNode = getNodeByName('Prefill_2', clientContext)
    let prefillNodeExpandInfo = prefillNode?.getNodeMeta().expandInfo

    const allInputLines = getInputLines()
    adjustInputLinesReleventContainerNodeExpandInfo(allInputLines)


     prefillNode = getNodeByName('Prefill_2', clientContext)
    prefillNodeExpandInfo = prefillNode?.getNodeMeta().expandInfo


    function adjustInputLinesReleventContainerNodeExpandInfo(allInputLines: ILineEntity[]) {
      allInputLines.map(inputLine => {
        const isFromNodeContainer = isContainerNode(inputLine.from, clientContext)
        if (!isFromNodeContainer) {
          return
        }

        const formNodeExandInfo = getNodeExpandInfo(inputLine.from)
        const outputLines = formNodeExandInfo?.outputLines.map(outputLine => {


          if (
            //[...allInnerNodeIds, node.id].includes(outputLine.to)
            outputLine.from == inputLine.from && outputLine.fromPort == inputLine.fromPort
            &&
            outputLine.to == inputLine.oldTo && outputLine.toPort == inputLine.oldToPort

          ) {
            return {
              ...outputLine,
              oldTo: outputLine.to,
              oldTo_name: getNodeName(outputLine.to),
              oldToPort: outputLine.toPort,

              to: inputLine.to,
              to_name: getNodeName(inputLine.to),
              toPort: inputLine.toPort

              // toPort: inputLine.toPort,
              // oldToPort: inputLine.oldToPort

            }
          } else {
            return outputLine
          }
        })

        if (getNodeById(inputLine.from, clientContext)?.form?.getValueIn('name_') == 'Prefill_2') {
          let k = 0
        }

        getNodeById(inputLine.from, clientContext)!.getNodeMeta().expandInfo = { ...formNodeExandInfo, outputLines }

        let temp = getNodeById(inputLine.from, clientContext)!.getNodeMeta().expandInfo
        let j = 0


      })
    }


    /** lines from this this node's children to  outside */
    function getOutputLines() {
      const allOutputLines = allInnerNodes.map((innerNode) => {
        let lines = innerNode.getData(WorkflowNodeLinesData).outputLines.filter((line) => {
          return !allInnerNodeIds.includes(line.to!.id)
        })



        return lines.map(line => {

          let form = line.from?.form
          let outputPorts = form?.getValueIn('outputs_')
          let port = outputPorts.find((output: any) => {

            return output.id == line.fromPort!.portID

          })

          let result: ILineEntity = {

            oldFrom: line.from!.id,
            oldFrom_name: getNodeName(line.from!.id),
            oldFromPort: line.fromPort!.portID,

            from: node.id,
            from_name: getNodeName(node.id),
            //from: node.id, 

            fromPort: "new_port_" + Math.random().toString(36).substr(2, 9),
            //fromPort: "new_port_" + Math.random().toString(36).substr(2, 9),
            to: line.to!.id,
            to_name: getNodeName(line.to!.id),

            toPort: line.toPort!.portID,

            oldTo: '',
            oldTo_name: '',
            oldToPort: '',

            type_: port?.type_,
            desc_: port?.desc_,
          }

          const isToNodeContainer = isContainerNode(line.to!.id, clientContext)
          if (isToNodeContainer) {
            const toNodeExandInfo = getNodeExpandInfo(line.to!.id)
            toNodeExandInfo?.inputLines.map(inputLine => {

              if (inputLine.from == result.oldFrom && inputLine.fromPort == result.oldFromPort
                &&
                inputLine.to == result.to && inputLine.toPort == result.toPort

              ) {
                result.oldTo = inputLine.oldTo
                result.oldTo_name = inputLine.oldTo_name
                result.oldToPort = inputLine.oldToPort
              }

            })
          }



          line.dispose()

          return result


        })


      })

      return allOutputLines.flat()


    }

    const allOutputLines = getOutputLines()
    adjustOutputLinesReleventContainerNodeExpandInfo(allOutputLines)

    function adjustOutputLinesReleventContainerNodeExpandInfo(allOutputLines: ILineEntity[]) {
      allOutputLines.map(outputLine => {
        const isToNodeContainer = isContainerNode(outputLine.to, clientContext)
        if (!isToNodeContainer) {
          return
        }

        const toNodeExandInfo = getNodeExpandInfo(outputLine.to)
        const inputLines = toNodeExandInfo?.inputLines.map(inputLine => {


          if (
            //[...allInnerNodeIds, node.id].includes(inputLine.from!)
            inputLine.from == outputLine.oldFrom && inputLine.fromPort == outputLine.oldFromPort
            &&
            inputLine.to == outputLine.to && inputLine.toPort == outputLine.toPort
          ) {
            return {
              ...inputLine,

              oldFrom_name: getNodeName(inputLine.from),
              oldFromPort: inputLine.fromPort,

              from: outputLine.from,
              from_name: getNodeName(outputLine.from),
              fromPort: outputLine.fromPort

              // fromPort: outputLine.fromPort,
              // oldFromPort: outputLine.oldFromPort



            }
          } else {
            return inputLine
          }
        })

        getNodeById(outputLine.to, clientContext)!.getNodeMeta().expandInfo = { ...toNodeExandInfo, inputLines }


      })
    }


     prefillNode = getNodeByName('Prefill_2', clientContext)
    prefillNodeExpandInfo = prefillNode?.getNodeMeta().expandInfo

    let inputs = allInputLines.map(item => {
      return {
        //id: "new_port_" + Math.random().toString(36).substr(2, 9),
        id: item.toPort,
        type_: item.type_,
        desc_: item.desc_,
      }
    })
    let outputs = allOutputLines.map(item => {
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

    }, 10)



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

      //node.getData(WorkflowNodePortsData).updateDynamicPorts()

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

        const formNodeExandInfo = getNodeExpandInfo(inputLine.from)
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
              to_name: getNodeName(outputLine.oldTo!),
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

        const toNodeExandInfo = getNodeExpandInfo(outputLine.to)
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
              from_name: getNodeName(outputLine.oldFrom!),

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
    }, 50)


  }

  return (
    <>
      <Header>
        {/* {getIcon(node)} */}
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

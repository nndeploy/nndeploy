import { FlowNodeEntity, FreeLayoutPluginContext, WorkflowNodeLinesData } from "@flowgram.ai/free-layout-editor";
import { ILineEntity } from "../entity";
import { getAllInnerNodes, getNodeById, getNodeExpandInfo, getNodeNameByNodeId, isContainerNode } from "../functions";

/** lines from  outsize this node's inner node */
export function getSubcavasInputLines(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {

  const newPortMap: { [fromNodeAndPort: string]: string } = {}

  const allInnerNodes = getAllInnerNodes(node)
  const allInnerNodeIds = allInnerNodes.map((node) => node.id)
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

      function getNewToPort() {
        let newToPort = ''

        const alreadyPort = newPortMap[`${line.from!.id}_${line.fromPort!.portID}`]


        if (alreadyPort) {
          newToPort = alreadyPort
        } else {
          newToPort = "new_port_" + Math.random().toString(36).substr(2, 9)
          newPortMap[`${line.from!.id}_${line.fromPort!.portID}`] = newToPort
        }
        return newToPort
      }

      const newToPort = getNewToPort()


      let old_to_name = getNodeNameByNodeId(line.to!.id, clientContext)
      let result: ILineEntity = {
        oldFrom: '',
        oldFromPort: '',

        from: line.from!.id,
        from_name: getNodeNameByNodeId(line.from!.id, clientContext),
        fromPort: line.fromPort!.portID,

        oldTo: line.to!.id,
        oldTo_name: old_to_name,
        oldToPort: line.toPort!.portID,

        to: node.id,
        to_name: getNodeNameByNodeId(node.id, clientContext),
        //toPort: line.toPort!.portID,

        toPort:  newToPort,   //"new_port_" + Math.random().toString(36).substr(2, 9),

        // toPort: "new_port_" + Math.random().toString(36).substr(2, 9),
        type_: port?.type_,
        desc_: `${port?.desc_}`,
      }


      const isFromNodeContainer = isContainerNode(line.from!.id, clientContext)
      if (isFromNodeContainer) {
        const formNodeExandInfo = getNodeExpandInfo(line.from!.id, clientContext)
        formNodeExandInfo?.outputLines.map(outputLine => {

          if (outputLine.from == result.from && outputLine.fromPort == result.fromPort
            && outputLine.to == result.oldTo && outputLine.toPort == result.oldToPort) {

            result.oldFrom = outputLine.oldFrom
            result.oldFrom_name = outputLine.oldFrom_name
            result.oldFromPort = outputLine.oldFromPort

          }

        })
      }
      // line.dispose()
      return result
    })
  })
  return allInputLines.flat()


}

export function destroySubcavasInputLines(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {
  const allInnerNodes = getAllInnerNodes(node)
  const allInnerNodeIds = allInnerNodes.map((node) => node.id)
  allInnerNodes.map((innerNode) => {
    let lines = innerNode.getData(WorkflowNodeLinesData).inputLines.filter((line) => {
      return !allInnerNodeIds.includes(line.from!.id)
    })
    lines.map(line => {
      line.dispose()
    })
  })
}

export function adjustInputLinesReleventContainerNodeExpandInfo(allInputLines: ILineEntity[], clientContext: FreeLayoutPluginContext) {
  allInputLines.map(inputLine => {
    const isFromNodeContainer = isContainerNode(inputLine.from, clientContext)
    if (!isFromNodeContainer) {
      return
    }

    const formNodeExandInfo = getNodeExpandInfo(inputLine.from!, clientContext)
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
          oldTo_name: getNodeNameByNodeId(outputLine.to!, clientContext),
          oldToPort: outputLine.toPort,

          to: inputLine.to,
          to_name: getNodeNameByNodeId(inputLine.to!, clientContext),
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
export function getSubcavasOutputLines(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {


  const newPortMap: { [fromNodeAndPort: string]: string } = {}

  const allInnerNodes = getAllInnerNodes(node)
  const allInnerNodeIds = allInnerNodes.map((node) => node.id)
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

      const oldFrom_name = getNodeNameByNodeId(line.from!.id, clientContext)

      function getNewFromPort() {
        let newFromPort = ''

        const alreadyPort = newPortMap[`${line.from!.id}_${line.fromPort!.portID}`]


        if (alreadyPort) {
          newFromPort = alreadyPort
        } else {
          newFromPort = "new_port_" + Math.random().toString(36).substr(2, 9)
          newPortMap[`${line.from!.id}_${line.fromPort!.portID}`] = newFromPort
        }
        return newFromPort
      }

      const newFromPort = getNewFromPort()



      let result: ILineEntity = {

        oldFrom: line.from!.id,
        oldFrom_name: oldFrom_name,
        oldFromPort: line.fromPort!.portID,

        from: node.id,
        from_name: getNodeNameByNodeId(node.id, clientContext),
        //from: node.id, 

        fromPort: newFromPort,  //"new_port_" + Math.random().toString(36).substr(2, 9),
        //fromPort: "new_port_" + Math.random().toString(36).substr(2, 9),
        to: line.to!.id,
        to_name: getNodeNameByNodeId(line.to!.id, clientContext),

        toPort: line.toPort!.portID,

        oldTo: '',
        oldTo_name: '',
        oldToPort: '',

        type_: port?.type_,
        desc_: `${port?.desc_}`,
      }

      const isToNodeContainer = isContainerNode(line.to!.id, clientContext)
      if (isToNodeContainer) {
        const toNodeExandInfo = getNodeExpandInfo(line.to!.id, clientContext)
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



      // line.dispose()

      return result


    })


  })

  return allOutputLines.flat()


}

export function destroySubcavasOutputLines(node: FlowNodeEntity, clientContext: FreeLayoutPluginContext) {

  const allInnerNodes = getAllInnerNodes(node)
  const allInnerNodeIds = allInnerNodes.map((node) => node.id)
  allInnerNodes.map((innerNode) => {
    let lines = innerNode.getData(WorkflowNodeLinesData).outputLines.filter((line) => {
      return !allInnerNodeIds.includes(line.to!.id)
    })
    lines.map(line => {
      line.dispose()
    })


  })
}

export function adjustOutputLinesReleventContainerNodeExpandInfo(allOutputLines: ILineEntity[], clientContext: FreeLayoutPluginContext) {
  allOutputLines.map(outputLine => {
    const isToNodeContainer = isContainerNode(outputLine.to, clientContext)
    if (!isToNodeContainer) {
      return
    }

    const toNodeExandInfo = getNodeExpandInfo(outputLine.to, clientContext)
    const inputLines = toNodeExandInfo?.inputLines.map(inputLine => {


      if (
        //[...allInnerNodeIds, node.id].includes(inputLine.from!)
        inputLine.from == outputLine.oldFrom && inputLine.fromPort == outputLine.oldFromPort
        &&
        inputLine.to == outputLine.to && inputLine.toPort == outputLine.toPort
      ) {
        return {
          ...inputLine,

          oldFrom_name: getNodeNameByNodeId(inputLine.from, clientContext),
          oldFromPort: inputLine.fromPort,

          from: outputLine.from,
          from_name: getNodeNameByNodeId(outputLine.from, clientContext),
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


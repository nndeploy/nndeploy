import { IconDownload, IconExpand } from "@douyinfe/semi-icons"
import { IconChangelog } from "@douyinfe/semi-icons-lab"
import { Badge, IconButton, List, Popover, Tooltip } from "@douyinfe/semi-ui"
import { useState } from "react";
import './index.less'
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context";
import { getNodeForm, useClientContext } from "@flowgram.ai/free-layout-editor";

const Download: React.FC<any> = (props) => {

  const [isRunning, setRunning] = useState(false);

  const [visible, setVisible] = useState(false);

  const clientContext = useClientContext();


  const flowEnviroment = useFlowEnviromentContext()

  async function onDownload() {

    setRunning(true)
    const allForms = clientContext.document.getAllNodes().map((node) => getNodeForm(node));

    await Promise.all(allForms.map(async (form) => form?.validate()));

    const json = clientContext.document.toJSON()
    await flowEnviroment.onDownload?.(json as any)
    setRunning(false)
  }

  return <Tooltip
    content={'Download'}>


    <IconButton
      icon={<IconDownload
      // color={runResult ? 'red' : undefined}
      //style={ runResult === 'error' ? {color: 'var(--semi-color-danger)'}: {}}
      // style={{ color: runResult ? 'red' : undefined }}
      />}
      loading={isRunning}
      type="tertiary"
      theme="borderless"
      //color={runResult ? 'red' : undefined}
      onClick={() => {
        onDownload()
      }} />

  </Tooltip>

}

export default Download
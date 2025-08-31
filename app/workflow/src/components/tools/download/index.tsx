import { IconDownload, IconExpand } from "@douyinfe/semi-icons"
import { IconChangelog } from "@douyinfe/semi-icons-lab"
import { Badge, IconButton, List, Modal, Popover, Tooltip } from "@douyinfe/semi-ui"
import { useEffect, useState } from "react";
import './index.less'
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context";
import { getNodeForm, useClientContext } from "@flowgram.ai/free-layout-editor";

const Download: React.FC<any> = (props) => {

  const [isRunning, setRunning] = useState(false);


  const clientContext = useClientContext();


  const flowEnviroment = useFlowEnviromentContext()

  const { downloadModalList, downloadModalVisible, setDownloadModalVisible } = flowEnviroment


  async function onDownload() {

    setDownloadModalVisible(true)


  }

  async function startDownload() {
    setRunning(true)
    const allForms = clientContext.document.getAllNodes().map((node) => getNodeForm(node));

    await Promise.all(allForms.map(async (form) => form?.validate()));

    const json = clientContext.document.toJSON()
    await flowEnviroment.onDownload?.(json as any)
    setRunning(false)
    setDownloadModalVisible(false)

  }

  function handleCancel() {
    setDownloadModalVisible(false)
  }



  return <>
    <Tooltip
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
    <Modal
      title="download modals"
      visible={downloadModalVisible}
      onOk={startDownload}
      onCancel={handleCancel}
      maskClosable={false}
    //okButtonProps={{ size: 'small', type: 'warning' }}
    //cancelButtonProps={{ size: 'small', disabled: true }}
    >
      <div> modals list: </div>
      {downloadModalList.map(item => item)}
    </Modal>
  </>


}

export default Download
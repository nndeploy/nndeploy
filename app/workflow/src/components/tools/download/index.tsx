import { IconDownload, IconExpand } from "@douyinfe/semi-icons"
import { IconChangelog } from "@douyinfe/semi-icons-lab"
import { Badge, IconButton, List, Modal, Popover, Progress, Tooltip } from "@douyinfe/semi-ui"
import { useEffect, useState } from "react";
import styles from './index.module.scss'
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context";
import { getNodeForm, useClientContext } from "@flowgram.ai/free-layout-editor";
import { IDownloadProgress } from "../../../pages/components/flow/entity";

const Download: React.FC<any> = (props) => {

  const [isRunning, setRunning] = useState(false);


  const clientContext = useClientContext();


  const flowEnviroment = useFlowEnviromentContext()

  const { downloadModalList, downloadModalVisible, setDownloadModalVisible, runInfo } = flowEnviroment

  const {downloadProgress} = runInfo


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

  function getDownloadInfo(modelName: string) {

     let nameParts = modelName.split(':')
     let name = nameParts[nameParts.length - 1]

    const downloadInfo =  downloadProgress[name] ? downloadProgress[name] : {
      filename: name,
      percent: 0,
      downloaded: 0,
      elapsed: 0,
      total: 0
    }
    return <Progress percent={downloadInfo?.percent || 0} showInfo={true} aria-label="download progress" />
    
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
      title="download"
      visible={downloadModalVisible}
      okText="download"
      onOk={startDownload}
      onCancel={handleCancel}
      maskClosable={false}
      className={styles['download-model']}
      // modal={false}
      //mask = {true}


      width={700}
    //okButtonProps={{ size: 'small', type: 'warning' }}
    //cancelButtonProps={{ size: 'small', disabled: true }}
    >

      <ol>
        {downloadModalList.map(item => {

          return <li className={styles["model-item"]} key={item}>
            <div className={styles["model-name"]}> {item}</div>
            {
             
             getDownloadInfo(item)
              
            }
          </li>
        })}
      </ol>
    </Modal>
  </>


}

export default Download
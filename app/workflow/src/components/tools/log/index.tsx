import { Badge, Button, List, SideSheet, Tooltip } from "@douyinfe/semi-ui"
import { useState } from "react";
import './index.less'
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context";

const Log: React.FC<any> = (props) => {

  const [visible, setVisible] = useState(false);

  const {  element: flowElementRef, runInfo } = useFlowEnviromentContext()

  const {log, result: runResult } = runInfo


  function showLog() {
    setVisible(true)
  }

  function onClose() {
    setVisible(false)
  }

  function getPopupContainer() {
    return flowElementRef?.current!!
  }


  return <>
    <Badge type='danger'

      dot={runResult === 'error' }
    >
      <Tooltip
        content={'Log'}>
        <Button

          color={runResult ? 'red' : undefined}
          style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
          onClick={() => {
            showLog()
          }}

        >Log</Button>
      </Tooltip>
    </Badge>
    <SideSheet title="Log" visible={visible} onCancel={onClose} placement={'bottom'}
      getPopupContainer={getPopupContainer}
      height={300}
    >
      <div className="tools-log-content">

        <List
          //header={<div>log list</div>}
          footer={<>
            {
              log.time_profile.init_time && <div>init time: <span className="number">{log.time_profile.init_time}</span></div>
            }
            {
              log.time_profile.run_time && <div>run time: <span className="number">{log.time_profile.run_time}</span></div>
            }
          </>}
          bordered
          dataSource={log.items}
          renderItem={item => <List.Item className='list-item'>{item}</List.Item>}
        />


      </div>
    </SideSheet>
  </>

}

export default Log
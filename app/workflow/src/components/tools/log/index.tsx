import { IconExpand } from "@douyinfe/semi-icons"
import { IconChangelog } from "@douyinfe/semi-icons-lab"
import { Badge, Button, IconButton, List, Popover, SideSheet, Tooltip } from "@douyinfe/semi-ui"
import { useState } from "react";
import './index.less'
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context";
import { result } from "lodash";

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

  return <Popover
    trigger="custom"
    position="bottomLeft"
    className="log-popover"
    closeOnEsc
    visible={visible}
    //style={{width: '100%'}}

    onVisibleChange={(v) => {
      //onPopupVisibleChange?.(v);
    }}
    onClickOutSide={() => {
      setVisible(false);
    }}
    spacing={10}

    content={
      <div className="tools-log-content">
        {/* {
            log.items.map((item, index) => {
              return <div key={index}>{item}</div>
            })


          } */}
        <List
          header={<div>log list</div>}
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

    }


  >
    <Badge type='danger'

      dot={runResult === 'error'}
    >
      <Tooltip
        content={'Log'}>
        <Button
          // icon={<IconChangelog
          //   color={runResult ? 'red' : undefined}
          //   //style={ runResult === 'error' ? {color: 'var(--semi-color-danger)'}: {}}
          //   style={{ color: runResult ? 'red' : undefined }}
          // />}
          // type="tertiary"
          //theme="borderless"
          color={runResult ? 'red' : undefined}
          style={{ backgroundColor: 'rgba(171,181,255,0.3)', borderRadius: '8px' }}
          onClick={() => {
            //setVisible(!visible)
            showLog()
          }}


        //style={{marginRight: 10}}
        >Log</Button>
      </Tooltip>

    </Badge>
  </Popover>




}

export default Log
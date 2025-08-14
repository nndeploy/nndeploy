import { IconExpand } from "@douyinfe/semi-icons"
import { IconChangelog } from "@douyinfe/semi-icons-lab"
import { Badge, IconButton, List, Popover, Tooltip } from "@douyinfe/semi-ui"
import { useState } from "react";
import './index.less'
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context";

const Log: React.FC<any> = (props) => {

  const [visible, setVisible] = useState(false);

  const { log, runResult } = useFlowEnviromentContext()

  return <Popover
    trigger="custom"
    position="bottomLeft"
    closeOnEsc
    visible={visible}
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
      // count={runResult ? runResult : undefined}
      //  position="leftTop" 
      dot={runResult === 'error'}
    >
      <Tooltip
        content={'Log'}>
        <IconButton
          icon={<IconChangelog
            color={runResult ? 'red' : undefined}
            //style={ runResult === 'error' ? {color: 'var(--semi-color-danger)'}: {}}
            style={{ color: runResult ? 'red' : undefined }}
          />}
          type="tertiary"
          theme="borderless"
          color={runResult ? 'red' : undefined}
          onClick={() => {
            setVisible(!visible)
          }}


        //style={{marginRight: 10}}
        />
      </Tooltip>

    </Badge>
  </Popover>




}

export default Log
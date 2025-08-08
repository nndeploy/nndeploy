import { IconExpand } from "@douyinfe/semi-icons"
import { IconChangelog } from "@douyinfe/semi-icons-lab"
import { IconButton, List, Popover, Tooltip } from "@douyinfe/semi-ui"
import { useState } from "react";
import './index.less'
import { useFlowEnviromentContext } from "../../../context/flow-enviroment-context";

const Log: React.FC<any> = (props) => {

  const [visible, setVisible] = useState(false);

  const { log } = useFlowEnviromentContext()

  return <Popover
    trigger="custom"
    position="topLeft"
    closeOnEsc
    visible={visible}
    onVisibleChange={(v) => {
      //onPopupVisibleChange?.(v);
    }}
    onClickOutSide={() => {
      setVisible(false);
    }}
    spacing={20}

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
    <Tooltip
      content={'log'}>
      <IconButton
        icon={<IconChangelog />}
        type="tertiary"
        theme="borderless"
        onClick={() => {
          setVisible(!visible)
        }}
      />
    </Tooltip>
  </Popover>

}

export default Log
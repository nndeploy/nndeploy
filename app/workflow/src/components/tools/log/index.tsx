import { IconExpand } from "@douyinfe/semi-icons"
import { IconChangelog } from "@douyinfe/semi-icons-lab"
import { IconButton, Popover, Tooltip } from "@douyinfe/semi-ui"
import { useState } from "react";
import './index.less'

const Log: React.FC<any> = (props) => {

  const [visible, setVisible] = useState(false);

  return <Tooltip
    content={'log'}>

    <Popover
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
        <h1>log content</h1>
      </div>

    }


    >

      <IconButton
        icon={<IconChangelog />}
        type="tertiary"
        theme="borderless"
        onClick={() => { 
          setVisible(!visible)
         }}
      />
    </Popover>
  </Tooltip>
}

export default Log
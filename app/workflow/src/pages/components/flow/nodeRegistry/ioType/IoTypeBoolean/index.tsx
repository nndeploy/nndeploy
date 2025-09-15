import { Checkbox, InputNumber, Switch, TextArea } from "@douyinfe/semi-ui";
import classNames from "classnames";

import styles from './index.module.scss'
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";


interface IoTypeStringProps {
  value: boolean;
  direction: 'input' | 'output';
  onChange: (value: boolean) => void;
}
const IoTypeBoolean: React.FC<IoTypeStringProps> = (props) => {
  const { value, onChange, direction } = props;



  const { runInfo } = useFlowEnviromentContext()
  const { outputResource } = runInfo

  const { node } = useNodeRender();
  const form = getNodeForm(node)!

  if (!form) {
    console.log('form', form)
    let j = 0
    return <></>
  } else {
    // console.log('form', form)
  }


  const nodeName = form.getValueIn('name_')

  function checkOutputNeedShow() {

    const needShow = direction == 'output' && outputResource.content.hasOwnProperty(nodeName)  //path_.includes('&time=')

    return needShow
  }

  const isOutputNeedShow = checkOutputNeedShow()


  return (
    <div className={styles["io-type-container"]}>
      {
        direction === 'input' ?

          <div className={classNames(styles['io-type-input-container'])}>
            <Switch checked={!!value}


              onChange={(checked, event) => {
                event.stopPropagation();
                onChange(checked)
              }
              }


            // onClick={(event) => {
            //   event.stopPropagation();
            // }} 
            />
          </div>
          :
          <div className={classNames(styles['io-type-output-container'], { [styles.show]: isOutputNeedShow })}>
            <Switch checked={!!value} disabled={true}

            />

          </div>
      }
    </div>
  )
}

export default IoTypeBoolean
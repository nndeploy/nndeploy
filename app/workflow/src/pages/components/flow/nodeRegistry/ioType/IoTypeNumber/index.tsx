import { InputNumber, TextArea } from "@douyinfe/semi-ui";
import classNames from "classnames";

import styles from './index.module.scss'
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";


interface IoTypeStringProps {
  value: number;
  direction: 'input' | 'output';
  onChange: (value: number|string) => void;
}
const IoTypeNumber: React.FC<IoTypeStringProps> = (props) => {
  const { value, onChange, direction } = props;

  const { node } = useNodeRender();

    const { runInfo } = useFlowEnviromentContext()
  const { outputResource } = runInfo

  const form = getNodeForm(node)!

  
  if (!form) {
    console.log('IoTypeNumber form', form)
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
            <InputNumber showClear value={value}


              onChange={onChange}


              onClick={(event) => {
                event.stopPropagation();
              }} />
          </div>
          :
          <div className={classNames(styles['io-type-output-container'], { [styles.show]: isOutputNeedShow })}>
            <InputNumber value={outputResource.content[nodeName] || ''} readOnly
              onClick={(event) => {
                event.stopPropagation();
              }} />

          </div>
      }
    </div>
  )
}

export default IoTypeNumber
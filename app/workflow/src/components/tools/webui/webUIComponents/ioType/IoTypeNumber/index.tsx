import { InputNumber, TextArea } from "@douyinfe/semi-ui";
import classNames from "classnames";

import styles from './index.module.scss'
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import { getIoTypeFieldName, getIoTypeFieldValue, setNodeFieldValue } from "../../../functions";


interface IoTypeStringProps {
  node: any;
  direction: 'input' | 'output';
  onChange: (node: any) => void;
}
const IoTypeNumber: React.FC<IoTypeStringProps> = (props) => {
  const { onChange, direction, node } = props;

  const { runInfo } = useFlowEnviromentContext()
  const { outputResource } = runInfo


  const value = getIoTypeFieldValue(node)

  const nodeName = node.name_
  const ioFieldName = getIoTypeFieldName(node)

  return (
    <div className={styles["io-type-container"]}>
      {
        direction === 'input' ?

          <div className={classNames(styles['io-type-input-container'])}>
            <InputNumber showClear value={value} className={styles['input-number']}


              onChange={(value) => {

                const newNode = setNodeFieldValue(node, ioFieldName, value)

                onChange(newNode)
              }}


            />
          </div>
          :
          <div className={classNames(styles['io-type-output-container'])}>
            <InputNumber
              value={outputResource.content[nodeName] || ''}
              readOnly
            />

          </div>
      }
    </div>
  )
}

export default IoTypeNumber
import { InputNumber, Switch, TextArea } from "@douyinfe/semi-ui";
import classNames from "classnames";

import styles from './index.module.scss'
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import { getIoTypeFieldName, getIoTypeFieldValue, setNodeFieldValue } from "../../../functions";


interface IoTypeStringProps {
  node: any;
  direction: 'input' | 'output';
  onChange: (node: any) => void;
}
const IoTypeBoolean: React.FC<IoTypeStringProps> = (props) => {
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
            <Switch checked={!!value}


              onChange={(checked, event) => {
                const newNode = setNodeFieldValue(node, ioFieldName, checked)

                onChange(newNode)
              }}
            />

          </div>
          :
          <div className={classNames(styles['io-type-output-container'])}>
            <Switch checked={outputResource.content[nodeName] ? true : false}
              disabled={true}

            />
          </div>

      }
    </div >
  )
}

export default IoTypeBoolean
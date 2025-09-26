import { TextArea } from "@douyinfe/semi-ui";
import classNames from "classnames";

import styles from './index.module.scss'
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import { EnumIODataType } from "..";
import { getIoTypeFieldName, getIoTypeFieldValue, setNodeFieldValue } from "../../../functions";


interface IoTypeStringProps {
  // value: string;
  ioDataType: EnumIODataType,
  direction: 'input' | 'output';
  node: any,
  onChange: (node: any) => void;
}
const IoTypeString: React.FC<IoTypeStringProps> = (props) => {
  const { node, onChange, direction, ioDataType } = props;

  const value = getIoTypeFieldValue(node)

  const { runInfo } = useFlowEnviromentContext()
  const { outputResource } = runInfo

  const nodeName = node.name_

  const ioFieldName = getIoTypeFieldName(node)



  function checkOutputNeedShow() {

    const needShow = direction == 'output' && outputResource.content.hasOwnProperty(nodeName)  //path_.includes('&time=')

    return needShow
  }

  const isOutputNeedShow = checkOutputNeedShow()

  return (
    <div className={styles["io-type-container"]}>
      {
        direction === 'input' ?

          <div className={classNames(styles['io-type-input-container'])}

          >
            <textarea
              value={value}
              className={styles['textArea']}


              onChange={(event) => {

                const newNode = setNodeFieldValue(node, ioFieldName, event.target.value)

                onChange(newNode)

              }}


            />
          </div>
          :
          <div className={classNames(styles['io-type-output-container'], { [styles.show]: isOutputNeedShow })}>
            {/* <TextArea value={outputResource.content[nodeName] || ''} readOnly
              onClick={(event) => {
                event.stopPropagation();
              }} /> */}

            <textarea
              value={outputResource.content[nodeName] || ''}
              className={styles['textArea']}
              readOnly
            />



          </div>
      }
    </div>
  )
}

export default IoTypeString
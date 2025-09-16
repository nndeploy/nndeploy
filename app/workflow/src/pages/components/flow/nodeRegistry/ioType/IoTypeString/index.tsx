import { TextArea } from "@douyinfe/semi-ui";
import classNames from "classnames";

import styles from './index.module.scss'
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import { EnumIODataType } from "..";
import CodeEditor from "../../CodeEditor";
import { useRef, useState } from "react";
import { useGetRows } from "../effect";


interface IoTypeStringProps {
  value: string;
  ioDataType: EnumIODataType,
  direction: 'input' | 'output';
  onChange: (value: string) => void;
}
const IoTypeString: React.FC<IoTypeStringProps> = (props) => {
  const { value, onChange, direction, ioDataType } = props;

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


  //const [rows, setRows] = useState(4);
  //const textAreaRef = useRef<HTMLDivElement>(null);

  // const handleResize = (e: React.MouseEvent<HTMLTextAreaElement, MouseEvent>) => {
  //   const lineHeight = 20; // 根据实际行高调整
  //   var temp = textAreaRef.current?.offsetHeight
    
  //   const newRows = Math.floor(temp! / lineHeight);
  //   setRows(newRows);
  // };

  const [ textAreaContainaerRef, rows] = useGetRows(20);




  return (
    <div className={styles["io-type-container"]}>
      {
        direction === 'input' ?

          <div className={classNames(styles['io-type-input-container'])} ref={textAreaContainaerRef} >
            <TextArea showClear value={value}
              className={styles['textArea']}
              

              onChange={(value, event) => {
                //handleResize(event)
                onChange(value)
              }}

              rows={rows}

              onClick={(event) => {
                event.stopPropagation();
              }} />
            {/* <CodeEditor value={value} onChange={onChange} ioDataType={ioDataType} direction={"input"} /> */}
          </div>
          :
          <div className={classNames(styles['io-type-output-container'], { [styles.show]: isOutputNeedShow })}>
            <TextArea value={outputResource.content[nodeName] || ''} readOnly
              onClick={(event) => {
                event.stopPropagation();
              }} />

            {/* <CodeEditor value={value} onChange={onChange} ioDataType={ioDataType} direction={"output"} /> */}

          </div>
      }
    </div>
  )
}

export default IoTypeString
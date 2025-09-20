import { TextArea } from "@douyinfe/semi-ui";
import classNames from "classnames";

import styles from './index.module.scss'
import { getNodeForm, useNodeRender, usePlayground } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import { EnumIODataType } from "..";
import CodeEditor from "../../CodeEditor";
import { useEffect, useRef, useState } from "react";
import { useGetRows } from "../effect";
import { DragArea } from "../../drag-area";


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

  //const [textAreaContainaerRef, rows] = useGetRows(20);



  const { node, selected: focused, selectNode } = useNodeRender();

  const playground = usePlayground();

  const [active, setActive] = useState(false);

  // useEffect(() => {
  //   // 当编辑器失去焦点时，取消激活状态
  //   if (!focused) {
  //     setActive(false);
  //   }
  // }, [focused]);


  const form = getNodeForm(node)!





  if (!form) {
    console.log('IoTypeString form', form)
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


  const handleMouseDown = (mouseDownEvent: React.MouseEvent) => {
    if (active) {
      return;
    }
    mouseDownEvent.preventDefault();
    mouseDownEvent.stopPropagation();

    selectNode(mouseDownEvent);
    playground.node.focus(); // 防止节点无法被删除

    const startX = mouseDownEvent.clientX;
    const startY = mouseDownEvent.clientY;

    const handleMouseUp = (mouseMoveEvent: MouseEvent) => {

      // mouseDownEvent.preventDefault();
      // mouseDownEvent.stopPropagation();

      const deltaX = mouseMoveEvent.clientX - startX;
      const deltaY = mouseMoveEvent.clientY - startY;
      // 判断是拖拽还是点击
      const delta = 5;
      if (Math.abs(deltaX) < delta && Math.abs(deltaY) < delta) {
        // 点击后隐藏
        setActive(true);
      }
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('click', handleMouseUp);
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('click', handleMouseUp);
  };





  return (
    <div className={styles["io-type-container"]}>
      {
        direction === 'input' ?

          <div className={classNames(styles['io-type-input-container'])}
          //ref={textAreaContainaerRef} 
          >
            {/* <TextArea showClear value={value}
              className={styles['textArea']}


              onChange={(value, event) => {
                //handleResize(event)

                let j = 0;
                j = j+1
                onChange(value)
              }}

              rows={rows}

              onClick={(event) => {
                event.stopPropagation();
              }} /> */}
            <textarea
              value={value}
              className={styles['textArea']}


              onChange={(event) => {
                //handleResize(event)

                let j = 0;
                j = j + 1
                onChange(event.target.value)
              }}
              onClick={(event) => {
                event.stopPropagation();
              }}
              onBlur={() => {
                setActive(false)
              }}

            />

            <div
              onMouseDown={handleMouseDown}
              onClick={(mouseDownEvent) => {
                mouseDownEvent.preventDefault();
                mouseDownEvent.stopPropagation();
              }}
              className={classNames(styles['workflow-input-content-drag-area'], { [styles.hidden]: active })}
              >
              {/* <DragArea
                style={{
                  position: 'relative',
                  width: '100%',
                  height: '100%',
                }}

                stopEvent={false}
              /> */}
            </div>




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
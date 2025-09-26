import { TextArea, Upload, Typography, Input, Button } from "@douyinfe/semi-ui";
import { BeforeUploadProps } from "@douyinfe/semi-ui/lib/es/upload";
import { useEffect, useRef, useState } from "react";
import { apiGetFileContent, apiOtherFileSave } from "./api";
import { IconDelete, IconDownload, IconFile, IconFolderOpen, IconPlus, IconSearch } from "@douyinfe/semi-icons";
import styles from './index.module.scss'
import classNames from "classnames";
import { useGetFileInfo } from "./effect";
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import request from "../../../../../../request";
import { getIoTypeFieldName, getIoTypeFieldValue, setNodeFieldValue } from "../../../functions";

const { Text } = Typography;


interface IoTypeTextFileProps {
  node: any;
  direction: 'input' | 'output';
  onChange: (node: any) => void;
}
const IoTypeBinary: React.FC<IoTypeTextFileProps> = (props) => {
  const { onChange, direction, node } = props;



  const { runInfo } = useFlowEnviromentContext()

  const { result: runResult } = runInfo

  const value = getIoTypeFieldValue(node)

  const ioFieldName = getIoTypeFieldName(node)



  // const dropZoneRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);


  useEffect(() => {


    if (direction != 'input') {
      return
    }
    fileInputRef.current?.addEventListener('change', handleFiles);
  }, [fileInputRef.current, direction])

  const fileInfo = useGetFileInfo(value, direction, runResult)


  async function handleFiles(e: any) {
    const files: File[] = e.dataTransfer?.files || e.target.files;


    const formData: FormData = new FormData()
    formData.append('file', files[0])

    const response = await apiOtherFileSave(formData)
    if (response.flag === 'success') {
      const newNode = setNodeFieldValue(node, ioFieldName, response.result.saved_path)

      onChange(newNode)
    }


  }



  function onDownload(event: any) {
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    const url = `/api/download?file_path=${value}`
    request.download(url, {}, {}, undefined)

    //onChange('')
  }


  function getdownloadFileName() {
    if (fileInfo && fileInfo.filename) {
      return fileInfo.filename
    }
    if (value) {
      const parts = (value as string).split('/')
      return parts[parts.length - 1]
    }
    return ''

  }

  const downloadFileName = getdownloadFileName()


  return (
    <div className={styles["io-type-container"]}>
      {
        direction === 'input' ?

          <div className={classNames(styles['io-type-input-container']
            //, { [styles.show]: direction === 'input' }
          )}>
            <div className={classNames(styles["upload-area"])} onClick={event => {
              event.stopPropagation();
              event.nativeEvent.stopImmediatePropagation();


            }}>
              <div className={classNames(styles["binary-container"])}>


                <Input readOnly suffix={<IconFolderOpen
                  onClick={event => {
                    event.stopPropagation();
                    event.nativeEvent.stopImmediatePropagation();

                    if (fileInputRef.current) {
                      fileInputRef.current.click();
                    }
                  }} />}
                  style={{ cursor: 'default!important' }}
                  onFocus={(e) => e.target.style.caretColor = 'transparent'}

                  value={fileInfo.filename ?? 'No file selected'}

                  showClear={true}
                  onClear={() => onChange('')}
                  onClick={event => {
                    event.stopPropagation();

                    event.nativeEvent.stopImmediatePropagation();

                    if (fileInputRef.current) {
                      fileInputRef.current.click();
                    }
                  }} />


                <input type="file" id="fileInput" hidden ref={fileInputRef} />

              </div>
            </div>
          </div>
          :
          <div className={classNames(styles['io-type-output-container']

          )}>

            <div className={classNames(styles["binary-container"])}>
              {
                value && fileInfo &&
                <div className={classNames(styles['file-item-info'])} >

                  <span className={styles.fileName}>{downloadFileName} </span>
                  {/* <span className={styles.fileSize}>{fileInfo.size} </span> */}
                  <Button theme='light'
                    disabled={runResult != 'success'}
                    type='primary'
                    onClick={onDownload}
                    icon={<IconDownload />}
                    className={styles.download}>
                  </Button>

                </div>
              }

            </div>

          </div>
      }
    </div>

  );

}

export default IoTypeBinary
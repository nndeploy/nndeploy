import { TextArea, Upload, Typography, Input, Button, Spin } from "@douyinfe/semi-ui";
import { BeforeUploadProps } from "@douyinfe/semi-ui/lib/es/upload";
import { useEffect, useRef, useState } from "react";
import { apiGetFileContent, apiOtherFileSave } from "./api";
import { IconDownload, IconFolderOpen, IconPlus, IconSearch } from "@douyinfe/semi-icons";
import styles from './index.module.scss'
import classNames from "classnames";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import request from "../../../../../../request";
import { getIoTypeFieldValue } from "../../../functions";

const { Text } = Typography;


interface IoTypeTextFileProps {
  node: any;
  direction: 'input' | 'output';
  onChange: (node: string) => void;
}
const IoTypeTextFile: React.FC<IoTypeTextFileProps> = (props) => {
  const { onChange, direction, node } = props;

  const value = getIoTypeFieldValue(node)

  const [fileContent, setFileContent] = useState('');

  const [fileName, setFileName] = useState('')

  const { runInfo } = useFlowEnviromentContext()

  const { result: runResult } = runInfo

  async function getFileContent(filePath: string) {

    const response = await apiGetFileContent(filePath)
    if (response.flag != 'success') {
      return
    }

    setFileContent(response.result)
  }

  useEffect(() => {
    if (!value) {
      return
    }

    if (direction === 'input') {
      getFileContent(value)
      return
    } else if (direction === 'output') {
      if (runResult == 'success') {
        getFileContent(value)
      } else {
        setFileContent('')
      }
    }

  }, [value, runResult])


  const dropZoneRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  async function handleFiles(e: any) {
    const files: File[] = e.dataTransfer?.files || e.target.files;

    // 文件验证
    // if (files.length > 5) {
    //   alert('最多上传5个文件');
    //   return;
    // }

    const formData: FormData = new FormData()
    formData.append('file', files[0])

    const response = await apiOtherFileSave(formData)
    if (response.flag === 'success') {
      onChange(response.result.saved_path)
    }

    setFileName(files[0].name)

  }

  useEffect(() => {
    // if (dropZoneRef.current) {


    //   (['dragenter', 'dragover'] as Array<keyof HTMLElementEventMap>).forEach((event: keyof HTMLElementEventMap) => {
    //     dropZoneRef.current?.addEventListener(event, (e) => {
    //       e.preventDefault();
    //       dropZoneRef.current?.classList.add('dragover');
    //     });
    //   });

    //   (['dragleave', 'drop'] as Array<keyof HTMLElementEventMap>).forEach((event: keyof HTMLElementEventMap) => {
    //     dropZoneRef.current?.addEventListener(event, (e) => {
    //       e.preventDefault();
    //       dropZoneRef.current?.classList.remove('dragover');
    //     });
    //   });

    //   // 文件选择处理
    //   dropZoneRef.current?.addEventListener('drop', handleFiles);
    //   fileInputRef.current?.addEventListener('change', handleFiles);


    // }
    fileInputRef.current?.addEventListener('change', handleFiles);
  }, [fileInputRef.current])



  function onChooseFile(event: any) {
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();

    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  function getdownloadFileName() {

    if (value) {
      const parts = (value as string).split('/')
      return parts[parts.length - 1]
    }
    return ''

  }

  const downloadFileName = getdownloadFileName()

  function onDownload(event: any) {
    event.stopPropagation();
    event.nativeEvent.stopImmediatePropagation();
    const url = `/api/download?file_path=${value}`
    request.download(url, {}, {}, undefined)

    //onChange('')
  }




  return (
    <div className={styles["io-type-container"]}>
      {
        direction === 'input' ?

          <div className={classNames(styles['io-type-input-container']
            //, { [styles.show]: direction === 'input' }
          )}>
            <div className={classNames(styles["upload-area"])}>

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

                value={fileName}

                showClear={false}
                onClick={event => {
                  event.stopPropagation();

                  event.nativeEvent.stopImmediatePropagation();

                  if (fileInputRef.current) {
                    fileInputRef.current.click();
                  }
                }} />


              <input type="file" id="fileInput" hidden ref={fileInputRef} />


              <div className={styles['file-content']}>
                {
                  fileContent
                }

              </div>
            </div>
          </div>
          :
          <div className={classNames(styles['io-type-output-container'])} >

            <div className={styles['file-content']}>
              {runInfo.isRunning  ?
                <div className={styles.spin}>
                  <Spin size="middle" />
                </div> : fileContent}
            </div>
            <div className={classNames(styles['file-download-info'])} >

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


          </div>

      }

    </div>

  );

}

export default IoTypeTextFile
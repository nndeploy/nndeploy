import { Upload, Typography, Image, Modal, Button, Spin } from "@douyinfe/semi-ui";
import { customRequestArgs, FileItem } from "@douyinfe/semi-ui/lib/es/upload";
import { IconDelete, IconEyeOpened, IconPlus } from "@douyinfe/semi-icons";
import { apiImageSave } from "./api";
import styles from './index.module.scss'
import { useEffect, useRef, useState } from "react";
import classNames from "classnames";
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import { getIoTypeFieldName, getIoTypeFieldValue, setNodeFieldValue } from "../../../functions";
import lodash from 'lodash'
import { Map } from 'immutable';
import UiParams from "../../uiParams";

const { Text } = Typography;


export interface IoTypeProps {
  // value: any;
  direction: 'input' | 'output';
  onChange: (node: any) => void;
  node: any,
  className: string
}
const IoTypeImage: React.FC<IoTypeProps> = (props) => {
  const {
    //value,
    onChange, direction, node, className } = props;


  const value = getIoTypeFieldValue(node)


  const [previewVisible, setPreviewVisible] = useState(false);

  const { runInfo, element } = useFlowEnviromentContext()
  const { result: runResult } = runInfo


  const dropZoneRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ioFieldName = getIoTypeFieldName(node)


  useEffect(() => {

    if (direction === 'input') {


      if (dropZoneRef.current) {

        (['dragenter', 'dragover'] as Array<keyof HTMLElementEventMap>).forEach((event: keyof HTMLElementEventMap) => {
          dropZoneRef.current?.addEventListener(event, (e) => {
            e.preventDefault();
            dropZoneRef.current?.classList.add('dragover');
          });
        });

        (['dragleave', 'drop'] as Array<keyof HTMLElementEventMap>).forEach((event: keyof HTMLElementEventMap) => {
          dropZoneRef.current?.addEventListener(event, (e) => {
            e.preventDefault();
            dropZoneRef.current?.classList.remove('dragover');
          });
        });

        // 文件选择处理
        dropZoneRef.current?.addEventListener('drop', handleFiles);
        fileInputRef.current?.addEventListener('change', handleFiles);


      }
    }
  }, [])


  const previewUrl = `/api/preview?file_path=${value}&time=${runInfo.time}&returnMimeType=image`



  async function handleFiles(e: any) {
    const files: File[] = e.dataTransfer?.files || e.target.files;
    const formData: FormData = new FormData()
    formData.append('file', files[0])

    const response = await apiImageSave(formData)
    if (response.flag === 'success') {

      const newNode = setNodeFieldValue(node, ioFieldName, response.result.saved_path)

      onChange(newNode)
    }

  }




  return (
    <>
      <div className={classNames(styles["io-type-container"])}>

        {
          direction === 'input' ?

            <div className={classNames(styles['io-type-input-container'])}>

              {/* <UiParams {...props} className={classNames(styles['ui-params'])}/> */}


              <div className={classNames(styles["upload-area"],)}
                ref={dropZoneRef}
                onClick={(event) => {
                  event.stopPropagation();
                  if (fileInputRef.current) {
                    fileInputRef.current.click();
                  }
                }}>

                <div className={classNames(styles["image-container"], { [styles.hasImage]: !!value })}>
                  {
                    value ? <>
                      <IconDelete size="extra-large" className={styles["icon-delete"]}
                        onClick={(event) => {
                          debugger
                          event.stopPropagation();
                          event.nativeEvent.stopImmediatePropagation();
                          const newNode = setNodeFieldValue(node, ioFieldName, '')

                          onChange(newNode)
                        }}
                      />

                      <IconEyeOpened size="extra-large" className={styles["icon-view"]} onClick={(event) => {
                        debugger
                        event.stopPropagation();
                        setPreviewVisible(oldValue => {

                          return true
                        })
                      }} />
                      <img src={previewUrl} alt=""
                      //style={{ maxWidth: '100%' }}
                      />
                    </> : <IconPlus size="extra-large" />
                  }

                </div>



                <input type="file" id="fileInput" hidden ref={fileInputRef} />
              </div>
            </div>
            :
            <div className={classNames(styles['io-type-output-container'])}>

              {/* <UiParams {...props} className={classNames(styles['ui-params'])}/> */}
              <div className={classNames(styles["image-container"], { [styles.hasImage]: runResult == 'success' && !!value })}>
                <IconEyeOpened size="extra-large" className={styles["icon-view"]} onClick={(event) => {
                  debugger
                  event.stopPropagation();
                  setPreviewVisible(true)
                }} />
                {

                  runInfo.isRunning ? <div className={styles.spin}>
                    <Spin size="middle" />
                  </div> :
                    runResult == 'success' ? <img src={previewUrl} alt="" /> : <></>
                }

              </div>
            </div>
        }



      </div>
      <Modal
        title="Preview"
        visible={previewVisible}
        closeOnEsc={true}
        centered={true}
        fullScreen={false}
        className="model-flex"
        width={'95%'}
        height={'95%'}
        getPopupContainer={() => {
          //return document.body
          return element?.current!
        }}
        // width={'90%'}
        // height={'90%'}
        onCancel={() => {
          setPreviewVisible(false)
        }
        }
        footer={[
          <Button key="cancel" onClick={() => setPreviewVisible(false)}>
            Close
          </Button>,
        ]}
      >
        <div className={classNames('model-content')}>
          <div className={classNames("preview-image-modal")}>

            <img src={previewUrl} style={{ maxWidth: '100%' }} alt="" />
          </div>
        </div>

      </Modal>
    </>
  );

}

export default IoTypeImage
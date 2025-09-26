import { Upload, Typography, Image, Modal, Button } from "@douyinfe/semi-ui";
import { customRequestArgs, FileItem } from "@douyinfe/semi-ui/lib/es/upload";
import { IconDelete, IconEyeOpened, IconPlus } from "@douyinfe/semi-icons";
import { apiImageSave } from "./api";
import styles from './index.module.scss'
import { useEffect, useRef, useState } from "react";
import classNames from "classnames";
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";

const { Text } = Typography;


interface IoTypeTextFileProps {
  value: any;
  direction: 'input' | 'output';
  onChange: (value: string) => void;
}
const IoTypeImage: React.FC<IoTypeTextFileProps> = (props) => {
  const { value, onChange, direction } = props;


  const [previewVisible, setPreviewVisible] = useState(false);

  const { runInfo, element } = useFlowEnviromentContext()
  const { result: runResult } = runInfo


  const dropZoneRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);




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
      onChange(response.result.saved_path)
    }

  }




  return (
    <>
      <div className={styles["io-type-container"]}>

        {
          direction === 'input' ?

            <div className={classNames(styles['io-type-input-container'])}>
              <div className={classNames(styles["upload-area"], { [styles.hasImage]: !!value })}
                ref={dropZoneRef}
                onClick={(event) => {
                  event.stopPropagation();
                  if (fileInputRef.current) {
                    fileInputRef.current.click();
                  }
                }}>
                {value ?
                  <div className={styles["image-container"]}>

                    <IconDelete size="extra-large" className={styles["icon-delete"]}
                      onClick={(event) => {
                        debugger
                        event.stopPropagation();
                        event.nativeEvent.stopImmediatePropagation();
                        onChange('')
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
                  </div>
                  : <>
                    <IconPlus size="extra-large" />

                  </>
                }

                <input type="file" id="fileInput" hidden ref={fileInputRef} />
              </div>
            </div>
            :
            <div className={classNames(styles['io-type-output-container'], { [styles.show]: runResult == 'success' && !!value })}>
              <div className={styles["image-container"]}>
                <IconEyeOpened size="extra-large" className={styles["icon-view"]} onClick={(event) => {
                  debugger
                  event.stopPropagation();
                  setPreviewVisible(true)
                }} />
                <img src={previewUrl} style={{ maxWidth: '100%' }} alt="" />
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
        <div className={classNames( 'model-content')}>
          <div className={classNames("preview-image-modal")}>

            <img src={previewUrl} style={{ maxWidth: '100%' }} alt=""/>
          </div>
        </div>

      </Modal>
    </>
  );

}

export default IoTypeImage
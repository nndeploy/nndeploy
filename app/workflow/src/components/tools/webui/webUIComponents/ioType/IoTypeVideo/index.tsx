import { Upload, Typography, Image, Modal, Button, VideoPlayer, Input, Spin } from "@douyinfe/semi-ui";
import { customRequestArgs, FileItem } from "@douyinfe/semi-ui/lib/es/upload";
import { IconDelete, IconEyeOpened, IconFolderOpen, IconPlus } from "@douyinfe/semi-icons";
import { apiVideoSave } from "./api";
import styles from './index.module.scss'
import { useEffect, useRef, useState } from "react";
import classNames from "classnames";
import { getNodeForm, useNodeRender } from "@flowgram.ai/free-layout-editor";
import { useFlowEnviromentContext } from "../../../../../../context/flow-enviroment-context";
import { useGetFileInfo } from "./effect";
import { getIoTypeFieldName, getIoTypeFieldValue, setNodeFieldValue } from "../../../functions";

const { Text } = Typography;


interface IoTypeTextFileProps {

  direction: 'input' | 'output';
  node: any;
  onChange: (node: any) => void;
}
const IoTypeVideo: React.FC<IoTypeTextFileProps> = (props) => {
  const { onChange, direction, node } = props;

  const { runInfo, element } = useFlowEnviromentContext()
  const { result: runResult } = runInfo

  const value = getIoTypeFieldValue(node)


  //  const dropZoneRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ioFieldName = getIoTypeFieldName(node)



  const [previewVisible, setPreviewVisible] = useState(false);

  useEffect(() => {

    // if (direction === 'input') {


    //   if (dropZoneRef.current) {

    //     (['dragenter', 'dragover'] as Array<keyof HTMLElementEventMap>).forEach((event: keyof HTMLElementEventMap) => {
    //       dropZoneRef.current?.addEventListener(event, (e) => {
    //         e.preventDefault();
    //         dropZoneRef.current?.classList.add('dragover');
    //       });
    //     });

    //     (['dragleave', 'drop'] as Array<keyof HTMLElementEventMap>).forEach((event: keyof HTMLElementEventMap) => {
    //       dropZoneRef.current?.addEventListener(event, (e) => {
    //         e.preventDefault();
    //         dropZoneRef.current?.classList.remove('dragover');
    //       });
    //     });

    //     // 文件选择处理
    //     dropZoneRef.current?.addEventListener('drop', handleFiles);
    //     fileInputRef.current?.addEventListener('change', handleFiles);


    //   }
    // }
    if (direction != 'input') {
      return
    }

    fileInputRef.current?.addEventListener('change', handleFiles);
  }, [fileInputRef.current, direction])


  // const fileInfo = useGetFileInfo(value, direction,  runResult)




  const previewUrl = `/api/preview?file_path=${value}&time=${runInfo.time}`



  async function handleFiles(e: any) {
    const files: File[] = e.dataTransfer?.files || e.target.files;
    const formData: FormData = new FormData()
    formData.append('file', files[0])

    const response = await apiVideoSave(formData)
    if (response.flag === 'success') {
      //onChange(response.result.saved_path)
      const newNode = setNodeFieldValue(node, ioFieldName, response.result.saved_path)

      onChange(newNode)
    }

  }


  return (
    <>
      <div className={styles["io-type-container"]}>

        {
          direction === 'input' ?

            <div className={classNames(styles['io-type-input-container'])}>
              <div className={classNames(styles["upload-area"], { [styles.hasVideo]: !!value })}

              >
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

                  value={value ? value : 'No file selected'}

                  showClear={true}
                  onClear={() => {
                    fileInputRef!.current!.value = ''

                    const newNode = setNodeFieldValue(node, ioFieldName, '')

                    onChange(newNode)
                  }}
                  onClick={event => {
                    event.stopPropagation();

                    event.nativeEvent.stopImmediatePropagation();

                    if (fileInputRef.current) {
                      fileInputRef.current.click();
                    }
                  }} />


                <input type="file" id="fileInput" hidden ref={fileInputRef} />
              </div>


              <div className={styles["video-container"]} >
                {/* 
                  <IconDelete size="extra-large" className={styles["icon-delete"]}
                    onClick={(event) => {
                      debugger
                      event.stopPropagation();
                      event.nativeEvent.stopImmediatePropagation();
                      onChange('')
                    }}
                  /> */}

                {value && <>

                  <IconEyeOpened size="extra-large" className={styles["icon-view"]} onClick={(event) => {
                    debugger
                    event.stopPropagation();
                    setPreviewVisible(oldValue => {
                      return true
                    })
                  }} />
                  <VideoPlayer
                    //height={200}
                    controlsList={
                      //['play', 'time', 'volume', 'playbackRate', 'fullscreen']
                      ['volume']
                    }
                    clickToPlay={false}
                    autoPlay={false}



                    src={`/api/preview?file_path=${value}`}
                  //poster={'https://lf3-static.bytednsdoc.com/obj/eden-cn/ptlz_zlp/ljhwZthlaukjlkulzlp/poster2.jpeg'}
                  />
                </>
                }
              </div>
            </div>

            :
            <div className={classNames(styles['io-type-output-container'], { [styles.show]: runResult == 'success' && !!value })}>


              <div className={styles["video-container"]}>

                {
                  runInfo.isRunning ? <div className={styles.spin}>
                    <Spin size="middle" />
                  </div> :
                    value && runResult == 'success' &&
                    <>
                      <IconEyeOpened size="extra-large" className={styles["icon-view"]} onClick={(event) => {
                        debugger
                        event.stopPropagation();
                        setPreviewVisible(true)
                      }} />
                      <VideoPlayer
                        //height={200}
                        controlsList={
                          //['play', 'time', 'volume', 'playbackRate', 'fullscreen']
                          ['volume']
                        }
                        clickToPlay={true}
                        autoPlay={false}



                        src={`/api/preview?file_path=${value}`}
                      //poster={'https://lf3-static.bytednsdoc.com/obj/eden-cn/ptlz_zlp/ljhwZthlaukjlkulzlp/poster2.jpeg'}
                      />
                    </>
                }
              </div>

            </div>
        }



      </div >
      <Modal
        title="Preview"
        visible={previewVisible}
        closeOnEsc={true}
        centered={true}
        // width={'800px'}
        width={'95%'}
        height={'95%'}
        getPopupContainer={() => {
          //return document.body
          return element?.current!
        }}
        className="model-flex"
        //height={'500px'}
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
          <div className={styles["preview-video-modal"]}>

            <VideoPlayer
              //height={'80%'}
              // height={400}
              controlsList={['play', 'time', 'volume', 'playbackRate', 'fullscreen']}
              clickToPlay={false}
              autoPlay={false}

              src={`/api/preview?file_path=${value}`}
            //poster={'https://lf3-static.bytednsdoc.com/obj/eden-cn/ptlz_zlp/ljhwZthlaukjlkulzlp/poster2.jpeg'}
            />
          </div>
        </div>

      </Modal>
    </>
  );

}

export default IoTypeVideo
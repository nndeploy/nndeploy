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

  const { node } = useNodeRender();
  const form = getNodeForm(node)!
  const nodeName = form.getValueIn('name_')

  const {  runInfo } = useFlowEnviromentContext()
  const {result: runResult } = runInfo






  const [previewVisible, setPreviewVisible] = useState(false);

  const previewUrl = `/api/preview?file_path=${value}&time=${runInfo.time}`

  const dropZoneRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleFiles(e: any) {
    const files: File[] = e.dataTransfer?.files || e.target.files;
    const formData: FormData = new FormData()
    formData.append('file', files[0])

    const response = await apiImageSave(formData)
    if (response.flag === 'success') {
      onChange(response.result.saved_path)
    }

  }

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


  return (<div className={styles["io-type-container"]}>
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
                event.stopPropagation();
                event.nativeEvent.stopImmediatePropagation();
                onChange('')
              }}
            />

            <IconEyeOpened size="extra-large" className={styles["icon-view"]} onClick={(event) => {
              event.stopPropagation();
              setPreviewVisible(true)
            }} />
            <img src={previewUrl} style={{ maxWidth: '100%' }} />
          </div>
          : <>
            <IconPlus size="extra-large" />
            {/* <p>拖放文件到这里或点击上传</p> */}
          </>
        }

        <input type="file" id="fileInput" hidden ref={fileInputRef} />
      </div>
    </div>
    : 
    <div className={classNames(styles['io-type-output-container'], { [styles.show]: runResult == 'success' && !!value })}>
      <div className={styles["image-container"]}>
        <img src={previewUrl} style={{ maxWidth: '100%' }} />
      </div>
    </div>
}

    <Modal
      title="Preview"
      visible={previewVisible}
      onCancel={() => setPreviewVisible(false)}
      footer={[
        <Button key="cancel" onClick={() => setPreviewVisible(false)}>
          Close
        </Button>,
      ]}
    >
      <div className={styles["preview-image-modal"]}>
        <img src={previewUrl} style={{ maxWidth: '100%' }} />
      </div>

    </Modal>
  </div>
  );

}

export default IoTypeImage
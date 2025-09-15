import { TextArea, Upload, Typography, Input } from "@douyinfe/semi-ui";
import { BeforeUploadProps } from "@douyinfe/semi-ui/lib/es/upload";
import { useEffect, useRef, useState } from "react";
import { apiGetFileContent, apiOtherFileSave } from "./api";
import { IconFolderOpen, IconPlus, IconSearch } from "@douyinfe/semi-icons";
import styles from './index.module.scss'
import classNames from "classnames";

const { Text } = Typography;


interface IoTypeTextFileProps {
  value: any;
  onChange: (value: string) => void;
}
const IoTypeTextFile: React.FC<IoTypeTextFileProps> = (props) => {
  const { value, onChange } = props;

  const [fileContent, setFileContent] = useState('');

  const [fileName, setFileName] = useState('')

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

    getFileContent(value)

  }, [value])


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


  return (
    <>
      <div onClick={event => {
        event.stopPropagation();
        event.nativeEvent.stopImmediatePropagation();


      }}>
        {/* <IconPlus size="extra-large"

          onClick={event => {
            event.stopPropagation();
            event.nativeEvent.stopImmediatePropagation();

            if (fileInputRef.current) {
              fileInputRef.current.click();
            }
          }}
        /> */}
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

        {value && fileContent && <div className={styles['preview']} onClick={(event) => {
          event.stopPropagation();
        }}>{fileContent}</div>

        }

      </div>
    </>

  );

}

export default IoTypeTextFile
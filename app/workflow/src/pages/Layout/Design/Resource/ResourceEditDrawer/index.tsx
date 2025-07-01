import { useRef, useState, useEffect } from "react";
import { IResourceEntity, IResourceTreeNodeEntity, ResourceTreeNodeData } from "../entity";
import { Button, Form, Input, Toast, VideoPlayer } from "@douyinfe/semi-ui";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";
import { apiGetResource, apiResourceSave, apiResourceUpload } from "../api";
import "./index.scss";

export interface ResourceEditDrawerProps {
  onSure: (node: IResourceTreeNodeEntity) => void;
  onClose: () => void;
  node: IResourceTreeNodeEntity;
}

const ResourceEditDrawer: React.FC<ResourceEditDrawerProps> = (props) => {

  const {node} = props

  const [entity, setEntity] = useState<IResourceTreeNodeEntity>({ ...props.node });

  const [file, setFile] = useState<File | null>(null);

  const fileRef = useRef<any>()

  const formRef = useRef<FormApi<any>>();

  //  useEffect(() => {
  //       if(props.entity.id){
  //         apiGetResource(props.entity.id).then((res) => {
  //           if(res.flag == "success"){
  //             setEntity(res.result);
  //           }
  //         })
  //       }

  //   }, [props.entity])

  async function onSure() {
    try {

      if (!file) {
        return
      }
      // await formRef!.current!.validate();
      // const formData = formRef!.current!.getValues();
      // console.log("Form Data:", formData);

      const url = `/api/files/${entity.parentId}`

      // const data = {
      //   ...entity, 
      //   ...formData,
      //   id: props.entity.id ?? "",
      // };

      const formData = new FormData();
      formData.append('file', file);

      const response = await apiResourceSave(url, formData);
      if (response.flag == "success") {
        props.onSure(
          {
            id: response.result.filename,
            name: response.result.filename,
            parentId: node.parentId,
            type: 'leaf', 
            entity: response.result
          });
      }

      //Toast.success("add sucess!");
    } catch (error) {
      Toast.error("add fail " + error);
    }
  }

  // async function handleFileUpload() {
  //   if (file) {
  //     const formData = new FormData();
  //     formData.append("file", file);
  //     formData.append("sex", 'man');

  //     var tempFile = {
  //       name: file.name,
  //       type: file.type,


  //     }

  //     try {
  //       const response = await apiResourceUpload(tempFile);

  //       if (response.flag == "success") {

  //         setEntity({ ...entity,  mime: response.result.mime, name: response.result.name })
  //         Toast.success("File uploaded successfully!");
  //       } else {
  //         Toast.error("File upload failed!");
  //       }
  //     } catch (error) {
  //       Toast.error("File upload error: " + error);
  //     }
  //   } else {
  //     Toast.error("No file selected!");
  //   }
  // }

  function onFileChange(event: any) {
    var i = 0;
    var files = fileRef.current.files
    setFile(files[0])
  }

  return (
    <>
      <div className="drawer-content">

        {
          entity.id ? <>
            {entity.parentId?.includes("images") && entity.name ? (
              <div className="image-preview">
                <img src={`/api/preview/${entity.parentId}/${entity.name}`} />
              </div>
            ) : entity.parentId?.includes("videos") && entity.name ? (
              <div className="video-preview">
                <VideoPlayer
                  height={430}
                  src={`/api/preview/${entity.parentId}/${entity.name}`}
                //poster={'https://lf3-static.bytednsdoc.com/obj/eden-cn/ptlz_zlp/ljhwZthlaukjlkulzlp/poster2.jpeg'}
                />
              </div>

            )




              : (
                <></>
              )}

            <div className="fileInfo">
              <div className="item">
                  fileName: {node.entity?.filename}
              </div>
              <div className="item">
                  saved_path: {node.entity?.saved_path}
              </div>
              <div className="item">
                  size: {node.entity?.size}
              </div>
              <div className="item">
                  uploaded_at: {node.entity?.uploaded_at}
              </div>
            </div>
            </> : <>

      <Form
        getFormApi={(formApi) => (formRef.current = formApi)}
        onValueChange={(v) => console.log(v)}
      >
        {/* <Input
          field="name"
          label="name"
          rules={[{ required: true, message: "please input" }]}
        /> */}
        <div style={{ display: "flex", alignItems: "center" }}>
          <Input
            type="file"
            ref={fileRef}
            ///@ts-ignore
            onChange={(e) => onFileChange(e)}
            style={{ marginRight: 10 }}
          />
          {/* <Button onClick={handleFileUpload}>Upload File</Button> */}
        </div>
      </Form>
    </>
}

      </div >
  <div className="semi-sidesheet-footer">
    {!node.id && <Button onClick={() => onSure()}>confirm</Button>}
      
    <Button type="tertiary" onClick={() => props.onClose()}>
      close
    </Button>
  </div>
    </>
  );
};

export default ResourceEditDrawer;

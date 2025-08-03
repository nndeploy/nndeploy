import { useRef, useState, useEffect } from "react";
import { Button, Form, Toast, VideoPlayer } from "@douyinfe/semi-ui";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";
import "./index.scss";
import { INodeEntity } from "../entity";
import { apiGetNodeById, apiNodeSave } from "../Tree/api";
import { JsonSchemaEditor } from "../../components/json-schema-editor";
import { JsonSchema } from "../../components/type-selector/types";

export interface NodeEditDrawerProps {
  onSure: (node: INodeEntity) => void;
  onClose: () => void;
  entity: INodeEntity;
}

const NodeEditDrawer: React.FC<NodeEditDrawerProps> = (props) => {
  const [entity, setEntity] = useState<INodeEntity>({ ...props.entity });

  const [file, setFile] = useState<File | null>(null);

  const fileRef = useRef<any>();

  const formRef = useRef<FormApi<any>>();

  useEffect(() => {
    if (props.entity.key_) {
      apiGetNodeById(props.entity.key_).then((res) => {
        if (res.flag == "success") {
          setEntity(res.result);
        }
      });
    }
  }, [props.entity]);

  async function onSure() {
    try {
      await formRef!.current!.validate();
      const formData = formRef!.current!.getValues();
      console.log("Form Data:", formData);

      const data = {
        ...entity,
        ...formData,
        id: props.entity.key_ ?? "",
      };

      const response = await apiNodeSave(data);
      if (response.flag == "success") {
        props.onSure({ ...response.result });
      }

      //Toast.success("add sucess!");
    } catch (error) {
      Toast.error("add fail " + error);
    }
  }

  async function handleFileUpload() {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("sex", "man");

      var tempFile = {
        name: file.name,
        type: file.type,
      };

      try {
        // const response = await apiResourceUpload(tempFile);
        // if (response.flag == "success") {
        //   setEntity({...entity, url: response.result.url, mime: response.result.mime, name: response.result.name})
        //   Toast.success("File uploaded successfully!");
        // } else {
        //   Toast.error("File upload failed!");
        // }
      } catch (error) {
        Toast.error("File upload error: " + error);
      }
    } else {
      Toast.error("No file selected!");
    }
  }

  function onFileChange(event: any) {
    var i = 0;
    var files = fileRef.current.files;
    setFile(files[0]);
  }

  function onSchemeChange(schema: JsonSchema) {
    setEntity({ ...entity, schema });
  }

  return (
    <div className="semi-sidesheet-body">
      <div className="drawer-content">
        <Form
          getFormApi={(formApi) => (formRef.current = formApi)}
          initValues={entity}
          onValueChange={(v) => console.log(v)}
        >
          <Form.Input
            field="name"
            label="name"
            rules={[{ required: true, message: "please input" }]}
          />
          {/* <div style={{ display: "flex", alignItems: "center" }}>
            <Input
              type="file"
              ref={fileRef}
              ///@ts-ignore
              onChange={(e) => onFileChange(e)}
              style={{ marginRight: 10 }}
            />
            <Button onClick={handleFileUpload}>Upload File</Button>
          </div> */}
          <JsonSchemaEditor
            value={entity.schema}
            onChange={(value) => onSchemeChange(value)}
          />
        </Form>
      </div>
      <div className="semi-sidesheet-footer">
        <Button onClick={() => onSure()}>confirm</Button>
        <Button type="tertiary" onClick={() => props.onClose()}>
          close
        </Button>
      </div>
    </div>
  );
};

export default NodeEditDrawer;

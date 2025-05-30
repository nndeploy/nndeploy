import { Form, Button, Toast } from "@douyinfe/semi-ui";
const { Input } = Form;

import { apiGetWorkFlow, apiWorkFlowSave } from "../api";
import { IWorkFlowEntity, IWorkFlowTreeNodeEntity } from "../entity";
import { useEffect, useRef, useState } from "react";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";

export interface WorkFlowEditDrawerProps {
  onSure: (node: IWorkFlowTreeNodeEntity) => void;
  onClose: () => void;
  entity: IWorkFlowTreeNodeEntity;
}

const WorkFlowEditDrawer: React.FC<WorkFlowEditDrawerProps> = (props) => {
  const formRef = useRef<FormApi<any>>();

  const [entity, setEntity] = useState<IWorkFlowEntity>({
    ...props.entity,
    content: {
      nodes: [],
      edges: [],
    },
  });

  useEffect(() => {
      if(props.entity.id){
        apiGetWorkFlow(props.entity.id).then((res) => {
          if(res.flag == "success"){
            //setEntity(res.result);
          }
        })
      }
    
  }, [props.entity])

  async function onSure() {
    try {
      await formRef!.current!.validate();
      const formData = formRef!.current!.getValues();
      console.log("Form Data:", formData);

      const data = {
        ...entity,
        ...formData
      };

      const response = await apiWorkFlowSave(data);
      if (response.flag == "success") {
        props.onSure( {...response.result, type: 'leaf'} );
      }

      Toast.success("add sucess!");
    } catch (error) {
      Toast.error("add fail " + error);
    }
  }

  return (
    <>
      <div className="drawer-content">
        <Form
          //style={{ padding: 10, width: "100%" }}
          getFormApi={(formApi) => (formRef.current = formApi)}
          onValueChange={(v) => console.log(v)}
        >
          <Input
            field="name"
            label="name"
            rules={[{ required: true, message: "please input" }]}
          />
        </Form>
      </div>
      <div className="semi-sidesheet-footer">
        <Button onClick={() => onSure()}>confirm</Button>
        <Button type="tertiary" onClick={() => props.onClose()}>
          close
        </Button>
      </div>
    </>
  );
};

export default WorkFlowEditDrawer;

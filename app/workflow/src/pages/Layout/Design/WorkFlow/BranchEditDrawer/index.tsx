import { Form, Button, Toast } from "@douyinfe/semi-ui";
const { Input } = Form;

import { apiWorkFlowBranchSave } from "../api";
import { IWorkFlowTreeNodeEntity, WorkFlowTreeNodeData } from "../entity";
import { useRef } from "react";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";

export interface WorkFlowEditDrawerProps {
  onSure: (node: IWorkFlowTreeNodeEntity) => void;
  onClose: () => void;
  entity: IWorkFlowTreeNodeEntity;
}

const BranchEditDrawer: React.FC<WorkFlowEditDrawerProps> = (props) => {
  const formRef = useRef<FormApi<any>>();

  async function onSure() {
    try {
      await formRef!.current!.validate();
      const formData = formRef!.current!.getValues();
      console.log("Form Data:", formData);

      const entity = {
        ...formData,
        id: props.entity.id ? props.entity.id: "",
        parentId: props.entity.parentId ?? "",
      };

      const response = await apiWorkFlowBranchSave(entity);
      if (response.flag == "success") {
        props.onSure(response.result);
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

export default BranchEditDrawer;

import { Form, Button, Toast } from "@douyinfe/semi-ui";
const { Input } = Form;

import { apiResourceBranchSave } from "../api";
import { ResourceTreeNodeData } from "../entity";
import { useRef } from "react";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";

export interface BranchDrawerProps {
  onSure: (node: ResourceTreeNodeData) => void;
  onClose: () => void;
  parentNode?: ResourceTreeNodeData;
  currentNode?: ResourceTreeNodeData;
}

const BranchDrawer: React.FC<BranchDrawerProps> = (props) => {
  const { parentNode, currentNode } = props;

  const formRef = useRef<FormApi<any>>();

  async function onSure() {
    try {
      await formRef!.current!.validate();
      const formData = formRef!.current!.getValues();
      console.log("Form Data:", formData);

      const entiry = {
        ...formData,
        id: currentNode?.key ?? "",
        parentId: parentNode?.key ?? "",
      };

      const response = await apiResourceBranchSave(entiry);
      if (response.flag == "success") {
        props.onSure({ ...response.result, type: "branch" });
      }

      Toast.success("add sucess!");
    } catch (error) {
      Toast.error("add fail " + error);
    }
  }

  return (
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
      <div className="drawer-footer">
        <Button onClick={() => onSure()}>sure</Button>
        <Button type="tertiary" onClick={() => props.onClose()}>
          close
        </Button>
      </div>
    </div>
  );
};

export default BranchDrawer;

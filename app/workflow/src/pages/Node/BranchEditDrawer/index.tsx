import { Form, Button, Toast } from "@douyinfe/semi-ui";
const { Input } = Form;
import { useRef } from "react";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";
import { apiNodeBranchSave } from "../Tree/api";
import { INodeBranchEntity } from "../entity";

export interface BranchEditDrawerProps {
  onSure: (node: INodeBranchEntity) => void;
  onClose: () => void;
  entity: INodeBranchEntity;
}

const BranchEditDrawer: React.FC<BranchEditDrawerProps> = (props) => {
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

      const response = await apiNodeBranchSave(entity);
      if (response.flag == "success") {
        props.onSure({...response.result});
      }

      Toast.success("add sucess!");
    } catch (error) {
      Toast.error("add fail " + error);
    }
  }

  return (
    <div className="semi-sidesheet-body">
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
        <Button onClick={() => onSure()}>sure</Button>
        <Button type="tertiary" onClick={() => props.onClose()}>
          close
        </Button>
      </div>
    </div>
  );
};

export default BranchEditDrawer;

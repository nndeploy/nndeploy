import { Form, Button, Toast } from "@douyinfe/semi-ui";
const { Input, TreeSelect } = Form;

import { useRef } from "react";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";
import {  useGetWorkflowTree } from "../../../Layout/Design/WorkFlow/effect";
import { IBusinessNode, IWorkFlowEntity } from "../../../Layout/Design/WorkFlow/entity";
import { apiWorkFlowSave } from "../../../Layout/Design/WorkFlow/api";
import { designDataToBusinessData } from "./functions";
import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";

export interface BranchEditDrawerProps {
  onSure: (node: IWorkFlowEntity) => void;
  onClose: () => void;
  entity: IWorkFlowEntity;
}

const FlowSaveDrawer: React.FC<BranchEditDrawerProps> = (props) => {
  const formRef = useRef<FormApi<any>>();
  const flowEnviroment = useFlowEnviromentContext();

  // const { treeData } = useGetWorkflowBranch()

  async function onSure() {
    try {
      await formRef!.current!.validate();
      const formData = formRef!.current!.getValues();
      //console.log("Form Data:", formData);

        const businessContent = designDataToBusinessData(
              props.entity.designContent, 
              flowEnviroment.graphTopNode
            );

      const data: IWorkFlowEntity = {
        ...props.entity, 
        businessContent, 
        ...formData,

      };

      businessContent.name_ = formData['name']

      const response = await apiWorkFlowSave(businessContent);
      if (response.flag == "success") {
        props.onSure(data);
      }

      Toast.success("save sucess!");
    } catch (error) {
      Toast.error("save fail " + error);
    }
  }

  const initValues = {
    parentId: props.entity.parentId,
    name: props.entity.name,
  }

  return (
    <>
      <div className="drawer-content">
        <Form
        initValues={initValues}
          //style={{ padding: 10, width: "100%" }}
          getFormApi={(formApi) => (formRef.current = formApi)}
          onValueChange={(v) => console.log(v)}
        >
           {/* <TreeSelect
            label="parent"
            field="parentId"
            showClear
           // rules={[{ required: true, message: "please input" }]}
            style={{ width: '100%' }}
            dropdownStyle={{ maxHeight: 400, overflow: 'auto' }}
            treeData={treeData}
            placeholder="please choose"
        /> */}
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

export default FlowSaveDrawer;

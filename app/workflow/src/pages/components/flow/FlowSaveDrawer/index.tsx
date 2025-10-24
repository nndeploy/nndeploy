import { Form, Button, Toast } from "@douyinfe/semi-ui";
const { Input, TreeSelect } = Form;

import { useRef } from "react";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";
import { IWorkFlowEntity } from "../../../Layout/Design/WorkFlow/entity";
import { apiWorkFlowSave } from "../../../Layout/Design/WorkFlow/api";

import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
import { EnumFlowType } from "../../../../enum";
import { useClientContext } from "@flowgram.ai/free-layout-editor";
import { designDataToBusinessData } from "./toBusiness";

export interface BranchEditDrawerProps {
  onSure: (node: IWorkFlowEntity) => void;
  onClose: () => void;
  flowType: EnumFlowType;

  entity: IWorkFlowEntity;
}

const FlowSaveDrawer: React.FC<BranchEditDrawerProps> = (props) => {
  const formRef = useRef<FormApi<any>>();
  const flowEnviroment = useFlowEnviromentContext();

  const clientContext = useClientContext();

  // let find = props.entity.designContent.nodes.find(item=>item.data.name_ == 'Prefill_1')

 // let node = clientContext.document.getNode(find!.id)
  //const extraInfo = node?.getExtInfo()

  

  // const { treeData } = useGetWorkflowBranch()

  async function onSure() {
    try {
      await formRef!.current!.validate();
      const formData = formRef!.current!.getValues();
      //console.log("Form Data:", formData);

        const businessContent = designDataToBusinessData(
              props.entity.designContent, 
              flowEnviroment.graphTopNode, 
              props.entity.designContent.nodes, 
              clientContext
            );

      const data: IWorkFlowEntity = {
        ...props.entity, 
        businessContent, 
        ...formData,

      };

      businessContent.name_ = formData['name']

      //return
      const id = props.flowType == EnumFlowType.template ? '':  props.entity.id ?? ""

      //return

      const response = await apiWorkFlowSave(id,   businessContent );
      if (response.flag == "success") {
        props.onSure({...data, id: response.result.id});

      }

      Toast.success("save sucess!");
    } catch (error) {
      Toast.error("save fail " + error);
    }
  }

  const initValues = {
    parentId: props.entity.parentId,
    name: props.entity.businessContent.name_,

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

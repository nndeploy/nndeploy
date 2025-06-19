import { Form, Button, Toast } from "@douyinfe/semi-ui";
const { Input } = Form;

import { apiGetWorkFlow, apiWorkFlowSave } from "../api";
import { IWorkFlowEntity, IWorkFlowTreeNodeEntity } from "../entity";
import { useEffect, useRef, useState } from "react";
import { FormApi } from "@douyinfe/semi-ui/lib/es/form";
import { designDataToBusinessData } from "../../../../components/flow/FlowSaveDrawer/functions";

export interface WorkFlowEditDrawerProps {
  onSure: (node: IWorkFlowTreeNodeEntity) => void;
  onClose: () => void;
  entity: IWorkFlowTreeNodeEntity;
}

const WorkFlowEditDrawer: React.FC<WorkFlowEditDrawerProps> = (props) => {
  const formRef = useRef<FormApi<any>>();

  const [entity, setEntity] = useState<IWorkFlowEntity>({
    ...props.entity,
    designContent: {
      nodes: [],
      edges: [],
    },
    businessContent: {
      key_: "nndeploy::dag::Graph",
      name_: "demo",
      device_type_: "kDeviceTypeCodeX86:0",
      inputs_: [],
      outputs_: [
        {
          name_: "detect_out",
          type_: "kNotSet",
        },
      ],
      is_external_stream_: false,
      is_inner_: false,
      is_time_profile_: true,
      is_debug_: false,
      is_graph_node_share_stream_: true,
      queue_max_size_: 16,
      node_repository_: [],
    },
  });

  useEffect(() => {
    if (props.entity.id) {
      apiGetWorkFlow(props.entity.id).then((res) => {
        if (res.flag == "success") {
          setEntity(res.result);
        }
      });
    }
  }, [props.entity]);

  async function onSure() {
    debugger;
    try {
      await formRef!.current!.validate();
      const formData = formRef!.current!.getValues();
      //console.log("Form Data:", formData);

      const businessContent = designDataToBusinessData(
        entity.designContent
      );

      const data: IWorkFlowEntity = {
        ...entity,
        businessContent,
        ...formData,
      };

      const response = await apiWorkFlowSave(data);
      if (response.flag == "success") {
        props.onSure({ ...response.result, type: "leaf" });
      }

      Toast.success("add sucess!");
    } catch (error) {
      Toast.error("add fail " + error);
     // console.log("add fail ", error)
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

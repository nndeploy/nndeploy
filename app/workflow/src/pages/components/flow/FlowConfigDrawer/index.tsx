// import { Form, Button, Toast } from "@douyinfe/semi-ui";
// import { NodeEntityForm } from "../NodeRepositoryEditor";
// import { useState } from "react";
// import { useFlowEnviromentContext } from "../../../../context/flow-enviroment-context";
// import { INodeEntity } from "../../../Node/entity";

// export interface FlowConfigDrawerProps {
//   onSure: (node: any) => void;
//   onClose: () => void;
//   entity: INodeEntity; 
//   visible: boolean
// }

// const FlowConfigDrawer: React.FC<FlowConfigDrawerProps> = (props) => {


// const flowContent = useFlowEnviromentContext()

// const { nodeList, paramTypes, } = flowContent
//   function closeEditor(){
//     props.onClose()
//   }
//   async function onSure() {
//     try {
    
//       Toast.success("save sucess!");
//     } catch (error) {
//       Toast.error("save fail " + error);
//     }
//   }

//   function handleSave(values:any){
//     props.onSure(values)
//   }
//   return (
//     <>
//        <NodeEntityForm
//           nodeEntity={props.entity}
//           visible={props.visible}
//           onClose={closeEditor}
//           onSave={handleSave}
//           nodeList={nodeList!}
//           paramTypes={paramTypes}
//         />
//     </>
//   );
// };

// export default FlowConfigDrawer;

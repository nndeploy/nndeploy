import {
  EditorRenderer,
  FreeLayoutEditorProvider,
  FreeLayoutPluginContext,
  usePlaygroundTools,
} from "@flowgram.ai/free-layout-editor";

import "@flowgram.ai/free-layout-editor/index.css";
import "./styles/index.css";
//import { nodeRegistries } from "../../../nodes";
import { initialData } from "./initial-data";
import { useEditorProps } from "../../../hooks";
import { AutoLayoutHandle, DemoTools } from "../../../components/tools";
import { SidebarProvider, SidebarRenderer } from "../../../components/sidebar";
import { useEffect, useRef, useState } from "react";
import { FlowEnviromentContext } from "../../../context/flow-enviroment-context";
import { apiGetNodeById, apiGetWorkFlow, getNodeRegistry } from "./api";

import { FlowDocumentJSON, FlowNodeRegistry } from "../../../typings";
import { SideSheet, Toast } from "@douyinfe/semi-ui";
import FlowSaveDrawer from "./FlowSaveDrawer";
import { IBusinessNode, IWorkFlowEntity } from "../../Layout/Design/WorkFlow/entity";
import { useGetNodeList, useGetParamTypes, useGetRegistry } from "./effect";
import { designDataToBusinessData, transferBusinessContentToDesignContent } from "./FlowSaveDrawer/functions";
import { apiWorkFlowRun, apiWorkFlowSave } from "../../Layout/Design/WorkFlow/api";
import { IconLoading } from "@douyinfe/semi-icons";

let nameId = 0; 

interface FlowProps {
  id: string;
  onFlowSave: (flow: IWorkFlowEntity) => void;
}
const Flow: React.FC<FlowProps> = (props) => {
  //const [flowData, setFlowData] = useState<FlowDocumentJSON>();

  const ref = useRef<FreeLayoutPluginContext | undefined>();

  //const tools = usePlaygroundTools();

  const [entity, setEntity] = useState<IWorkFlowEntity>({
    id: props.id,
    name: props.id,
    parentId: "",
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

  const [loading, setLoading] = useState(true);

  
  const autoLayOutRef = useRef<AutoLayoutHandle>(); 

  const [saveDrawerVisible, setSaveDrawerVisible] = useState(false);

  function handleSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }

  const nodeList = useGetNodeList()

  const paramTypes = useGetParamTypes()

  const [nodeRegistries, setNodeRegistries] = useState<FlowNodeRegistry[]>([]);

  const fetchData = async (flowName: string) => {
    //setLoading(true);

    const nodeRegistries = await getNodeRegistry();
    setNodeRegistries(nodeRegistries);

    if (!flowName) {
      setLoading(false);
      return;
    }
    const response = await apiGetWorkFlow(flowName);
    if (response.flag == "error") {
      return;
    }

    const designContent = transferBusinessContentToDesignContent(response.result, nodeRegistries)

    

    setEntity({...entity, designContent, businessContent: response.result});


    ref?.current?.document.reload(designContent);

    //ref?.current?.document.reload(response.result.content);

    //setFlowDocumentJSON()

    setTimeout(() => {
      // 加载后触发画布的 fitview 让节点自动居中
      ref?.current?.document.fitView();
      
      autoLayOutRef.current?.autoLayout()
      //tools.autoLayout()
      
    }, 100);

    //setFlowData(response);
    setLoading(false);
  };
  useEffect(() => {
    fetchData(props.id);
  }, [props.id]);

  const flowRef = useRef<HTMLDivElement | null>(null);

  const dropzone = useRef<HTMLDivElement | null>(null);

  function onSave(flowJson: FlowDocumentJSON) {
    setEntity({
      ...entity,
      designContent: flowJson,
    });

    setSaveDrawerVisible(true);
  }

  async function onRun(flowJson: FlowDocumentJSON) {
      try {
    
        const businessContent = designDataToBusinessData(
              flowJson
            );

      const response = await apiWorkFlowRun(businessContent);
     
      Toast.success("run sucess!");
    } catch (error) {
      Toast.error("run fail " + error);
    }

  }

  function onflowSaveDrawrSure(entity: IWorkFlowEntity) {
    setSaveDrawerVisible(false);
    setEntity({...entity});
    props.onFlowSave(entity);
  }
  function onFlowSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }

  useEffect(() => {
    let handleDrop: any = null;
    if (dropzone.current) {
      dropzone.current.addEventListener("dragover", (e) => {
        e.preventDefault();
        //dropzone.classList.add('over'); // 鼠标经过目标区域时强调样式
      });

      // dropzone.current.addEventListener('dragleave', () => {
      //     //dropzone.classList.remove('over'); // 离开时恢复样式
      // });

      if (handleDrop) {
        dropzone.current.removeEventListener("drop", handleDrop);
        handleDrop = null;
      }

      handleDrop = dropzone.current.addEventListener("drop", async (e) => {
        e.preventDefault();

        const position =
          ref?.current?.playground.config.getPosFromMouseEvent(e)!;

        const nodeString = e?.dataTransfer?.getData("text")!;

        const entity = JSON.parse(nodeString)

        //const response = await apiGetNodeById(nodeId!);

        //const entity = nodeList.find(item=>item.key_ == nodeId)!
        //nodeRegistries.find(item=>item.)
       
        //let type = ['nndeploy::detect::YoloGraph'].includes(  response.result.key_) ? 'group':  response.result.key_
         var type = entity.is_graph_ ? 'group':  entity.key_
        
        let node = {
          // ...response.result,
          id: Math.random().toString(36).substr(2, 9),
          type,
          meta: {
            position: {
              x: position?.x,
              y: position?.y,
            },
          },
          data: {
            //title: response.result.key_,
            ...entity,
            name_: `${entity.name_}_${nameId++}`,
          },
        }
        //if(response.result.is_dynamic_input_){

          node.data.inputs_ = node.data.inputs_.map((item : any)=>{
            return {
              ...item, 
              id: 'port' + Math.random().toString(36).substr(2, 9),
            }
          })
        //}

        //if(response.result.is_dynamic_output_){
          
           node.data.outputs_ = node.data.outputs_.map((item:any)=>{
            return {
              ...item, 
              id: 'port' + Math.random().toString(36).substr(2, 9),
            }
          })
        //}

        ref?.current?.document.createWorkflowNode(node);
      });
    }
    //清理函数
    return () => {
      if (handleDrop) {
        dropzone?.current?.removeEventListener("dragover", (e) =>
          e.preventDefault()
        );
        dropzone?.current?.removeEventListener("drop", handleDrop);
        handleDrop = null;
      }
    };
  }, [dropzone]);


  //const nodeRegistries = useGetRegistry()

  const editorProps = useEditorProps(entity.designContent, nodeRegistries);
  return (
    <div className="doc-free-feature-overview" ref={dropzone}>
      {loading ? (
        <IconLoading />
      ) : (
        <FreeLayoutEditorProvider
          {...editorProps}
          ///@ts-ignore
          ref={ref}
        >
          <SidebarProvider>
            <div className="demo-container" ref={flowRef}>
              <EditorRenderer className="demo-editor" />
            </div>
            <FlowEnviromentContext.Provider
              value={{ element: flowRef, onSave, onRun, nodeList, paramTypes }}
            >
              <DemoTools 
              ///@ts-ignore
              ref={autoLayOutRef}/>

              <SidebarRenderer />
            </FlowEnviromentContext.Provider>
          </SidebarProvider>
        </FreeLayoutEditorProvider>
      )}
      <SideSheet
        width={"30%"}
        visible={saveDrawerVisible}
        onCancel={handleSaveDrawerClose}
        title={"save flow"}
      >
        <FlowSaveDrawer
          entity={entity!}
          onSure={onflowSaveDrawrSure}
          onClose={onFlowSaveDrawerClose}
        />
      </SideSheet>
    </div>
  );
};

export default Flow;

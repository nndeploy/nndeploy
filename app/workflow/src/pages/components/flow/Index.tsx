import {
  EditorRenderer,
  FreeLayoutEditorProvider,
  FreeLayoutPluginContext,
} from "@flowgram.ai/free-layout-editor";

import "@flowgram.ai/free-layout-editor/index.css";
import "./styles/index.css";
import { nodeRegistries } from "../../../nodes";
import { initialData } from "./initial-data";
import { useEditorProps } from "../../../hooks";
import { DemoTools } from "../../../components/tools";
import { SidebarProvider, SidebarRenderer } from "../../../components/sidebar";
import { useEffect, useRef, useState } from "react";
import { FlowEnviromentContext } from "../../../context/flow-enviroment-context";
import { apiGetNodeType, apiGetWorkFlow } from "./api";

import { FlowDocumentJSON } from "../../../typings";
import { SideSheet } from "@douyinfe/semi-ui";
import FlowSaveDrawer from "./FlowSaveDrawer";
import { IWorkFlowEntity } from "../../Layout/Backend/WorkFlow/entity";
interface FlowProps {
  id: string;
  onFlowSave: (flow: IWorkFlowEntity) => void;
}
const Flow: React.FC<FlowProps> = (props) => {
  //const [flowData, setFlowData] = useState<FlowDocumentJSON>();

  const ref = useRef<FreeLayoutPluginContext | undefined>();

  const [entity, setEntity] = useState<IWorkFlowEntity>({
    id: props.id,
    name: "",
    parentId: "",
    content: {
      nodes: [],
      edges: [],
    },
  });

  const [loading, setLoading] = useState(true);

  const [saveDrawerVisible, setSaveDrawerVisible] = useState(false);

  function handleSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }

  const fetchData = async (id: string) => {
    //setLoading(true);

    const response = await apiGetWorkFlow(id);
    if (response.flag == "error") {
      return;
    }

    setEntity(response.result);

    ref?.current?.document.reload(initialData);

    //ref?.current?.document.reload(response.result.content);

    //setFlowDocumentJSON()

    setTimeout(() => {
      // 加载后触发画布的 fitview 让节点自动居中
      ref?.current?.document.fitView();
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
      content: flowJson,
    });

    setSaveDrawerVisible(true);
  }

  function onflowSaveDrawrSure(entity: IWorkFlowEntity) {
    setSaveDrawerVisible(false);
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

      if(handleDrop){
         dropzone.current.removeEventListener("drop", handleDrop);
         handleDrop = null 
      }

      handleDrop = dropzone.current.addEventListener("drop", async (e) => {
        e.preventDefault();

        const position =
          ref?.current?.playground.config.getPosFromMouseEvent(e)!;

        const nodeType = e?.dataTransfer?.getData("text");

        const response = await apiGetNodeType(nodeType!);

        ref?.current?.document.createWorkflowNode({
          ...response.result,
          id: Math.random().toString(36).substr(2, 9),
          type: nodeType!,
          meta: {
            position: {
              x: position?.x,
              y: position?.y,
            },
          },
        });
      });
    }
    //清理函数
    return () => {
      if ( handleDrop) {
        dropzone?.current?.removeEventListener("dragover", (e) =>
          e.preventDefault()
        );
        dropzone?.current?.removeEventListener("drop", handleDrop);
        handleDrop = null ; 
      }
    };
  }, [dropzone]);

  const editorProps = useEditorProps(entity.content, nodeRegistries);
  return (
    <div className="doc-free-feature-overview" ref={dropzone}>
      {loading ? (
        <></>
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
              value={{ element: flowRef, onSave }}
            >
              <DemoTools />

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

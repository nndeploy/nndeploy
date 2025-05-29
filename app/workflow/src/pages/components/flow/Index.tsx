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
import { FlowElementContext } from "../../../context/flow-element-context";
import { apiGetNodeType } from "./api";

import { FlowDocumentJSON } from '../../../typings';

const Flow = () => {
  //const [flowData, setFlowData] = useState<FlowDocumentJSON>();

  const ref = useRef<FreeLayoutPluginContext | undefined>();

  const [loading, setLoading] = useState(true);

  const [flowDocumentJSON, setFlowDocumentJSON] = useState<FlowDocumentJSON>({nodes: [], edges: []});


  const fetchData = async (params = {}) => {
    //setLoading(true);

    const response: any = await new Promise((resolve) => {
      setTimeout(() => {
        resolve({...initialData});
      }, 5000);
    });

    // ref?.current?.document.reload(response);
    // setTimeout(() => {
    //   // 加载后触发画布的 fitview 让节点自动居中
    //   ref?.current?.document.fitView();
    // }, 100);

    setFlowDocumentJSON(response)

    //setFlowData(response);
    setLoading(false);
  };
  useEffect(()=>{
    fetchData();
  }, [])

  const flowRef = useRef<HTMLDivElement | null>(null);

  const dropzone = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (dropzone.current) {

       dropzone.current.addEventListener('dragover', (e) => {
            e.preventDefault();
            //dropzone.classList.add('over'); // 鼠标经过目标区域时强调样式
        });

        // dropzone.current.addEventListener('dragleave', () => {
        //     //dropzone.classList.remove('over'); // 离开时恢复样式
        // });

      dropzone.current.addEventListener("drop", async(e) => {
        e.preventDefault();

        const position =
          ref?.current?.playground.config.getPosFromMouseEvent(e)!;

        const nodeType = e?.dataTransfer?.getData("text");

        const response = await apiGetNodeType(nodeType!)


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
  }, [dropzone.current]);

  const editorProps = useEditorProps(flowDocumentJSON, nodeRegistries);
  return (
    <div className="doc-free-feature-overview" ref={dropzone}>
      {
        loading ? <></>: 
         <FreeLayoutEditorProvider
        {...editorProps}
        ///@ts-ignore
        ref={ref}
      >
        <SidebarProvider>
          <div className="demo-container" ref={flowRef}>
            <EditorRenderer className="demo-editor" />
          </div>
          <DemoTools />
          <FlowElementContext.Provider value={{ element: flowRef }}>
            <SidebarRenderer />
          </FlowElementContext.Provider>
        </SidebarProvider>
      </FreeLayoutEditorProvider>
      
      }
      
     
    </div>
  );
};

export default Flow;

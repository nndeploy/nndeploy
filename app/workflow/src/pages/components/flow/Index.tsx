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
import { useRef, useState } from "react";
import { FlowDocumentJSON } from "../../../typings";

 const Flow = () => {
  //const [flowData, setFlowData] = useState<FlowDocumentJSON>();

  const ref = useRef<FreeLayoutPluginContext | undefined>();

  const [loading, setLoading] = useState(false);

  const fetchData = async (params = {}) => {
    setLoading(true);

    const response: any = await new Promise((resolve) => {
      setTimeout(() => {
        resolve(initialData);
      }, 1000);
    });

    ref?.current?.document.fromJSON(response);
    setTimeout(() => {
      // 加载后触发画布的 fitview 让节点自动居中
      ref?.current?.document.fitView();
    }, 100);

    //setFlowData(response);
    setLoading(false);
  };

  const editorProps = useEditorProps(initialData, nodeRegistries);
  return (
    <div className="doc-free-feature-overview">
      <FreeLayoutEditorProvider
        {...editorProps}
        ///@ts-ignore
        ref={ref}
      >
        <SidebarProvider>
          <div className="demo-container">
            <EditorRenderer className="demo-editor" />
          </div>
          <DemoTools />
          <SidebarRenderer />
        </SidebarProvider>
      </FreeLayoutEditorProvider>
    </div>
  );
};

export default Flow

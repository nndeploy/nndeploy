import {
  EditorRenderer,
  FreeLayoutEditorProvider,
  FreeLayoutPluginContext,
  WorkflowNodeJSON,
} from "@flowgram.ai/free-layout-editor";

import "@flowgram.ai/free-layout-editor/index.css";
import "./styles/index.css";
import "./styles/my.css";
//import { nodeRegistries } from "../../../nodes";
import { useEditorProps } from "../../../hooks";
import { AutoLayoutHandle, DemoTools } from "../../../components/tools";
import { SidebarProvider, SidebarRenderer } from "../../../components/sidebar";
import { useEffect, useRef, useState } from "react";
import { FlowEnviromentContext } from "../../../context/flow-enviroment-context";
import { apiGetTemeplateWorkFlow, apiGetWorkFlow, setupWebSocket } from "./api";

import { FlowDocumentJSON } from "../../../typings";
import { SideSheet, Toast } from "@douyinfe/semi-ui";
import FlowSaveDrawer from "./FlowSaveDrawer";
import { IBusinessNode, IWorkFlowEntity } from "../../Layout/Design/WorkFlow/entity";
import {
  //useGetNodeList, 
  useGetParamTypes,
  //useGetRegistry
} from "./effect";
import { designDataToBusinessData, transferBusinessContentToDesignContent } from "./FlowSaveDrawer/functions";
import { apiModelsRunDownload, apiWorkFlowRun } from "../../Layout/Design/WorkFlow/api";
import { IconLoading } from "@douyinfe/semi-icons";
import lodash from "lodash";
import { getNextNameNumberSuffix } from "./functions";
import store, { } from "../../Layout/Design/store/store";
import React from "react";
import { initFreshFlowTree } from "../../Layout/Design/store/actionType";
import { IFlowNodesRunningStatus, ILog, IOutputResource } from "./entity";
import { NodeEntityForm } from "./NodeRepositoryEditor";
import { IResponse } from "../../../request/types";
import { EnumFlowType } from "../../../enum";

let nameId = 0;

interface FlowProps {
  id: string;
  flowType: EnumFlowType;

  activeKey: string;
  onFlowSave: (flow: IWorkFlowEntity) => void;
}
const Flow: React.FC<FlowProps> = (props) => {
  //const [flowData, setFlowData] = useState<FlowDocumentJSON>();

  // const [state, dispatch] = useReducer(reducer, (initialState))
  const { state, dispatch } = React.useContext(store);

  const { nodeRegistries, nodeList } = state
  const [downloadModalVisible, setDownloadModalVisible] = useState(false)
  const [downloadModalList, setDownloadModalList] = useState<string[]>([])

  const [flowType, setFlowType] = useState<EnumFlowType>(props.flowType);


  const [outputResource, setOutputResource] = useState<IOutputResource>({ path: [], text: [] })

  const [flowNodesRunningStatus, setFlowNodesRunningStatus] = useState<IFlowNodesRunningStatus>({})

  const [graphTopNode, setGraphTopNode] = useState<IBusinessNode>({} as IBusinessNode)

  const [runResult, setRunResult] = useState<string>('');

  const [downloading, setDownloading] = useState(false)

  const [log, setLog] = useState<ILog>({
    items: [],
    time_profile: {
      init_time: undefined,
      run_time: undefined

    }
  })

  useEffect(() => {
    setGraphTopNode(lodash.cloneDeep(state.dagGraphInfo.graph))
  }, [state.dagGraphInfo.graph])



  const ref = useRef<FreeLayoutPluginContext | undefined>();

  const [entity, setEntity] = useState<IWorkFlowEntity>({
    id: props.id,
    // name: '',
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
  const [configDrawerVisible, setConfigDrawerVisible] = useState(false);

  function handleConfigDrawerSure(values: any) {
    setConfigDrawerVisible(false)
    setGraphTopNode(values)
  }

  function handleConfigDrawerClose() {
    setConfigDrawerVisible(false)
  }

  const [saveDrawerVisible, setSaveDrawerVisible] = useState(false);

  function handleSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }

  const paramTypes = useGetParamTypes()


  const fetchData = async (flowId: string, flowType: EnumFlowType) => {

    if (!flowId) {
      setLoading(false);
      return;
    }

    function getdownloadModals(businessNode: IBusinessNode) {

      const modals: string[] = []
      const fields = lodash.pick(businessNode, ['image_url_', 'video_url_', 'audio_url_', 'model_url_', 'other_url_'])
      for (let field in fields) {
        const urlArray: string[] = fields[field]
        urlArray.map(url => {
          if (url.startsWith('modelscope')) {
            modals.push(url)
          }
        })

      }
      return modals

    }

    let response: IResponse<IBusinessNode>

    if (flowType == EnumFlowType.template) {
      response = await apiGetTemeplateWorkFlow(flowId);
      if (response.flag == "error") {
        return;
      }
      const modals = getdownloadModals(response.result)
      setDownloadModalList(modals)
      setDownloadModalVisible(true)

    } else {
      response = await apiGetWorkFlow(flowId);
      if (response.flag == "error") {
        return;
      }
    }




    const designContent = transferBusinessContentToDesignContent(response.result, nodeRegistries)



    setEntity({ ...entity, designContent, businessContent: response.result });

    setGraphTopNode(lodash.omit(response.result, ['nndeploy_ui_layout', 'node_repository_']) as any)


    ref?.current?.document.reload(designContent);

    setTimeout(() => {
      // 加载后触发画布的 fitview 让节点自动居中
      ref?.current?.document.fitView();

      if (!response.result.nndeploy_ui_layout) {
        autoLayOutRef.current?.autoLayout()
      }



    }, 100);


    setLoading(false);
  };
  useEffect(() => {

    if (nodeRegistries.length < 1) {
      return
    }
    fetchData(props.id, props.flowType);
  }, [props.id, nodeRegistries, props.flowType]);

  // useEffect(() => {
  //   if (ref.current) {
  //     setTimeout(() => {
  //       ref?.current?.document.fitView();

  //       autoLayOutRef.current?.autoLayout()
  //     }, 100)
  //   }
  // }, [props.activeKey, ref.current])

  const demoContainerRef = useRef<HTMLDivElement | null>(null);

  const dropzone = useRef<HTMLDivElement | null>(null);

  function onSave(flowJson: FlowDocumentJSON) {
    setEntity({
      ...entity,
      designContent: flowJson,
    });

    setSaveDrawerVisible(true);
  }

  useEffect(() => {
    const socket = setupWebSocket()
    setSocket(socket)

    return () => {
      socket.close()
    }
  }, [])

  function onConfig() {

    setConfigDrawerVisible(true);
  }

  const [socket, setSocket] = useState<WebSocket>();

  useEffect(() => {



    const socket = setupWebSocket()
    setSocket(socket)

    socket!.onerror = (err) => {
      console.error("WebSocket error:", err);

      //connect()

    };

    socket!.onclose = () => {
      console.log("WebSocket 已断开连接");

      //connect()

    };

    return () => {
      socket!.close()
    }
  }, [])


  async function onDownload(flowJson: FlowDocumentJSON) {
    try {

      setDownloading(true)

      const businessContent = designDataToBusinessData(
        flowJson,
        graphTopNode,
        flowJson.nodes
      );

      const response = await apiModelsRunDownload(businessContent);

      if (response.flag == "error") {
        setDownloading(false)
        Toast.error("run fail " + response.message);
        return;
      }
      const taskId = response.result.task_id

      socket!.send(JSON.stringify({ type: "bind", task_id: taskId }));

      socket!.onclose = () => {

      };

      var downloadResolve: any;
      var downloadReject: any

      socket!.onmessage = (event) => {

        const response = JSON.parse(event.data);




        if (response.flag != "success") {

          downloadReject()
          Toast.error(response.message);
          setDownloading(false)
          return;
        } else {

          if (response.result.type == 'model_download_done') {
            downloadResolve()
            Toast.success(response.message)
            setDownloading(false)
          } else if (response.result.type == 'log')


            setLog((oldLog) => {
              var newLog = {
                ...oldLog,
                items: [...oldLog.items, response.result.log],

              }
              return newLog
            })


        }


      };


      return new Promise((resolve, reject) => {
        downloadResolve = resolve;
        downloadReject = reject

      })


      //Toast.success("run sucess!");
    } catch (error) {
      Toast.error("run fail " + error);
      setDownloading(false)
    } finally {
      //setDownloading(false)
    }

  }



  async function onRun(flowJson: FlowDocumentJSON) {
    try {

      setRunResult('')
      setLog({
        items: [],
        time_profile: {
          init_time: undefined,
          run_time: undefined
        }
      })
      const businessContent = designDataToBusinessData(
        flowJson,
        graphTopNode,
        flowJson.nodes
      );

      const response = await apiWorkFlowRun(businessContent);

      if (response.flag == "error") {
        Toast.error("run fail " + response.message);
        return;
      }
      const taskId = response.result.task_id

      socket!.send(JSON.stringify({ type: "bind", task_id: taskId }));

      socket!.onclose = () => {

      };

      socket!.onmessage = (event) => {

        function modifyNodeByName(nodeName: string, newContent: any, designContent: FlowDocumentJSON) {
          function nodeIterate(
            node: WorkflowNodeJSON,
            process: (node: WorkflowNodeJSON) => void
          ) {
            process(node);
            if (node.blocks && node.blocks.length > 0) {
              node.blocks.forEach((block) => {
                nodeIterate(block, process);
              });
            }
          }

          designContent.nodes.map((node) => {
            nodeIterate(node, (node) => {
              if (node.data.name_ == nodeName) {
                node.data = {
                  ...node.data,
                  ...newContent
                }
              }
            });
          });
        }



        const response = JSON.parse(event.data);

        if (response.flag != "success") {

          if (response.result.type == 'task_run_info') {
            setFlowNodesRunningStatus({})
            setRunResult('error')
            Toast.error("run fail ");
          }
          return;
        }

        if (response.result.type == 'preview') {
          response.result.path = response.result.path.map(item => {
            return {
              ...item,
              path: `${item.path}&time=${Date.now()}`
            }
          })

          setOutputResource(response.result)
        } else if (response.result.type == 'progress') {
          setFlowNodesRunningStatus(response.result.detail)
        } else if (response.result.type == 'log') {


          setLog((oldLog) => {
            var newLog = {
              ...oldLog,
              items: [...oldLog.items, response.result.log],

            }
            return newLog
          })
        } else if (response.result.type == 'task_run_info') {

          setLog((oldLog) => {
            var newLog = {
              ...oldLog,
              time_profile: response.result.time_profile,

            }
            return newLog
          })
        }





      };



      //Toast.success("run sucess!");
    } catch (error) {
      Toast.error("run fail " + error);
    }

  }

  function onflowSaveDrawrSure(entity: IWorkFlowEntity) {

    setFlowType(EnumFlowType.workspace) //after save, template converted to user's flow

    setSaveDrawerVisible(false);
    setEntity({ ...entity });

    dispatch(initFreshFlowTree({}))


    props.onFlowSave(entity);
  }
  function onFlowSaveDrawerClose() {
    setSaveDrawerVisible(false);
  }

  useEffect(() => {
    let handleDrop: any = null;

    function dragover(e: any) {
      e.preventDefault();
    }

    async function dropFunction(e: any) {
      e.preventDefault();

      const position =
        ref?.current?.playground.config.getPosFromMouseEvent(e)!;

      const nodeString = e?.dataTransfer?.getData("text")!;

      const entity = JSON.parse(nodeString)

      var type = entity.key_

      let numberSuffix = getNextNameNumberSuffix(ref?.current?.document.toJSON() as FlowDocumentJSON)

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
          name_: `${entity.name_}_${numberSuffix}`,
        },
      }
      //if(response.result.is_dynamic_input_){

      node.data.inputs_ = node.data.inputs_.map((item: any) => {
        return {
          ...item,
          id: 'port' + Math.random().toString(36).substr(2, 9),
        }
      })
      //}

      //if(response.result.is_dynamic_output_){

      node.data.outputs_ = node.data.outputs_.map((item: any) => {
        return {
          ...item,
          id: 'port' + Math.random().toString(36).substr(2, 9),
        }
      })



      //}

      ref?.current?.document.createWorkflowNode(node);
    }
    if (dropzone.current) {


      dropzone.current.addEventListener("dragover", dragover);

      // dropzone.current.addEventListener('dragleave', () => {
      //     //dropzone.classList.remove('over'); // 离开时恢复样式
      // });


      dropzone.current.removeEventListener("drop", dropFunction);



      dropzone.current.addEventListener("drop", dropFunction);
    }
    //清理函数
    return () => {
      //if (handleDrop) {
      dropzone?.current?.removeEventListener("dragover", dragover);
      dropzone?.current?.removeEventListener("drop", dropFunction);
      // handleDrop = null;

    };
  }, [dropzone]);


  //const nodeRegistries = useGetRegistry()

  const editorProps = useEditorProps(entity.designContent, nodeRegistries);
  return (
    <div className="doc-free-feature-overview" ref={dropzone}>
      {loading ? (
        <IconLoading />
      ) : (
        <FlowEnviromentContext.Provider
          value={{
            element: demoContainerRef, onSave, onRun, onDownload, downloading, onConfig, graphTopNode, nodeList, paramTypes, outputResource, flowNodesRunningStatus,

            log: log,
            runResult: runResult, 
            downloadModalVisible, 
            setDownloadModalVisible, 
            downloadModalList
          }}
        >
          <FreeLayoutEditorProvider
            {...editorProps}
            ///@ts-ignore
            ref={ref}
          >
            <SidebarProvider>
              <div className="demo-container" ref={demoContainerRef}>
                <EditorRenderer className="demo-editor" />
              </div>

              <DemoTools
                ///@ts-ignore
                ref={autoLayOutRef} />

              <SidebarRenderer />

            </SidebarProvider>
            <NodeEntityForm
              nodeEntity={graphTopNode}
              visible={configDrawerVisible}
              onClose={handleConfigDrawerClose}
              onSave={handleConfigDrawerSure}
              nodeList={nodeList!}
              paramTypes={paramTypes}
            />
          </FreeLayoutEditorProvider>

          <SideSheet
            width={"80%"}
            visible={saveDrawerVisible}
            onCancel={handleSaveDrawerClose}
            title={"save flow"}
          >
            <FlowSaveDrawer
              entity={entity!}
              onSure={onflowSaveDrawrSure}
              onClose={onFlowSaveDrawerClose}
              flowType={flowType}

            />
          </SideSheet>

          {/* <SideSheet
            width={"30%"}
            visible={configDrawerVisible}
            onCancel={handleConfigDrawerClose}
            title={"config flow"}
          > */}

          {/* </SideSheet> */}
        

        </FlowEnviromentContext.Provider>
      )}

    </div>
  );
};

export default Flow;

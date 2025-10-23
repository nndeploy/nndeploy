import {
  EditorRenderer,
  FreeLayoutEditorProvider,
  FreeLayoutPluginContext,
  IPoint,
  usePlaygroundTools,
  WorkflowLinePortInfo,
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
import { IBusinessNode, Inndeploy_ui_layout, IWorkFlowEntity } from "../../Layout/Design/WorkFlow/entity";
import {
  //useGetNodeList, 
  useGetParamTypes,
  //useGetRegistry
} from "./effect";
import { businessNodeNormalize } from "./FlowSaveDrawer/functions";
import { apiModelsRunDownload, apiWorkFlowRun } from "../../Layout/Design/WorkFlow/api";
import { IconLoading } from "@douyinfe/semi-icons";
import lodash from "lodash";
import { getNextNameNumberSuffix, nodeIterate } from "./functions";
import store, { } from "../../Layout/Design/store/store";
import React from "react";
import { initFreshFlowTree, initFreshResourceTree } from "../../Layout/Design/store/actionType";
import { IDownloadProgress, IRunInfo } from "./entity";
import { NodeEntityForm } from "./NodeRepositoryEditor";
import { IResponse } from "../../../request/types";
import { EnumFlowType } from "../../../enum";
import { designDataToBusinessData } from "./FlowSaveDrawer/toBusiness";
import { buildEdges, transferBusinessContentToDesignContent, transferBusinessNodeToDesignNodeIterate } from "./FlowSaveDrawer/toDesign";

interface FlowProps {
  id: string;
  flowType: EnumFlowType;

  activeKey: string;
  onFlowSave: (flow: IWorkFlowEntity) => void;
}
const Flow: React.FC<FlowProps> = (props) => {

  const { state, dispatch } = React.useContext(store);

  const { nodeRegistries, nodeList } = state
  const [downloadModalVisible, setDownloadModalVisible] = useState(false)
  const [downloadModalList, setDownloadModalList] = useState<string[]>([])


  const [flowType, setFlowType] = useState<EnumFlowType>(props.flowType);

  const [runInfo, setRunInfo] = useState<IRunInfo>({
    isRunning: false,
    time: Date.now(),
    result: '',
    runningTaskId: '',
    downloadProgress: {},
    log: {
      items: [],
      time_profile: {
        init_time: undefined,
        run_time: undefined

      }
    },

    outputResource: {
      type: 'memory',
      content: {},
      time: Date.now()
    },
    flowNodesRunningStatus: {}
  })

  const [graphTopNode, setGraphTopNode] = useState<IBusinessNode>({} as IBusinessNode)

  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    setGraphTopNode(lodash.cloneDeep(state.dagGraphInfo.graph))
  }, [state.dagGraphInfo.graph])


  const ref = useRef<FreeLayoutPluginContext | undefined>();

  const [entity, setEntity] = useState<IWorkFlowEntity>({
    id: props.id,
    // name: '',
    name: '',
    parentId: "",
    designContent: {
      nodes: [],
      edges: [],
    },
    businessContent: {
      key_: "nndeploy::dag::Graph",
      name_: "demo",
      desc_: "",
      device_type_: "kDeviceTypeCodeX86:0",
      inputs_: [],
      outputs_: [
        {
          name_: "detect_out",
          type_: "kNotSet",
          desc_: ""
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

        ///@ts-ignore
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

      const modals = getdownloadModals(response.result)
      setDownloadModalList(modals)
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

    }, 10);


    setLoading(false);
  };
  useEffect(() => {

    if (nodeRegistries.length < 1) {
      return
    }
    fetchData(props.id, props.flowType);
  }, [props.id, nodeRegistries, props.flowType]);

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
      //console.log("WebSocket 已断开连接");

      //connect()

    };

    return () => {
      socket!.close()
    }
  }, [])


  async function onDownload(flowJson: FlowDocumentJSON) {
    try {

      setDownloading(true)

      setRunInfo((oldRunInfo) => {
        return {
          ...oldRunInfo,
          downloadProgress: {}
        }
      })

      const businessContent = designDataToBusinessData(
        flowJson,
        graphTopNode,
        flowJson.nodes,
        ref?.current!

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
            setRunInfo((oldRunInfo) => {


              const downloadProgress: IRunInfo['downloadProgress'] = {}

              for (const item of downloadModalList) {

                let nameParts = item.split(':')
                let name = nameParts[nameParts.length - 1]


                downloadProgress[name] = {
                  filename: name,
                  percent: 100,
                  downloaded: 0,
                  elapsed: 0,
                  total: 0
                }
              }


              return {
                ...oldRunInfo,
                downloadProgress
              }
            })
          } else if (response.result.type == 'download_progress') {

            const detail: IDownloadProgress = response.result.detail
            setRunInfo((oldRunInfo) => {
              return {
                ...oldRunInfo,
                downloadProgress: {
                  ...oldRunInfo.downloadProgress,
                  [detail.filename]: detail
                }
              }
            })
          }

          else if (response.result.type == 'log') {
            setRunInfo((oldRunInfo) => {
              return {
                ...oldRunInfo,
                log: {
                  ...oldRunInfo.log,
                  items: [...oldRunInfo.log.items, response.result.log],
                }
              }
            })
          }

        }
      }


      return new Promise((resolve, reject) => {
        downloadResolve = resolve;
        downloadReject = reject

      })

    } catch (error) {
      Toast.error("run fail " + error);
      setDownloading(false)
    } finally {
      //setDownloading(false)
    }

  }

  async function onRun(flowJson: FlowDocumentJSON) {
    try {

      setRunInfo(oldRunInfo => {
        return {
          ...oldRunInfo,
          isRunning: true,
          result: '',
          log: {
            items: [],
            time_profile: {
              init_time: undefined,
              run_time: undefined
            }
          }
        }

      })
      const businessContent = designDataToBusinessData(
        flowJson,
        graphTopNode,
        flowJson.nodes,
        ref?.current!
      );



      const response = await apiWorkFlowRun(businessContent);

      if (response.flag == "error") {
        Toast.error("run fail " + response.message);
        return;
      }
      const taskId = response.result.task_id

      //setRunningTaskId(taskId)
      setRunInfo(oldRunInfo => {
        return {


          ...oldRunInfo,
          runningTaskId: taskId,
        }
      })

      socket!.send(JSON.stringify({ type: "bind", task_id: taskId }));

      socket!.onclose = () => {
        //let j = 0
      };

      socket!.onmessage = (event) => {

        const response = JSON.parse(event.data);

        if (response.flag != "success") {

          if (response.result.type == 'task_run_info') {

            setRunInfo(oldRunInfo => {


              if (response.result.task_id == oldRunInfo.runningTaskId) {
                return {


                  ...oldRunInfo,
                  isRunning: false,
                  result: 'error',
                  flowNodesRunningStatus: {},
                }

              } else {
                return {
                  ...oldRunInfo
                }
              }

            })

            Toast.error("run fail ");
          }
          return;
        }

        if (response.result.type == 'memory') {

          setRunInfo(oldRunInfo => {
            return {

              ...oldRunInfo,
              outputResource: { ...response.result, time: Date.now() }
            }
          })

        } else if (response.result.type == 'progress') {

          setRunInfo(oldRunInfo => {

            return {


              ...oldRunInfo,
              flowNodesRunningStatus: response.result.detail
            }
          })
        } else if (response.result.type == 'log') {
          setRunInfo(oldRunInfo => {
            return {

              ...oldRunInfo,
              log: {
                ...oldRunInfo.log,
                items: [...oldRunInfo.log.items, response.result.log],
              }
            }
          })
        } else if (response.result.type == 'task_run_info') {

          setRunInfo(oldRunInfo => {

            return {
              ...oldRunInfo,
              isRunning: false,
              result: 'success',
              log: {
                ...oldRunInfo.log,
                time_profile: response.result.time_profile,
              },
              time: Date.now(),
            }


          })
          dispatch(initFreshResourceTree({}))
        }
      }

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

    function dragover(e: any) {
      e.preventDefault();
    }

    async function dropFunction(e: any) {
      e.preventDefault();

      const position =
        ref?.current?.playground.config.getPosFromMouseEvent(e)!;

      const nodeString = e?.dataTransfer?.getData("text")!;

      let entity: IBusinessNode = JSON.parse(nodeString)

      businessNodeNormalize(entity)


      const edges = buildEdges(entity, {})

      const lines: WorkflowLinePortInfo[] = edges.map(item => {
        return {
          from: item.sourceNodeID,
          fromPort: item.sourcePortID,
          to: item.targetNodeID,
          toPort: item.targetPortID
        }
      })

      let numberSuffix = getNextNameNumberSuffix(ref?.current?.document.toJSON() as FlowDocumentJSON)

      entity.name_ = `${entity.name_}_${numberSuffix}`

      function buildLayout(position: IPoint, entity: IBusinessNode) {

        const layout: Inndeploy_ui_layout['layout'] = entity.nndeploy_ui_layout?.layout ?? {

          "tokenizer_encode": {

            position: { x: 20, y: 10 },
            size: { width: 200, height: 80 }

          },
          "prefill_infer": {
            position: { x: 250, y: 10 },
            size: { width: 200, height: 80 }
          },
          "prefill_sampler": {
            position: { x: 20, y: 120 },
            size: { width: 200, height: 80 }
          },


        }

        // for (let nodeName in layout) {

        //   const nodeLayout = layout[nodeName]

        //   nodeIterate(nodeLayout, 'children', (node, parents) => {
        //     node.position = {
        //       x: node?.position?.x! , //+ position.x
        //       y: node.position?.y!  //+ position.y
        //     }
        //   })
        // }

        const result: Inndeploy_ui_layout['layout'] = {
          [entity.name_]: {
            position: position,
            size: { width: 200, height: 80 },
            children: layout
          },

        }

        return result
      }

      const layout: Inndeploy_ui_layout['layout'] = buildLayout(position, entity)

      let node = transferBusinessNodeToDesignNodeIterate(entity, layout)
      node.meta!.needInitAutoLayout = true

      ref?.current?.document.createWorkflowNode(node);



      const linesManager = ref?.current?.document.linesManager

      lines.map(line => {
        linesManager?.createLine(line)
      })

    }
    if (dropzone.current) {


      dropzone.current.addEventListener("dragover", dragover);


      dropzone.current.removeEventListener("drop", dropFunction);

      dropzone.current.addEventListener("drop", dropFunction);
    }

    return () => {

      dropzone?.current?.removeEventListener("dragover", dragover);
      dropzone?.current?.removeEventListener("drop", dropFunction);


    };
  }, [dropzone]);

  function getPopupContainer() {
    return demoContainerRef?.current!!
  }

  const editorProps = useEditorProps(entity.designContent, nodeRegistries);
  return (
    <div className="doc-free-feature-overview" ref={dropzone}>
      {loading ? (
        <IconLoading />
      ) : (
        <FlowEnviromentContext.Provider
          value={{
            element: demoContainerRef,
            onSave,
            onRun,
            onDownload,
            downloading,
            onConfig, graphTopNode, nodeList, paramTypes,


            downloadModalVisible,
            setDownloadModalVisible,
            downloadModalList,
            runInfo,
            setRunInfo

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
            <SideSheet
              width={"80%"}
              visible={saveDrawerVisible}
              onCancel={handleSaveDrawerClose}
              title={"save flow"}
              getPopupContainer={getPopupContainer}
            >
              <FlowSaveDrawer
                entity={entity!}
                onSure={onflowSaveDrawrSure}
                onClose={onFlowSaveDrawerClose}
                flowType={flowType}

              />
            </SideSheet>
          </FreeLayoutEditorProvider>



        </FlowEnviromentContext.Provider>
      )}

    </div>
  );
};

export default Flow;

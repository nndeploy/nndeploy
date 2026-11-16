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
import { getNextNameNumberSuffix, isTemplateDownloadedBefore, nodeIterate, setTemplateDownloaded } from "./functions";
import store, { } from "../../Layout/Design/store/store";
import React from "react";
import { initFreshFlowTree, initFreshResourceTree } from "../../Layout/Design/store/actionType";
import { IDownloadProgress, IRunInfo } from "./entity";
import { NodeEntityForm } from "./NodeRepositoryEditor";
import { IResponse } from "../../../request/types";
import { EnumFlowType } from "../../../enum";
import { designDataToBusinessData } from "./FlowSaveDrawer/toBusiness";
import { serverRequest } from '../../../components/tools/server';
import { recordRunState, updateRunTab } from '../../../components/tools/run-session';
import { watchRunTask, restoreRun } from '../../../components/tools/run-connection';
import { applyRunTaskState } from '../../../components/tools/run-state';
import { useEditorAccessAllowed } from '../../../components/editor-access';
import { buildEdges, transferBusinessContentToDesignContent, transferBusinessNodeToDesignNodeIterate } from "./FlowSaveDrawer/toDesign";

interface FlowProps {
  tabId: string;
  name: string;
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

  const [restoredRun] = useState(() => restoreRun(props.tabId));
  const [runInfo, setRunInfo] = useState<IRunInfo>({
    isRunning: Boolean(restoredRun),
    submissionPending: Boolean(restoredRun && !restoredRun.confirmed),
    time: Date.now(),
    result: '',
    runningTaskId: restoredRun?.taskId || '',
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
    if (!runInfo.isRunning && runInfo.outputResource.truncated) {
      Toast.warning('Output preview was truncated.');
    }
  }, [runInfo.isRunning, runInfo.runningTaskId, runInfo.outputResource.truncated]);

  useEffect(() => {
    if (!runInfo.isRunning && runInfo.outputResource.expired) {
      Toast.info('This task finished earlier. Its saved results have expired.');
    }
  }, [runInfo.isRunning, runInfo.runningTaskId, runInfo.outputResource.expired]);

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
  const restoredViewFitted = useRef(false);
  const restoredGraphLoaded = useRef(false);

  useEffect(() => {
    if (!restoredRun || loading || props.activeKey !== props.tabId || restoredViewFitted.current) return;
    const frame = requestAnimationFrame(() => {
      const context = ref.current;
      // Tabs can mount while hidden. Refresh the editor's cached viewport
      // dimensions after it becomes visible, before calculating the zoom.
      if (context?.playground.resize()) {
        void context.tools.fitView(false);
        restoredViewFitted.current = true;
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [loading, props.activeKey, props.tabId, restoredRun]);


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

    if (!flowId && (!restoredRun || restoredGraphLoaded.current)) {
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
          if (!url.startsWith('template')) {
            modals.push(url)
          }
        })

      }
      return modals

    }


    let response: IResponse<IBusinessNode>

    if (!flowId && restoredRun) {
      response = { flag: 'success', message: '', status: 200, error: 0, result: JSON.parse(restoredRun.payload) };
    } else if (flowType == EnumFlowType.template) {
      response = await apiGetTemeplateWorkFlow(flowId);
      if (response.flag == "error") {
        return;
      }

      const modals = getdownloadModals(response.result)
      setDownloadModalList(modals)

      setDownloadModalVisible(modals.length > 0 && !isTemplateDownloadedBefore(flowId))

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


    // ref?.current?.document.reload(designContent);
    ref?.current?.operation.fromJSON(designContent)
    if (!flowId) restoredGraphLoaded.current = true;

    setTimeout(() => {
      const context = ref.current;
      if (!context) return;
      // Layout is asynchronous. Fit the resulting positions, not the initial
      // overlapping nodes. Workflows with saved positions keep their layout.
      const layout = response.result.nndeploy_ui_layout ? Promise.resolve()
        : context.tools.autoLayout({ enableAnimation: false, disableFitView: true });
      void layout.then(() => {
        if (ref.current === context && props.activeKey === props.tabId) {
          context.playground.resize();
          return context.tools.fitView(false);
        }
      }).catch(() => Toast.warning('Unable to arrange the workflow. Try Auto Layout.'));
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

  async function onStartServer(flowJson: FlowDocumentJSON, token: string, name: string, exposed: Record<string, string[][]>) {
    const businessContent = designDataToBusinessData(flowJson, graphTopNode, flowJson.nodes, ref.current!);
    const info = await serverRequest('/api/workflow/server/start', token, {
      id: flowType === EnumFlowType.workspace ? entity.id : '', businessContent, name, exposed_parameters: exposed,
    });
    if (info.workflow_id) {
      onflowSaveDrawrSure({ ...entity, id: info.workflow_id, designContent: flowJson, businessContent });
    }
    return info;
  }

  async function onServerParameters(flowJson: FlowDocumentJSON, token: string) {
    const businessContent = designDataToBusinessData(flowJson, graphTopNode, flowJson.nodes, ref.current!);
    const response = await fetch('/api/workflow/server/parameters', {
      method: 'POST', headers: { 'Content-Type': 'application/json', ...(token ? {Authorization: `Bearer ${token}`} : {}) },
      body: JSON.stringify({id: flowType === EnumFlowType.workspace ? entity.id : '', businessContent}),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error?.message || 'Unable to load node parameters');
    return value;
  }

  function onConfig() {

    setConfigDrawerVisible(true);
  }

  const [socket, setSocket] = useState<WebSocket>();
  const editorAccessAllowed = useEditorAccessAllowed();

  useEffect(() => {
    if (!editorAccessAllowed) {
      setSocket(undefined);
      return;
    }
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
  }, [editorAccessAllowed])


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


            if (props.flowType == EnumFlowType.template) {
              setTemplateDownloaded(props.id)
            }



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

  useEffect(() => {
    updateRunTab({ tabId: props.tabId, id: props.id, name: props.name, flowType: props.flowType });
  }, [props.tabId, props.id, props.name, props.flowType]);

  useEffect(() => {
    if (!runInfo.isRunning && runInfo.runningTaskId) {
      const state = runInfo.result === 'success' ? 'SUCCEEDED' : runInfo.result === 'cancelled' ? 'CANCELLED' : 'FAILED';
      recordRunState(runInfo.runningTaskId, state);
    }
  }, [runInfo.isRunning, runInfo.runningTaskId, runInfo.result]);

  useEffect(() => {
    const taskId = runInfo.runningTaskId;
    if (!editorAccessAllowed || !runInfo.isRunning || !taskId) return;
    return watchRunTask(taskId, setupWebSocket, task => {
      setRunInfo(info => applyRunTaskState(info, task));
      if (task.state === 'SUCCEEDED') dispatch(initFreshResourceTree({}));
    }, response => {
        if (response.result.task_id && response.result.task_id !== taskId) return;

        if (response.flag != "success") {

          if (response.result.type == 'task_run_info') {

            const cancelled = response.flag === 'cancelled';
            setRunInfo(info => applyRunTaskState(info, {
              task_id: response.result.task_id, state: cancelled ? 'CANCELLED' : 'FAILED',
              outputs_expired: response.result.outputs_expired, output_files: response.result.output_files, outputs: response.result.outputs, outputs_truncated: response.result.outputs_truncated,
              time_profile: response.result.time_profile,
            }));
            if (cancelled) Toast.info('Run cancelled');
            else Toast.error('Run failed');
          }
          return;
        }

        if (response.result.type == 'memory') {

          setRunInfo(oldRunInfo => {
            if (!oldRunInfo.isRunning || oldRunInfo.runningTaskId !== taskId) return oldRunInfo;
            return {

              ...oldRunInfo,
              outputResource: { ...response.result, time: Date.now() }
            }
          })

        } else if (response.result.type == 'progress') {

          setRunInfo(oldRunInfo => {
            if (!oldRunInfo.isRunning || oldRunInfo.runningTaskId !== taskId) return oldRunInfo;

            return {


              ...oldRunInfo,
              flowNodesRunningStatus: response.result.detail
            }
          })
        } else if (response.result.type == 'log') {
          setRunInfo(oldRunInfo => {
            if (!oldRunInfo.isRunning || oldRunInfo.runningTaskId !== taskId) return oldRunInfo;
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
            const completed = applyRunTaskState(oldRunInfo, { task_id: response.result.task_id, state: 'SUCCEEDED',
              time_profile: response.result.time_profile,
              outputs_expired: response.result.outputs_expired, output_files: response.result.output_files, outputs: response.result.outputs, outputs_truncated: response.result.outputs_truncated });
            if (completed === oldRunInfo) return oldRunInfo;

            return {
              ...completed,
              log: {
                ...oldRunInfo.log,
                time_profile: response.result.time_profile,
              },
              time: Date.now(),
            }


          })
          dispatch(initFreshResourceTree({}))
        } else if (response.result.type == 'system') {
          if (response.result.event == 'worker_died') {

            Toast.error("worker process died,restarting...");
          } else if (response.result.event == 'worker_restarted') {
            Toast.success("worker process restarted success");
          }
        }
    });
  }, [editorAccessAllowed, runInfo.runningTaskId, runInfo.isRunning]);

  const submittingRun = useRef(false);
  async function onRun(flowJson: FlowDocumentJSON) {
    if (submittingRun.current || runInfo.isRunning) return;
    submittingRun.current = true;
    try {

      setRunInfo(oldRunInfo => {
        return {
          ...oldRunInfo,
          isRunning: true,
          cancelRequested: false,
          submissionPending: false,
          runningTaskId: '',
          flowNodesRunningStatus: {},
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



      const response = await apiWorkFlowRun(businessContent, {
        tabId: props.tabId, id: props.id, name: props.name, flowType: props.flowType,
      });

      if (response.flag == "error") {
        setRunInfo(info => ({ ...info, isRunning: false, result: 'error', cancelRequested: false }));
        Toast.error("run fail " + response.message);
        return;
      }
      const taskId = response.result.task_id

      //setRunningTaskId(taskId)
      setRunInfo(oldRunInfo => {
        return {


          ...oldRunInfo,
          runningTaskId: taskId,
          submissionPending: response.flag === 'pending',
        }
      })

      if (response.flag === 'pending') Toast.info(response.message);
      if (response.storageWarning) Toast.warning(response.storageWarning);

      //Toast.success("run sucess!");
    } catch (error) {
      setRunInfo(info => info.runningTaskId ? info : { ...info, isRunning: false, result: 'error' });
      Toast.error('Unable to prepare workflow submission: ' + String(error));
    } finally {
      submittingRun.current = false;
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

        const isContainer = entity.is_graph_ || entity.is_loop_ || entity.is_composite_node_

        const result: Inndeploy_ui_layout['layout'] = {
          [entity.name_]: {
            position: position,
            size: { width: isContainer ? 400 : 200, height: isContainer ? 160 : 80 },
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
            onStartServer,
            onServerParameters,
            serverName: graphTopNode.name_ || props.name,
            workflowId: flowType === EnumFlowType.workspace ? entity.id : undefined,
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

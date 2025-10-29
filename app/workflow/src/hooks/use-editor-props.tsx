/* eslint-disable no-console */
import { useMemo } from "react";

import { debounce } from "lodash-es";
import { createMinimapPlugin } from "@flowgram.ai/minimap-plugin";
import { createFreeSnapPlugin } from "@flowgram.ai/free-snap-plugin";
import { createFreeNodePanelPlugin } from "@flowgram.ai/free-node-panel-plugin";
import { createFreeLinesPlugin } from "@flowgram.ai/free-lines-plugin";
import { FlowNodeEntity, FreeLayoutProps } from "@flowgram.ai/free-layout-editor";
import { createFreeGroupPlugin } from "@flowgram.ai/free-group-plugin";
import { createContainerNodePlugin } from "@flowgram.ai/free-container-plugin";

import { FlowNodeRegistry, FlowDocumentJSON } from "../typings";
import { shortcuts } from "../shortcuts";
import { CustomService, RunningService } from "../services";
import { createSyncVariablePlugin } from "../plugins";
import { defaultFormMeta } from "../nodes/default-form-meta";
import { WorkflowNodeType } from "../nodes";
import { SelectorBoxPopover } from "../components/selector-box-popover";
import {
  BaseNode,
  CommentRender,
  GroupNodeRender,
  NodePanel,
} from "../components";
import store from "../pages/Layout/Design/store/store";
import React from "react";
import { getNodeById, isbothNodeOffSpringOftheSameFixedGraph, isCompositeNode, isContainerNode, isFxiedGraphNode, isGraphNode, isNodeOffspringOfCompositeNode, isNodeOffSpringOfFixedGraph } from "../pages/components/flow/functions";
import { IExpandInfo } from "../pages/components/flow/entity";
import lodash from 'lodash'

export function useEditorProps(
  initialData: FlowDocumentJSON,
  nodeRegistries: FlowNodeRegistry[]
): FreeLayoutProps {

  //const { state } = React.useContext(store);

  return useMemo<FreeLayoutProps>(() => {
    return {
      /**
       * Whether to enable the background
       */
      background: true,
      /**
       * Whether it is read-only or not, the node cannot be dragged in read-only mode
       */
      readonly: false,
      /**
       * Initial data
       * 初始化数据
       */
      initialData,
      /**
       * Node registries
       * 节点注册
       */
      nodeRegistries,
      /**
       * Get the default node registry, which will be merged with the 'nodeRegistries'
       * 提供默认的节点注册，这个会和 nodeRegistries 做合并
       */
      getNodeDefaultRegistry(type) {
        return {
          type,
          meta: {
            defaultExpanded: true,
          },
          formMeta: defaultFormMeta,
        };
      },
      lineColor: {
        hidden: "transparent",
        default: "#4d53e8",
        drawing: "#5DD6E3",
        hovered: "#37d0ff",
        selected: "#37d0ff",
        error: "red",
        flowing: "#5DD6E3",
      },
      /*
       * Check whether the line can be added
       * 判断是否连线
       */
      canAddLine(ctx, fromPort, toPort) {
        // not the same node
        if (fromPort.node === toPort.node) {
          return false;
        }

        if (toPort.availableLines.length >= 1) {
          return false
        }



        if (isNodeOffspringOfCompositeNode(fromPort.node, ctx) || isNodeOffspringOfCompositeNode(toPort.node, ctx)) {
          return false
        }

        if (isbothNodeOffSpringOftheSameFixedGraph(fromPort.node, toPort.node, ctx)) {
          return false
        }

        //   var succesorNodes = dagGraphInfo.accepted_edge_types[fromPort.node.flowNodeType]
        //   if(succesorNodes && succesorNodes.includes(toPort.node.flowNodeType as string) || 
        //   !succesorNodes && fromPort.type

        // ){
        //     return false
        //   }
        // dagGraphInfo.accepted_edge_types[fromPort.node.]



        return true;
      },
      /**
       * Check whether the line can be deleted, this triggers on the default shortcut `Bakspace` or `Delete`
       * 判断是否能删除连线, 这个会在默认快捷键 (Backspace or Delete) 触发
       */
      canDeleteLine(ctx, line, newLineInfo, silent) {



        if (isbothNodeOffSpringOftheSameFixedGraph(line.from, line.to, ctx)) {
          return false
        }
        if (isGraphNode(line.from!.id, ctx)) {
          const fromNode = getNodeById(line.from!.id, ctx)
          const expandInfo: IExpandInfo = fromNode?.getNodeMeta().expandInfo
          let { outputLines = [] } = expandInfo

          outputLines = outputLines.filter(outputLine => {
            if (outputLine.from == line.from!.id && outputLine.fromPort == line.fromPort?.portID
              && outputLine.to == line.to!.id && outputLine.toPort == line.toPort?.portID

            ) {
              return false
            }
            return true
          })

          fromNode!.getNodeMeta().expandInfo = { ...expandInfo, outputLines }



        }

        if (isGraphNode(line.to!.id, ctx)) {
          const toNode = getNodeById(line.to!.id, ctx)
          const expandInfo: IExpandInfo = toNode?.getNodeMeta().expandInfo
          let { inputLines = [] } = expandInfo

          inputLines = inputLines.filter(inputLine => {
            if (inputLine.from == line.from!.id && inputLine.fromPort == line.fromPort?.portID
              && inputLine.to == line.to!.id && inputLine.toPort == line.toPort?.portID

            ) {
              return false
            }
            return true
          })

          toNode!.getNodeMeta().expandInfo = { ...expandInfo, inputLines }

        }
        return true;
      },
      /**
       * Check whether the node can be deleted, this triggers on the default shortcut `Bakspace` or `Delete`
       * 判断是否能删除节点, 这个会在默认快捷键 (Backspace or Delete) 触发
       */
      canDeleteNode(ctx, node) {

        if (isNodeOffSpringOfFixedGraph(node, ctx)) {
          return false
        }

        if (isNodeOffspringOfCompositeNode(node, ctx)) {
          return false
        }

        return true;
      },
      /**
       * Whether allow dragging into the container node
       * 是否允许拖入容器节点
       */
      canDropToNode: (ctx, params) => {
        const { dragNode, dropNode } = params;
        // Nested container parent nodes cannot be dragged into child nodes
        if (dropNode?.parent?.id === dragNode?.id) {
          return false;
        }

        if (isFxiedGraphNode(dropNode?.id, ctx) || isCompositeNode(dropNode?.id, ctx)) {
          return false;
        }
        return true;
      },
      /**
       * Drag the end of the line to create an add panel (feature optional)
       * 拖拽线条结束需要创建一个添加面板 （功能可选）
       */
      onDragLineEnd: undefined,
      /**
       * SelectBox config
       */
      selectBox: {
        SelectorBoxPopover,
      },
      materials: {
        /**
         * Render Node
         */
        renderDefaultNode: BaseNode,
        renderNodes: {
          [WorkflowNodeType.Comment]: CommentRender,
        },
      },
      /**
       * Node engine enable, you can configure formMeta in the FlowNodeRegistry
       */
      nodeEngine: {
        enable: true,
      },
      /**
       * Variable engine enable
       */
      variableEngine: {
        enable: true,
      },
      /**
       * Redo/Undo enable
       */
      history: {
        enable: true,
        enableChangeNode: true, // Listen Node engine data change
      },
      /**
       * Content change
       */
      onContentChange: debounce((ctx, event) => {
        //console.log("Auto Save: ", event, ctx.document.toJSON());
      }, 1000),
      /**
       * Running line
       */
      isFlowingLine: (ctx, line) => ctx.get(RunningService).isFlowingLine(line),
      /**
       * Shortcuts
       */
      shortcuts,
      /**
       * Bind custom service
       */
      onBind: ({ bind }) => {
        bind(CustomService).toSelf().inSingletonScope();
        bind(RunningService).toSelf().inSingletonScope();
      },
      /**
       * Playground init
       */
      onInit() {
        console.log("--- Playground init ---");
      },
      /**
       * Playground render
       */
      onAllLayersRendered(ctx) {
        //  Fitview
        ctx.document.fitView(false);
        console.log("--- Playground rendered ---");
      },
      /**
       * Playground dispose
       */
      onDispose() {
        //console.log("---- Playground Dispose ----");
      },
      plugins: () => [
        /**
         * Line render plugin
         * 连线渲染插件
         */
        createFreeLinesPlugin({
          renderInsideLine: undefined  //LineAddButton,
        }),
        /**
         * Minimap plugin
         * 缩略图插件
         */
        createMinimapPlugin({
          disableLayer: true,
          canvasStyle: {
            canvasWidth: 182,
            canvasHeight: 102,
            canvasPadding: 50,
            canvasBackground: "rgba(242, 243, 245, 1)",
            canvasBorderRadius: 10,
            viewportBackground: "rgba(255, 255, 255, 1)",
            viewportBorderRadius: 4,
            viewportBorderColor: "rgba(6, 7, 9, 0.10)",
            viewportBorderWidth: 1,
            viewportBorderDashLength: undefined,
            nodeColor: "rgba(0, 0, 0, 0.10)",
            nodeBorderRadius: 2,
            nodeBorderWidth: 0.145,
            nodeBorderColor: "rgba(6, 7, 9, 0.10)",
            overlayColor: "rgba(255, 255, 255, 0.55)",
          },
          //inactiveDebounceTime: 1,
        }),
        /**
         * Variable plugin
         * 变量插件
         */
        createSyncVariablePlugin({}),
        /**
         * Snap plugin
         * 自动对齐及辅助线插件
         */
        createFreeSnapPlugin({
          edgeColor: "#00B2B2",
          alignColor: "#00B2B2",
          edgeLineWidth: 1,
          alignLineWidth: 1,
          alignCrossWidth: 8,
        }),
        /**
         * NodeAddPanel render plugin
         * 节点添加面板渲染插件
         */
        createFreeNodePanelPlugin({
          renderer: NodePanel,
        }),
        /**
         * This is used for the rendering of the loop node sub-canvas
         * 这个用于 loop 节点子画布的渲染
         */
        createContainerNodePlugin({}),
        createFreeGroupPlugin({
          groupNodeRender: GroupNodeRender,
        }),
      ],
    };
  }, [initialData, nodeRegistries]);
}

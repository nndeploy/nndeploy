import {
  FreeLayoutPluginContext,
  Rectangle,
  ShortcutsHandler,
  TransformData,
  WorkflowDocument,
  WorkflowEdgeJSON,
  WorkflowJSON,
  WorkflowLineEntity,
  WorkflowNodeEntity,
  WorkflowNodeJSON,
  WorkflowNodeLinesData,
  WorkflowNodeMeta,
  WorkflowSelectService,
} from '@flowgram.ai/free-layout-editor';
import { Toast } from '@douyinfe/semi-ui';

import type {
  WorkflowClipboardRect,
  WorkflowClipboardSource,
  WorkflowClipboardData,
} from '../type';
import { FlowCommandId, WorkflowClipboardDataID } from '../constants';
import { WorkflowNodeType } from '../../nodes';

export class CopyShortcut implements ShortcutsHandler {
  public commandId = FlowCommandId.COPY;

  public shortcuts = ['meta c', 'ctrl c'];

  private document: WorkflowDocument;

  private selectService: WorkflowSelectService;

  constructor(context: FreeLayoutPluginContext) {
    this.document = context.get(WorkflowDocument);
    this.selectService = context.get(WorkflowSelectService);
    this.execute = this.execute.bind(this);
  }

  /**
   * execute copy operation - 执行复制操作
   */
  public async execute(): Promise<void> {
    if (await this.hasSelectedText()) {
      return;
    }
    if (!this.isValid(this.selectedNodes)) {
      return;
    }
    const data = this.toClipboardData();
    await this.write(data);
  }

  /**
   *  has selected text - 是否有文字被选中
   */
  private async hasSelectedText(): Promise<boolean> {
    if (!window.getSelection()?.toString()) {
      return false;
    }
    await navigator.clipboard.writeText(window.getSelection()?.toString() ?? '');
    Toast.success({
      content: 'Text copied',
    });
    return true;
  }

  /**
   * get selected nodes - 获取选中的节点
   */
  private get selectedNodes(): WorkflowNodeEntity[] {
    return this.selectService.selection.filter(
      (n) => n instanceof WorkflowNodeEntity
    ) as WorkflowNodeEntity[];
  }

  /**
   * validate selected nodes - 验证选中的节点
   */
  private isValid(nodes: WorkflowNodeEntity[]): boolean {
    if (nodes.length === 0) {
      Toast.warning({
        content: 'No nodes selected',
      });
      return false;
    }
    return true;
  }

  /**
   * create clipboard data - 转换为剪贴板数据
   */
  toClipboardData(nodes?: WorkflowNodeEntity[]): WorkflowClipboardData {
    const validNodes = this.getValidNodes(nodes ? nodes : this.selectedNodes);
    const source = this.toSource();
    const json = this.toJSON(validNodes);
    const bounds = this.getEntireBounds(validNodes);
    return {
      type: WorkflowClipboardDataID,
      source,
      json,
      bounds,
    };
  }

  /**
   * get valid nodes - 获取有效的节点
   */
  private getValidNodes(nodes: WorkflowNodeEntity[]): WorkflowNodeEntity[] {
    return nodes.filter((n) => {
      if (
        [WorkflowNodeType.Start, WorkflowNodeType.End].includes(n.flowNodeType as WorkflowNodeType)
      ) {
        return false;
      }
      if (n.getNodeMeta<WorkflowNodeMeta>().copyDisable) {
        return false;
      }
      return true;
    });
  }

  /**
   * get source data - 获取来源数据
   */
  private toSource(): WorkflowClipboardSource {
    return {
      host: window.location.host,
    };
  }

  /**
   * convert nodes to JSON - 将节点转换为JSON
   */
  private toJSON(nodes: WorkflowNodeEntity[]): WorkflowJSON {
    const nodeJSONs = this.getNodeJSONs(nodes);
    const edgeJSONs = this.getEdgeJSONs(nodes);
    return {
      nodes: nodeJSONs,
      edges: edgeJSONs,
    };
  }

  /**
   * get JSON representation of nodes - 获取节点的JSON表示
   */
  private getNodeJSONs(nodes: WorkflowNodeEntity[]): WorkflowNodeJSON[] {
    const nodeJSONs = nodes.map((node) => {
      const nodeJSON = this.document.toNodeJSON(node);
      if (!nodeJSON.meta?.position) {
        return nodeJSON;
      }
      const { bounds } = node.getData(TransformData);
      // Use absolute positioning as coordinates - 使用绝对定位作为坐标
      nodeJSON.meta.position = {
        x: bounds.x,
        y: bounds.y,
      };
      return nodeJSON;
    });
    return nodeJSONs.filter(Boolean);
  }

  /**
   * get edges of all nodes - 获取所有节点的边
   */
  private getEdgeJSONs(nodes: WorkflowNodeEntity[]): WorkflowEdgeJSON[] {
    const lineSet = new Set<WorkflowLineEntity>();
    const nodeIdSet = new Set(nodes.map((n) => n.id));
    nodes.forEach((node) => {
      const linesData = node.getData(WorkflowNodeLinesData);
      const lines = [...linesData.inputLines, ...linesData.outputLines];
      lines.forEach((line) => {
        if (nodeIdSet.has(line.from.id) && line.to?.id && nodeIdSet.has(line.to.id)) {
          lineSet.add(line);
        }
      });
    });
    return Array.from(lineSet).map((line) => line.toJSON());
  }

  /**
   * get bounding rectangle of all nodes - 获取所有节点的边界矩形
   */
  private getEntireBounds(nodes: WorkflowNodeEntity[]): WorkflowClipboardRect {
    const bounds = nodes.map((node) => node.getData<TransformData>(TransformData).bounds);
    const rect = Rectangle.enlarge(bounds);
    return {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    };
  }

  /**
   * write data to clipboard - 将数据写入剪贴板
   */
  private async write(data: WorkflowClipboardData): Promise<void> {
    try {
      await navigator.clipboard.writeText(JSON.stringify(data));
      this.notifySuccess();
    } catch (err) {
      console.error('Failed to write text: ', err);
    }
  }

  /**
   * show success notification - 显示成功通知
   */
  private notifySuccess(): void {
    const nodeTypes = this.selectedNodes.map((node) => node.flowNodeType);
    if (nodeTypes.includes('start') || nodeTypes.includes('end')) {
      Toast.warning({
        content:
          'The Start/End node cannot be duplicated, other nodes have been copied to the clipboard',
        showClose: false,
      });
      return;
    }
    Toast.success({
      content: 'Nodes have been copied to the clipboard',
      showClose: false,
    });
    return;
  }
}

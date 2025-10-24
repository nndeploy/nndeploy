import { FC, useCallback, useState, type MouseEvent } from 'react';

import {
  delay,
  useClientContext,
  useService,
  WorkflowDragService,
  WorkflowNodeEntity,
  WorkflowSelectService,
} from '@flowgram.ai/free-layout-editor';
import { NodeIntoContainerService } from '@flowgram.ai/free-container-plugin';
import { IconButton, Dropdown } from '@douyinfe/semi-ui';
import { IconMore } from '@douyinfe/semi-icons';

import { FlowNodeRegistry } from '../../typings';
import { PasteShortcut } from '../../shortcuts/paste';
import { CopyShortcut } from '../../shortcuts/copy';
import { getNextNameNumberSuffix, isNodeOffspringOfCompositeNode, isNodeOffSpringOfFixedGraph } from '../../pages/components/flow/functions';

interface NodeMenuProps {
  node: WorkflowNodeEntity;
  deleteNode: () => void;
}

export const NodeMenu: FC<NodeMenuProps> = ({ node, deleteNode }) => {
  const [visible, setVisible] = useState(true);
  const clientContext = useClientContext();
  const registry = node.getNodeRegistry<FlowNodeRegistry>();
  const nodeIntoContainerService = useService(NodeIntoContainerService);
  const selectService = useService(WorkflowSelectService);
  const dragService = useService(WorkflowDragService);
  isNodeOffSpringOfFixedGraph(node, clientContext) || isNodeOffspringOfCompositeNode(node, clientContext)
  let canMoveOut = nodeIntoContainerService.canMoveOutContainer(node);

  let canCreateCopy = !((isNodeOffSpringOfFixedGraph(node, clientContext)
    || isNodeOffspringOfCompositeNode(node, clientContext)))

  if (isNodeOffSpringOfFixedGraph(node, clientContext)
    || isNodeOffspringOfCompositeNode(node, clientContext)) {
    canMoveOut = false
  }

  const rerenderMenu = useCallback(() => {
    // force destroy component - 强制销毁组件触发重新渲染
    setVisible(false);
    requestAnimationFrame(() => {
      setVisible(true);
    });
  }, []);

  const handleMoveOut = useCallback(
    async (e: MouseEvent) => {
      e.stopPropagation();
      const sourceParent = node.parent;
      // move out of container - 移出容器
      nodeIntoContainerService.moveOutContainer({ node });
      // clear invalid lines - 清除非法线条
      await nodeIntoContainerService.clearInvalidLines({
        dragNode: node,
        sourceParent,
      });
      rerenderMenu();
      await delay(16);
      // select node - 选中节点
      selectService.selectNode(node);
      // start drag node - 开始拖拽
      dragService.startDragSelectedNodes(e);
    },
    [nodeIntoContainerService, node, rerenderMenu]
  );

  const handleCopy = useCallback(
    (e: React.MouseEvent) => {
      const copyShortcut = new CopyShortcut(clientContext);
      const pasteShortcut = new PasteShortcut(clientContext);
      const data = copyShortcut.toClipboardData([node]);

      const nextNameNumberPart = getNextNameNumberSuffix(clientContext.document.toJSON() as any);

      data.json.nodes.map(node => {
        let name = ''
        ///@ts-ignore
        var parts = node.type.split('::')
        if (parts.length > 0) {
          name = parts[parts.length - 1]
        }

        name = name + '_' + nextNameNumberPart

        node.data.name_ = name;
      })

      pasteShortcut.apply(data);
      e.stopPropagation(); // Disable clicking prevents the sidebar from opening
    },
    [clientContext, node]
  );

  const handleDelete = useCallback(
    (e: React.MouseEvent) => {
      deleteNode();
      e.stopPropagation(); // Disable clicking prevents the sidebar from opening
    },
    [clientContext, node]
  );

  if (!visible) {
    return;
  }

  return (
    <Dropdown
      trigger="hover"
      position="bottomRight"
      render={
        <Dropdown.Menu>
          {canMoveOut && <Dropdown.Item onClick={handleMoveOut}>Move out</Dropdown.Item>}
          {canCreateCopy && <Dropdown.Item onClick={handleCopy} disabled={registry.meta!.copyDisable === true}>
            Create Copy
          </Dropdown.Item>
          }
          <Dropdown.Item
            onClick={handleDelete}
            disabled={!!(!registry.canDelete?.(clientContext, node) || registry.meta!.deleteDisable)}
          >
            Delete
          </Dropdown.Item>


        </Dropdown.Menu>
      }
    >
      <IconButton
        color="secondary"
        size="small"
        theme="borderless"
        icon={<IconMore />}
        onClick={(e) => e.stopPropagation()}
      />
    </Dropdown>
  );
};

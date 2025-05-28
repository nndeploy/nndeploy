import {
  Button,
  Dropdown,
  SideSheet,
  Tree,
  Typography,
} from "@douyinfe/semi-ui";
import { useGetTree } from "./effect";
import { IconMore, IconEyeClosed, IconEyeOpened } from "@douyinfe/semi-icons";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { ReactNode, useState } from "react";
import "./index.scss";
import Preview from "./Preview";
import { ResourceTreeNodeData } from "./entity";
import BranchDrawer, { BranchDrawerProps } from "./Branch";

const Resource: React.FC = () => {
  const [treeData, setTreeData] = useGetTree();

  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewItem, setPreviewItem] = useState<ResourceTreeNodeData>();

  function handlePreviewClose() {
    setPreviewVisible(false);
  }

  const [branchVisible, setBranchVisible] = useState(false);
  const [branchDrawerProps, setBranchDrawerProps] =
    useState<Partial<BranchDrawerProps>>();

  function handleBranchClose() {
    setBranchVisible(false);
  }

  const addNode = (parentKey: string, newNode: TreeNodeData[]) => {
    const addNodeRecursively = (nodes: TreeNodeData[]) => {
      return nodes.map((node) => {
        if (node.key === parentKey) {
          if (node.children) {
            node.children.push(newNode);
          } else {
            node.children = [newNode];
          }
        } else if (node.children) {
          node.children = addNodeRecursively(node.children);
        }
        return node;
      });
    };

    setTreeData(addNodeRecursively(treeData));
  };

  const deleteNode = (key: string) => {
    const deleteNodeRecursively = (nodes: TreeNodeData[]) => {
      return nodes
        .map((node) => {
          if (node.key === key) {
            return null;
          } else if (node.children) {
            node.children = deleteNodeRecursively(node.children).filter(
              (child) => child !== null
            );
          }
          return node;
        })
        .filter((node) => node !== null);
    };

    setTreeData(deleteNodeRecursively(treeData));
  };

  function onEditBranch(node: ResourceTreeNodeData) {
    setBranchDrawerProps({ currentNode: node });
  }
  function onAddBranch(parentNode: ResourceTreeNodeData) {
    setBranchDrawerProps({ parentNode });
  }

  function onAddBranchClose() {
    setBranchVisible(false);
  }

  function onAddBranchSure(node: ResourceTreeNodeData) {
    setBranchVisible(false);
  }

    function onEditResource(item: ResourceTreeNodeData) {
    setPreviewItem(item);
    setPreviewVisible(true);
  }

  

  const renderBtn = (node: ResourceTreeNodeData) => {
    return (
      <Dropdown
        closeOnEsc={true}
        trigger={"click"}
        position="right"
        render={
          <Dropdown.Menu>
            {node.type == "branch" && (
              <Dropdown.Item onClick={() => onEditBranch(node)}>
                edit
              </Dropdown.Item>
            )}
            {node.type == "leaf" && (
              <Dropdown.Item onClick={() => onEditResource(node)}>
                edit
              </Dropdown.Item>
            )}

            <Dropdown.Item onClick={() => onAddBranch(node)}>
              add children branch
            </Dropdown.Item>
            {node.type == "branch" && (
              <Dropdown.Item onClick={() => onEditResource(node)}>
                add resource
              </Dropdown.Item>
            )}
            <Dropdown.Item onClick={() => deleteNode(node.key!)}>
              delete
            </Dropdown.Item>
            <Dropdown.Item></Dropdown.Item>
          </Dropdown.Menu>
        }
      >
        {/* <Button
          onClick={(e) => {
            //Toast.info({ content });
            e.stopPropagation();
          }}
          icon={<IconMore />}
          size="small"
        /> */}

        <IconMore />
      </Dropdown>
    );
  };



  const renderLabel = (label: ReactNode, item: ResourceTreeNodeData) => (
    <div
      style={{ display: "flex", height: "24px" }}
      draggable
      ///@ts-ignore
      onDragStart={(dragEvent) => onDragStart(item!, dragEvent)}
    >
      <Typography.Text
        ellipsis={{ showTooltip: true }}
        style={{ width: "calc(100% - 48px)" }}
        className="label"
      >
        {label}
      </Typography.Text>
      <div className="operate">
        {/* {
          item.type == 'leaf' && <IconEyeOpened  onClick={()=>onShowPreview(item)}/>
        } */}

        {renderBtn(item!)}
      </div>
    </div>
  );
  return (
    <div className="tree-resource">
      <Tree
        treeData={treeData}
        ///@ts-ignore
        renderLabel={renderLabel}
        className="tree-node"
        //draggable
      />
      <SideSheet
        width={"30%"}
        visible={previewVisible}
        onCancel={handlePreviewClose}
        title={previewItem?.label}
      >
        <Preview {...previewItem!} />
      </SideSheet>

      <SideSheet
        width={"30%"}
        visible={branchVisible}
        onCancel={handleBranchClose}
        title={
          (branchDrawerProps?.parentNode?.label ?? "") + "add children branch"
        }
      >
        <BranchDrawer
          {...branchDrawerProps}
          onSure={onAddBranchSure}
          onClose={onAddBranchClose}
        />
      </SideSheet>
    </div>
  );
};

export default Resource;

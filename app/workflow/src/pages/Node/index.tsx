import React, { ReactNode, useState } from "react";

import {
  Dropdown,
  Popconfirm,
  SideSheet,
  Tooltip,
  Tree,
  Typography,
} from "@douyinfe/semi-ui";

import "./index.scss";
import {
  INodeBranchEntity,
  INodeEntity,
  INodeTreeNodeEntity,
  NodeTreeNodeData,
} from "./entity";
import { apiGetNodeById, apiNodeBranchDelete, apiNodeDelete } from "./api";
import { IconInherit, IconMore, IconPlus, IconPlusCircle } from "@douyinfe/semi-icons";
import { useGetNodeTree } from "./effect";
import { IResponse } from "../../request/types";
import BranchEditDrawer from "./BranchEditDrawer";
import NodeEditDrawer from "./NodeEditDrawer";
const { Text, Paragraph } = Typography;
const NodePage: React.FC = () => {
  const { flatData, setFlatData, treeData, getNodeTree } = useGetNodeTree();

  const [rightType, setRightType] = useState<"branch" | "leaf" | "">();

  const [branchEdit, setBranchEdit] = useState<INodeBranchEntity>();

  const [nodeEdit, setNodeEdit] = useState<INodeEntity>();

  function onBranchEdit(node: INodeBranchEntity) {
    setBranchEdit(node);
    setRightType("branch");
  }
  async function onNodeEdit(item: INodeTreeNodeEntity) {
    //setNodeEdit(item);
    if (item.id) {
      const response = await apiGetNodeById(item.id);
      if (response.flag == "success") {
        setNodeEdit(response.result);
      }
    } else {
      setNodeEdit({
        ...item,
        schema: {
          type: "object",
          properties: {},
          required: [],
        },
      });
    }
    setRightType("leaf");
  }

  async function deleteNode(item: INodeTreeNodeEntity) {
    function findDescendantsIncludingSelf(
      flatData: INodeTreeNodeEntity[],
      id: string
    ): INodeTreeNodeEntity[] {
      const descendants: INodeTreeNodeEntity[] = [];

      function findChildren(parentId: string) {
        flatData.forEach((node) => {
          if (node.parentId === parentId) {
            descendants.push(node);
            findChildren(node.id);
          }
        });
      }

      const self = flatData.find((node) => node.id === item.id);
      if (self) {
        descendants.push(self);
        findChildren(id);
      }

      return descendants;
    }

    let response: IResponse<any>;

    if (item.type == "branch") {
      response = await apiNodeBranchDelete(item.id);
    } else {
      response = await apiNodeDelete(item.id);
    }

    if (response.flag == "success") {
      var toDeleteIds = findDescendantsIncludingSelf(flatData, item.id).map(
        (item) => item.id
      );
      var newFlatData = flatData.filter(
        (item) => !toDeleteIds.includes(item.id)
      );
      setFlatData(newFlatData);
    }
  }

  const renderBtn = (resource: INodeTreeNodeEntity) => {
    return (
      <Dropdown
        closeOnEsc={true}
        trigger={"click"}
        position="right"
        render={
          <Dropdown.Menu>
            {resource.type == "branch" && (
              <Dropdown.Item onClick={() => onBranchEdit(resource)}>
                edit
              </Dropdown.Item>
            )}
            {resource.type == "leaf" && (
              <Dropdown.Item onClick={() => onNodeEdit(resource)}>
                edit
              </Dropdown.Item>
            )}
            {resource.type == "branch" && (
              <Dropdown.Item
                onClick={() =>
                  onBranchEdit({
                    id: "",
                    name: "",
                    parentId: resource.id,
                  })
                }
              >
                add children branch
              </Dropdown.Item>
            )}
            {resource.type == "branch" && (
              <Dropdown.Item
                onClick={() =>
                  onNodeEdit({
                    id: "",
                    name: "",
                    parentId: resource.id,
                    type: "leaf",
                  })
                }
              >
                add node
              </Dropdown.Item>
            )}
            <Dropdown.Item>
              <Popconfirm
                title="Are you sure?"
                content="Are you sure to delete this item?"
                onConfirm={() => deleteNode(resource)}
                onCancel={() => {}}
              >
                delete
              </Popconfirm>
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

  const renderLabel = (label: ReactNode, item: NodeTreeNodeData) => (
    <div
      style={{ display: "flex", height: "24px" }}
      draggable
      ///@ts-ignore
      //onDragStart={(dragEvent) => onDragStart(item!, dragEvent)}
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

        {renderBtn(item.nodeEntity)}
      </div>
    </div>
  );

  const addNode = (newNode: INodeTreeNodeEntity) => {
    var resultData: INodeTreeNodeEntity[] = [];
    const findIndex = flatData.findIndex((item) => item.id == newNode.id);
    if (findIndex > -1) {
      resultData = [
        ...flatData.slice(0, findIndex),
        newNode,
        ...flatData.slice(findIndex + 1),
      ];
    } else {
      resultData = [...flatData, newNode];
    }
    setFlatData(resultData);
  };

  function onBranchEditSure(node: INodeBranchEntity) {
    addNode({ ...node, type: "branch" });
    setRightType("");
  }
  function onBranchEditClose() {
    setRightType("");
  }

  function onNodeEditSure(node: INodeBranchEntity) {
    addNode({ ...node, type: "leaf" });
    setRightType("");
  }
  function onNodeEditClose() {
    setRightType("");
  }

  return (
    <div className="page-node">
      <div className="tree-pane">
        <div className="tree-pane-header">
          <Text>node tree</Text>

          <div className="operate">
            <Tooltip content="add branch" position="top">
              <Text
                link
                icon={<IconPlus />}
                onClick={() => onBranchEdit({ id: "", name: "", parentId: "" })}
              ></Text>
            </Tooltip>
            <Tooltip content="add node" position="top">
              <Text
                link
                icon={<IconInherit />}
                onClick={() =>
                  onNodeEdit({
                    id: "",
                    name: "",
                    parentId: "",
                    type: "leaf",
                  })
                }
              ></Text>
            </Tooltip>
          </div>
        </div>
        <Tree
          treeData={treeData}
          ///@ts-ignore
          renderLabel={renderLabel}
          className="tree-node"
          // onSelect={(
          //   selectedKey: string,
          //   selected: boolean,
          //   selectedNode: TreeNodeData
          // ) => {
          //   props.onSelect(selectedKey);
          // }}
          //draggable
        />
      </div>
      <div className="main">
        {rightType == "branch" ? (
          <BranchEditDrawer
            entity={branchEdit!}
            onSure={onBranchEditSure}
            onClose={onBranchEditClose}
          />
        ) : rightType == "leaf" ? (
          <NodeEditDrawer
            entity={nodeEdit!}
            onSure={onNodeEditSure}
            onClose={onNodeEditClose}
          />
        ) : (
          <></>
        )}
      </div>
    </div>
  );
};

export default NodePage;

import {
  Dropdown,
  Popconfirm,
  SideSheet,
  Tooltip,
  Tree,
  Typography,
} from "@douyinfe/semi-ui";
import { IconMore, IconPlus } from "@douyinfe/semi-icons";
import { ReactNode, useState } from "react";

import "./index.scss";
import NodeEditDrawer from "../NodeEditDrawer";

import BranchEditDrawer from "../BranchEditDrawer";
import { useGetNoteBranch } from "./effect";
import {
  INodeBranchEntity,
  INodeEntity,
  NodeTreeNodeData,
} from "../entity";
import { apiNodeBranchDelete } from "./api";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";

interface NodeTreeProps {
  onSelect: (key: string) => void;
}
const { Text, Paragraph } = Typography;
const NodeTree: React.FC<NodeTreeProps> = (props) => {
  const { flatData, setFlatData, treeData } = useGetNoteBranch();

  const [nodeEditVisible, setNodeEditVisible] = useState(false);
  const [nodeEdit, setNodeEdit] = useState<INodeEntity>();

  function handleResoureDrawerClose() {
    setNodeEditVisible(false);
  }

  const [branchVisible, setBranchVisible] = useState(false);
  const [branchEdit, setBranchEdit] = useState<INodeBranchEntity>();

  function handleBranchClose() {
    setBranchVisible(false);
  }

  const addBranchNode = (newNode: INodeBranchEntity) => {
    var resultData: INodeBranchEntity[] = [];
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

  async function deleteNodeBranch(id: string) {
    function findDescendantsIncludingSelf(
      flatData: INodeBranchEntity[],
      id: string
    ): INodeBranchEntity[] {
      const descendants: INodeBranchEntity[] = [];

      function findChildren(parentId: string) {
        flatData.forEach((node) => {
          if (node.parentId === parentId) {
            descendants.push(node);
            findChildren(node.id);
          }
        });
      }

      const self = flatData.find((node) => node.id === id);
      if (self) {
        descendants.push(self);
        findChildren(id);
      }

      return descendants;
    }

    const response = await apiNodeBranchDelete(id);

    if (response.flag == "success") {
      var toDeleteIds = findDescendantsIncludingSelf(flatData, id).map(
        (item) => item.id
      );
      var newFlatData = flatData.filter(
        (item) => !toDeleteIds.includes(item.id)
      );
      setFlatData(newFlatData);
    }
  }

  function onBranchEdit(node: INodeBranchEntity) {
    setBranchEdit(node);
    setBranchVisible(true);
  }
  function onAddBranch(node: INodeBranchEntity) {
    setBranchEdit(node);
    setBranchVisible(true);
  }

  function onBranchEditClose() {
    setBranchVisible(false);
  }

  function onBranchEditSure(node: INodeBranchEntity) {
    addBranchNode(node);
    setBranchVisible(false);
  }

  function onNodeEdit(item: INodeEntity) {
    setNodeEdit(item);
    setNodeEditVisible(true);
  }

  function onNodeEditDrawerSure(node: INodeEntity) {
    setNodeEditVisible(false);
  }

  function onNodeEditDrawerClose() {
    setNodeEditVisible(false);
  }

  const renderBtn = (resource: INodeBranchEntity) => {
    return (
      <Dropdown
        closeOnEsc={true}
        trigger={"click"}
        position="right"
        render={
          <Dropdown.Menu>
            <Dropdown.Item onClick={() => onBranchEdit(resource)}>
              edit
            </Dropdown.Item>

            <Dropdown.Item
              onClick={() =>
                onAddBranch({ id: "", name: "", parentId: resource.id })
              }
            >
              add children branch
            </Dropdown.Item>

            <Dropdown.Item
              onClick={() =>
                onNodeEdit({
                  key_: "",
                  name_: "",
                  inputs_: resource.id,
                  schema: {
                    type: "object",
                    properties: {},
                    required: [],
                  },
                })
              }
            >
              add node
            </Dropdown.Item>

            <Dropdown.Item>
              <Popconfirm
                title="Are you sure?"
                content="Are you sure to delete this item?"
                onConfirm={() => deleteNodeBranch(resource.id)}
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

        {renderBtn(item.entity)}
      </div>
    </div>
  );

  function onTreeSelect(
    selectedKey: string,
    selected: boolean,
    selectedNode: TreeNodeData
  ) {}
  return (
    <div className="tree-pane">
      <div className="tree-pane-header">
        <Text>resources</Text>
        <Tooltip content="add branch" position="top">
          <Text
            link
            icon={<IconPlus />}
            onClick={() => onBranchEdit({ id: "", name: "", parentId: "" })}
          ></Text>
        </Tooltip>
      </div>
      <Tree
        treeData={treeData}
        ///@ts-ignore
        renderLabel={renderLabel}
        className="tree-node"
        onSelect={(
          selectedKey: string,
          selected: boolean,
          selectedNode: TreeNodeData
        ) => {
          props.onSelect(selectedKey);
        }}
        //draggable
      />
      <SideSheet
        width={"calc(100% - 200px - 17px )"}
        visible={nodeEditVisible}
        onCancel={handleResoureDrawerClose}
        title={nodeEdit?.name_ ?? "add"}
      >
        <NodeEditDrawer
          entity={nodeEdit!}
          onSure={onNodeEditDrawerSure}
          onClose={onNodeEditDrawerClose}
        />
      </SideSheet>

      <SideSheet
        width={"30%"}
        visible={branchVisible}
        onCancel={handleBranchClose}
        title={(branchEdit?.name ?? "") + "add children branch"}
      >
        <BranchEditDrawer
          entity={branchEdit!}
          onSure={onBranchEditSure}
          onClose={onBranchEditClose}
        />
      </SideSheet>
    </div>
  );
};

export default NodeTree;

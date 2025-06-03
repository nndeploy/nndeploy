import {
  Dropdown,
  Popconfirm,
  SideSheet,
  Tooltip,
  Tree,
  Typography,
} from "@douyinfe/semi-ui";
import { useGetTree } from "./effect";
import { IconMore, IconPlus } from "@douyinfe/semi-icons";
import { ReactNode, useState } from "react";

import "./index.scss";
import ResourceEditDrawer from "./ResourceEditDrawer";
import { IResourceEntity, IResourceTreeNodeEntity, ResourceTreeNodeData } from "./entity";
import BranchEditDrawer from "./BranchEditDrawer";
import { apiResourceDelete } from "./api";

const { Text, Paragraph } = Typography;
const Resource: React.FC = () => {
  const { flatData, setFlatData, treeData } = useGetTree();

  const [resoureEditVisible, setResourceEditVisible] = useState(false);
  const [resourceEdit, setResourceEdit] = useState<IResourceTreeNodeEntity>();

  function handleResoureDrawerClose() {
    setResourceEditVisible(false);
  }

  const [branchVisible, setBranchVisible] = useState(false);
  const [branchEdit, setBranchEdit] = useState<IResourceTreeNodeEntity>();

  function handleBranchClose() {
    setBranchVisible(false);
  }

  const addNode = (newNode: IResourceTreeNodeEntity) => {
    var resultData: IResourceTreeNodeEntity[] = [];
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

  async function deleteNode(id: string) {
    function findDescendantsIncludingSelf(
      flatData: IResourceTreeNodeEntity[],
      id: string
    ): IResourceTreeNodeEntity[] {
      const descendants: IResourceTreeNodeEntity[] = [];

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

    const response = await apiResourceDelete(id);

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

  function onBranchEdit(node: IResourceTreeNodeEntity) {
    setBranchEdit(node);
    setBranchVisible(true);
  }
  function onAddBranch(node: IResourceTreeNodeEntity) {
    setBranchEdit(node);
    setBranchVisible(true);
  }

  function onBranchEditClose() {
    setBranchVisible(false);
  }

  function onBranchEditSure(resource: IResourceTreeNodeEntity) {
    addNode(resource);
    setBranchVisible(false);
  }

  function onResourceEdit(item: IResourceTreeNodeEntity) {
    setResourceEdit(item);
    setResourceEditVisible(true);
  }

  function onResourceEditDrawerSure(resource: IResourceTreeNodeEntity) {
    addNode(resource);
    setResourceEditVisible(false);
  }

  function onResourceEditDrawerClose() {
    setResourceEditVisible(false);
  }

  const renderBtn = (resource: IResourceTreeNodeEntity) => {
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
              <Dropdown.Item onClick={() => onResourceEdit(resource)}>
                edit
              </Dropdown.Item>
            )}
            {resource.type == "branch" && (
              <Dropdown.Item
                onClick={() =>
                  onAddBranch({ id: "", name: "", parentId: resource.id,type: "branch" })
                }
              >
                add children branch
              </Dropdown.Item>
            )}
            {resource.type == "branch" && (
              <Dropdown.Item
                onClick={() =>
                  onResourceEdit({ id: "", name: "", parentId: resource.id, type: "leaf"  })
                }
              >
                add resource
              </Dropdown.Item>
            )}
            <Dropdown.Item>
              <Popconfirm
                title="Are you sure?"
                content="Are you sure to delete this item?"
                onConfirm={() => deleteNode(resource.id)}
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

  const renderLabel = (label: ReactNode, item: ResourceTreeNodeData) => (
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
  return (
    <div className="tree-resource">
      <div className="tree-resource-header">
        <Text>resources</Text>
        <Tooltip content="add branch" position="top">
          <Text
            link
            icon={<IconPlus />}
            onClick={() => onBranchEdit({ id: "", name: "", parentId: "", type: "branch"  })}
          ></Text>
        </Tooltip>
      </div>
      <Tree
        treeData={treeData}
        ///@ts-ignore
        renderLabel={renderLabel}
        className="tree-node"
        //draggable
      />
      <SideSheet
        width={"30%"}
        visible={resoureEditVisible}
        onCancel={handleResoureDrawerClose}
        title={resourceEdit?.name ?? "add"}
      >
        <ResourceEditDrawer
          entity={resourceEdit!}
          onSure={onResourceEditDrawerSure}
          onClose={onResourceEditDrawerClose}
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

export default Resource;

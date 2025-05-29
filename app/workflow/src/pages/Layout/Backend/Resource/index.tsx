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
import { IResourceEntity, ResourceTreeNodeData } from "./entity";
import BranchEditDrawer from "./BranchEditDrawer";
import { apiResourceDelete } from "./api";

const { Text, Paragraph } = Typography;
const Resource: React.FC = () => {
  const { flatData, setFlatData, treeData } = useGetTree();

  const [resoureEditVisible, setResourceEditVisible] = useState(false);
  const [resourceEdit, setResourceEdit] = useState<IResourceEntity>();

  function handleResoureDrawerClose() {
    setResourceEditVisible(false);
  }

  const [branchVisible, setBranchVisible] = useState(false);
  const [branchEdit, setBranchEdit] = useState<IResourceEntity>();

  function handleBranchClose() {
    setBranchVisible(false);
  }

  const addNode = (newNode: IResourceEntity) => {
    var resultData: IResourceEntity[] = [];
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
      flatData: IResourceEntity[],
      id: string
    ): IResourceEntity[] {
      const descendants: IResourceEntity[] = [];

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

  function onBranchEdit(node: IResourceEntity) {
    setBranchEdit(node);
    setBranchVisible(true);
  }
  function onAddBranch(node: IResourceEntity) {
    setBranchEdit(node);
    setBranchVisible(true);
  }

  function onBranchEditClose() {
    setBranchVisible(false);
  }

  function onBranchEditSure(resource: IResourceEntity) {
    addNode(resource);
    setBranchVisible(false);
  }

  function onResourceEdit(item: IResourceEntity) {
    setResourceEdit(item);
    setResourceEditVisible(true);
  }

  function onResourceEditDrawerSure(resource: IResourceEntity) {
    addNode(resource);
    setResourceEditVisible(false);
  }

  function onResourceEditDrawerClose() {
    setResourceEditVisible(false);
  }

  const renderBtn = (resource: IResourceEntity) => {
    return (
      <Dropdown
        closeOnEsc={true}
        trigger={"click"}
        position="right"
        render={
          <Dropdown.Menu>
            {!resource.isLeaf && (
              <Dropdown.Item onClick={() => onBranchEdit(resource)}>
                edit
              </Dropdown.Item>
            )}
            {resource.isLeaf && (
              <Dropdown.Item onClick={() => onResourceEdit(resource)}>
                edit
              </Dropdown.Item>
            )}
            {!resource.isLeaf && (
              <Dropdown.Item
                onClick={() =>
                  onAddBranch({ id: "", name: "", parentId: resource.id })
                }
              >
                add children branch
              </Dropdown.Item>
            )}
            {!resource.isLeaf && (
              <Dropdown.Item
                onClick={() =>
                  onResourceEdit({ id: "", name: "", parentId: resource.id })
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
            onClick={() => onBranchEdit({ id: "", name: "", parentId: "" })}
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

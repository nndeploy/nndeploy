import {
  Button,
  Dropdown,
  Popconfirm,
  SideSheet,
  Toast,
  Tooltip,
  Tree,
  Typography,
} from "@douyinfe/semi-ui";
import { useGetWorkflowTree } from "./effect";
import { IconMore, IconPlus } from "@douyinfe/semi-icons";
import { forwardRef, ReactNode, useContext, useEffect, useImperativeHandle, useState } from "react";

import "./index.scss";
import BranchEditDrawer from "./BranchEditDrawer";
import { IWorkFlowTreeNodeEntity, WorkFlowTreeNodeData } from "./entity";
import { apiWorkFlowDelete } from "./api";
import WorkFlowEditDrawer from "./WorkFlowEditDrawer";
import { PopconfirmWithInput } from "../../../components/PopconfirmWithInput";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { IResourceTreeNodeEntity } from "../Resource/entity";
import request from "../../../../request";
import store from "../store/store";

// export interface WorkFlowComponentHandle {
//   refresh: () => void;
// }

interface WorkFlowProps {
  onShowFlow: (node: TreeNodeData) => void;
  onFlowDeleteCallBack: (flowName: string) => void
}
const { Text, Paragraph } = Typography;
function WorkFlow (props: WorkFlowProps) {


  const {state} = useContext(store)
  const {freshFlowTreeCnt} = state

  const { onFlowDeleteCallBack } = props
  const { treeData, setTreeData, getWorkFlowTree } = useGetWorkflowTree();

  const [workFlowEditVisible, setWorkFlowEditVisible] = useState(false);
  const [workFlowEdit, setWorkFlowEdit] = useState<IWorkFlowTreeNodeEntity>();

  // useImperativeHandle(ref, () => ({
  //   refresh: getWorkFlowTree,
  // }));

  useEffect(()=>{
    getWorkFlowTree()
  }, [freshFlowTreeCnt])

  function handleResoureDrawerClose() {
    setWorkFlowEditVisible(false);
  }

  const [branchVisible, setBranchVisible] = useState(false);
  const [branchEdit, setBranchEdit] = useState<IWorkFlowTreeNodeEntity>();

  function handleBranchClose() {
    setBranchVisible(false);
  }

  const addNode = (newNode: TreeNodeData) => {
    var resultData: TreeNodeData[] = [];
    const findIndex = treeData.findIndex((item) => item.label == newNode.label);
    if (findIndex > -1) {
      resultData = [
        ...resultData.slice(0, findIndex),
        newNode,
        ...resultData.slice(findIndex + 1),
      ];
    } else {
      resultData = [...resultData, newNode];
    }
    setTreeData(resultData);
  };

  async function deleteNode(id: string) {

    const response = await apiWorkFlowDelete(id);

    if (response.flag == "success") {
      onFlowDeleteCallBack(id)
      const newData = treeData.filter(item => {
        return item.key != id
      })

      setTreeData(newData);
    }
  }

  // function onBranchEdit(node: IWorkFlowTreeNodeEntity) {
  //   setBranchEdit(node);
  //   setBranchVisible(true);
  // }
  // function onAddBranch(node: IWorkFlowTreeNodeEntity) {
  //   setBranchEdit(node);
  //   setBranchVisible(true);
  // }

  function onBranchEditClose() {
    setBranchVisible(false);
  }

  function onBranchEditSure(workFlow: IWorkFlowTreeNodeEntity) {
    addNode(workFlow);
    setBranchVisible(false);
  }

  // function onWorkFlowEdit(item: IWorkFlowTreeNodeEntity) {
  //   setWorkFlowEdit(item);
  //   setWorkFlowEditVisible(true);
  // }

  function onWorkFlowEditDrawerSure(workFlow: IWorkFlowTreeNodeEntity) {
    addNode(workFlow);
    setWorkFlowEditVisible(false);
  }

  function onWorkFlowEditDrawerClose() {
    setWorkFlowEditVisible(false);
  }

  function onUploadFlow() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const formData = new FormData();
      formData.append("file", file);
      try {
        const response = await request.upload("/api/workflow/upload", formData, {});
        if (response.flag === "success") {
          getWorkFlowTree();
          Toast.success("upload success")
        } else {
          // Handle error
          //console.error(response.msg);
          Toast.error("upload failed")
        }
      } catch (error) {
        console.error("Upload failed:", error);
      }
    };
    input.click();
  }
  function onDownload(flowId: string) {

    //const fileName = fileNames[workFlowName] ;
    const url = `/api/workflow/download/${flowId}`;
    request.download(url, {})
  }

  const renderBtn = (workFlow: TreeNodeData) => {
    return (
      <Dropdown
        closeOnEsc={true}
        trigger={"click"}
        position="right"
        render={
          <Dropdown.Menu>
            {/* {workFlow.type == "branch" && (
              <>
                <Dropdown.Item onClick={() => onBranchEdit(workFlow)}>
                  edit
                </Dropdown.Item>
              </>
            )}
            {workFlow.type == "leaf" && (
              <Dropdown.Item onClick={() => onWorkFlowEdit(workFlow)}>
                edit
              </Dropdown.Item>
            )} */}
            {/* {workFlow.type == "branch" && (
              <Dropdown.Item
                onClick={() =>
                  onAddBranch({
                    id: "",
                    name: "",
                    parentId: workFlow.id,
                    type: "branch",
                  })
                }
              >
                add children branch
              </Dropdown.Item>
            )} */}
            {/* {workFlow.type == "branch" && (
              <Dropdown.Item
                onClick={() =>
                  onWorkFlowEdit({
                    id: "",
                    name: "",
                    parentId: workFlow.id,
                    type: "leaf",
                  })
                }
              >
                add workFlow
              </Dropdown.Item>
            )} */}
            <Dropdown.Item>
              <Popconfirm
                title="Are you sure?"
                content="Are you sure to delete this item?"
                onConfirm={() => deleteNode(workFlow.key as string)}
                onCancel={() => { }}
              >
                delete
              </Popconfirm>
            </Dropdown.Item>
            <Dropdown.Item onClick={() => onDownload(workFlow.key as string)}>
              download
            </Dropdown.Item>
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

  const renderLabel = (label: ReactNode, item: TreeNodeData) => {


    return <div
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

        {renderBtn(item)}
      </div>
    </div>
  }
    ;

  function onSelect(
    selectedKey: string,
    selected: boolean,
    selectedNode: TreeNodeData
  ) {
    //const node = selectedNode as WorkFlowTreeNodeData;
    // const entity = node.entity;
    // if (entity.type == "branch") {
    //   return;
    // }
    props.onShowFlow(selectedNode);
  }
  return (
    <div className="tree-workflow">
      <div className="tree-workflow-header">
        <Text>Workflow</Text>
        <Text
          link
          icon={<IconPlus />}
          onClick={() => onUploadFlow()}
        ></Text>


        {/* <Tooltip content="add branch" position="top">
          <Text
            link
            icon={<IconPlus />}
            onClick={() =>
              onBranchEdit({ id: "", name: "", parentId: "", type: "branch" })
            }
          ></Text>
        </Tooltip> */}
      </div>
      <Tree
        treeData={treeData}
        onSelect={onSelect}
        ///@ts-ignore
        renderLabel={renderLabel}
        className="tree-node"
      //draggable
      />
      <SideSheet
        width={"30%"}
        visible={workFlowEditVisible}
        onCancel={handleResoureDrawerClose}
        title={workFlowEdit?.name ?? "add"}
      >
        <WorkFlowEditDrawer
          entity={workFlowEdit!}
          onSure={onWorkFlowEditDrawerSure}
          onClose={onWorkFlowEditDrawerClose}
        />
      </SideSheet>

      <SideSheet
        width={"30%"}
        visible={branchVisible}
        onCancel={handleBranchClose}
        title={branchEdit?.name ? branchEdit?.name : "add children branch"}
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

export default WorkFlow;

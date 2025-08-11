import { Button, Dropdown, List, Toast, Tooltip, Tree, Typography } from "@douyinfe/semi-ui";
import { IconMore, IconPlus } from "@douyinfe/semi-icons";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { ReactNode } from "react";
import "./index.scss";
import { NodeTreeNodeData } from "../../../Node/entity";
import request from "../../../../request";
import React from "react";
import store from "../store/store";
import { initDagGraphInfo } from "../store/actionType";
import { apiGetDagInfo } from "../api";

const { Text, Paragraph } = Typography;

const NodeTree: React.FC = () => {
  //const { treeData, getNodeTree } = useGetNoteTree();

  const { state, dispatch } = React.useContext(store);

  const { treeData } = state

  //const nodeList = useGetNodeList();

  const renderBtn = (content: string) => {
    return (
      <Dropdown
        // position="rightTop"
        render={
          <Dropdown.Menu>
            <Dropdown.Item>Menu Item 1</Dropdown.Item>
            <Dropdown.Item>Menu Item 2</Dropdown.Item>
            <Dropdown.Item>Menu Item 3</Dropdown.Item>
          </Dropdown.Menu>
        }
      >
        <Button
          onClick={(e) => {
            //Toast.info({ content });
            e.stopPropagation();
          }}
          icon={<IconMore />}
          size="small"
        />
      </Dropdown>
    );
  };


  async function getDagInfo() {
    var response = await apiGetDagInfo()
    if (response.flag != 'success') {
      return
    }

    dispatch(initDagGraphInfo(response.result))

  }

  const renderLabel = (label: ReactNode, item: NodeTreeNodeData) => {


    if(item.nodeEntity.type == 'leaf'){

      var i = 0;
    }

    const desc = item.nodeEntity.type == 'branch' ? item.nodeEntity.desc : item.nodeEntity.nodeEntity?.desc_ || '';

    return <Tooltip content={ desc} position="right">
      <div
        style={{ display: "flex", height: "24px" }}
        draggable
        ///@ts-ignore
        onDragStart={(dragEvent) => onDragStart(item.nodeEntity, dragEvent)}
      >
        <Typography.Text
          ellipsis={{ showTooltip: true }}
          style={{ width: "calc(100% - 48px)" }}
          className="label"
        >

          {label}

        </Typography.Text>
        {/* {renderBtn(item?.key!)} */}
      </div>
    </Tooltip>
  };

  function onUploadNode(): void {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".py,.so";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const formData = new FormData();
      formData.append("file", file);
      try {
        const response = await request.upload("/api/nodes/upload", formData, {});
        if (response.flag === "success") {
          getDagInfo();
          Toast.success("upload node success")
        } else {
          // Handle error
          //console.error(response.msg);
          Toast.error("upload node failed")
        }
      } catch (error) {
        console.error("Upload node failed:", error);
      }
    };
    input.click();
  }



  function onDragStart(
    node: NodeTreeNodeData,
    dragEvent: React.DragEvent<HTMLDivElement>
  ) {

    if (node.type != 'leaf') {
      return
    }
    const dragImage = document.getElementById("drag-image");

    ///@ts-ignore
    dragEvent.dataTransfer.setDragImage(dragImage, 50, 50);

    // @ts-ignore
    dragEvent.dataTransfer.setData("text/plain", JSON.stringify(node.nodeEntity));
  }

  return (
    <>
      <div className="tree-node">

        <div className="tree-node-header">
          <Text>Node</Text>
          <Text
            link
            icon={<IconPlus />}
            onClick={() => onUploadNode()}
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
          ///@ts-ignore todo fix
          renderLabel={renderLabel}
          className="tree-node"
        //draggable
        />
        {/* <List
        className="node-list"
        header={<div>nodes</div>}
        // footer={<div>Footer</div>}
        bordered
        dataSource={nodeList}
        renderItem={(item) => {
          return (
            <Tooltip content={item.desc_} position="right">
              <List.Item>
                <div
                  onDragStart={(dragEvent) => onDragStart(item!, dragEvent)}
                  draggable
                >

                  <span className="node-name">{item.name_}</span>
                </div>
              </List.Item>
            </Tooltip>
          );
        }}
      /> */}
      </div>
      <div id="drag-image" className="drag-image">
        <div
          style={{
            width: "100px",
            height: "100px",
            backgroundColor: "rgba(0, 0, 255, 0.5)",
          }}
        >
          <p style={{ color: "white;", lineHeight: "100px" }}>Dragging...</p>
        </div>
      </div>
    </>
  );
};

export default NodeTree;


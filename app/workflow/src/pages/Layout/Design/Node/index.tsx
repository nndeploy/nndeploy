import { Button, Dropdown, List, Tree, Typography } from "@douyinfe/semi-ui";
import { useGetNodeList, useGetNoteTree } from "./effect";
import { IconMore } from "@douyinfe/semi-icons";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { ReactNode } from "react";
import "./index.scss";
import { INodeEntity } from "../../../Node/entity";

const NodeTree: React.FC = () => {
  // const {treeData} = useGetNoteTree();
  const nodeList = useGetNodeList();

  // const renderBtn = (content: string) => {
  //   return (
  //     <Dropdown
  //       // position="rightTop"
  //       render={
  //         <Dropdown.Menu>
  //           <Dropdown.Item>Menu Item 1</Dropdown.Item>
  //           <Dropdown.Item>Menu Item 2</Dropdown.Item>
  //           <Dropdown.Item>Menu Item 3</Dropdown.Item>
  //         </Dropdown.Menu>
  //       }
  //     >
  //       <Button
  //         onClick={(e) => {
  //           //Toast.info({ content });
  //           e.stopPropagation();
  //         }}
  //         icon={<IconMore />}
  //         size="small"
  //       />
  //     </Dropdown>
  //   );
  // };

  // const renderLabel = (label: ReactNode, item?: TreeNodeData) => (
  //   <div
  //     style={{ display: "flex", height: "24px" }}
  //     draggable
  //     ///@ts-ignore
  //     onDragStart={(dragEvent) => onDragStart(item!, dragEvent)}
  //   >
  //     <Typography.Text
  //       ellipsis={{ showTooltip: true }}
  //       style={{ width: "calc(100% - 48px)" }}
  //       className="label"
  //     >
  //       {label}
  //     </Typography.Text>
  //     {renderBtn(item?.key!)}
  //   </div>
  // );
  function onDragStart(
    node: INodeEntity,
    dragEvent: React.DragEvent<HTMLDivElement>
  ) {
    const dragImage = document.getElementById("drag-image");

    ///@ts-ignore
    dragEvent.dataTransfer.setDragImage(dragImage, 50, 50);

    // @ts-ignore
    dragEvent.dataTransfer.setData("text/plain", JSON.stringify(node));
  }

  return (
    <>
      {/* <Tree
        treeData={treeData}
        renderLabel={renderLabel}
        className="tree-node"
        //draggable
      /> */}
      <List
        className="node-list"
        header={<div>nodes</div>}
        // footer={<div>Footer</div>}
        bordered
        dataSource={nodeList}
        renderItem={(item) => {
          return (
            <List.Item>
              <div
                onDragStart={(dragEvent) => onDragStart(item!, dragEvent)}
                draggable
              >
                {item.name_}
              </div>
            </List.Item>
          );
        }}
      />
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

import { Button, Dropdown, Tree, Typography } from "@douyinfe/semi-ui";
import { useGetTree } from "./effect";
import { IconMore } from "@douyinfe/semi-icons";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { ReactNode } from "react";
import "./index.scss";

const NodeTree: React.FC = () => {
  const treeData = useGetTree();

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

  const renderLabel = (label: ReactNode, item?: TreeNodeData) => (
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
      {renderBtn(item?.key!)}
    </div>
  );
  function onDragStart(node: TreeNodeData, dragEvent: DragEvent) {
    const dragImage = document.getElementById("drag-image");

    ///@ts-ignore
    dragEvent.dataTransfer.setDragImage(dragImage, 50, 50);

    // @ts-ignore
    dragEvent.dataTransfer.setData("text/plain", node.type);
  }

  return (
    <>
      <Tree
        treeData={treeData}
        renderLabel={renderLabel}
        className="tree-node"
        //draggable
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

import { useEffect, useState } from "react";
import { INodeTreeNodeEntity, NodeTreeNodeData } from "./entity";
import { apiGetNodeTree } from "./api";

export function useGetNodeTree(): {
  flatData: INodeTreeNodeEntity[];
  setFlatData: React.Dispatch<React.SetStateAction<INodeTreeNodeEntity[]>>;

  treeData: NodeTreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<NodeTreeNodeData[]>>;
  getNodeTree: () => Promise<void>;
} {
  const [flatData, setFlatData] = useState<INodeTreeNodeEntity[]>([]);
  const [treeData, setTreeData] = useState<NodeTreeNodeData[]>([]);

  function buildTreeFromArray(
    data: INodeTreeNodeEntity[],
    parentId: string = ""
  ): NodeTreeNodeData[] {
    return data
      .filter((item) => {
        if (item.parentId == parentId) {
          return true;
        }
        return false;
      })
      .map((item) => {
        const children = buildTreeFromArray(data, item.id);

        return {
          key: item.id,
          label: item.name,

          entity: item,
          children: children.length > 0 ? children : undefined,
        };
      });
  }

  async function getNodeTree() {
    const res = await apiGetNodeTree();
    setFlatData(res.result);
  }

  useEffect(() => {
    getNodeTree()
  }, []);

  useEffect(() => {
    const tree = buildTreeFromArray(flatData);
    setTreeData(tree);
  }, [flatData]);

  return { flatData, setFlatData, treeData, setTreeData, getNodeTree };
}
import { useEffect, useState } from "react";
import { INodeBranchEntity, NodeBranchTreeNodeData } from "./entity";
import { apiGetNodeBranch } from "./api";

export function useGetNoteBranch(): {
  flatData: INodeBranchEntity[];
  setFlatData: React.Dispatch<React.SetStateAction<INodeBranchEntity[]>>;

  treeData: NodeBranchTreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<NodeBranchTreeNodeData[]>>;
  getNodeBranchTree: () => Promise<void>;
} {
  const [flatData, setFlatData] = useState<INodeBranchEntity[]>([]);

  const [treeData, setTreeData] = useState<NodeBranchTreeNodeData[]>([]);

  async function getNodeBranchTree() {
    const res = await apiGetNodeBranch();
    setFlatData(res.result);
  }

  function buildTreeFromArray(
    data: INodeBranchEntity[],
    parentId: string = ""
  ): NodeBranchTreeNodeData[] {
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

  useEffect(() => {
    apiGetNodeBranch().then((res) => {
      setFlatData(res.result);
    });
  }, []);

  useEffect(() => {
    const tree = buildTreeFromArray(flatData);
    setTreeData(tree);
  }, [flatData]);

  return { flatData, setFlatData, treeData, setTreeData, getNodeBranchTree };
}

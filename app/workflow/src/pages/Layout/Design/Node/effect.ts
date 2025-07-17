import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { useEffect, useState } from "react";
import {
  INodeBranchEntity,
  INodeEntity,
  INodeTreeNodeEntity,
  NodeTreeNodeData,
} from "../../../Node/entity";
import { apiGetNodeList, apiGetNodeTree } from "./api";

// export function useGetTree(){
//   const [treeData, setTreeData] = useState<TreeNodeData[]>([])

//   useEffect(() => {
//     apiGetNodeTree().then((res) => {
//       setTreeData(res.result)
//     })
//   }, [])

//   return treeData

// }

export function useGetNoteTree(): {
  flatData: INodeTreeNodeEntity[];
  setFlatData: React.Dispatch<React.SetStateAction<INodeTreeNodeEntity[]>>;

  treeData: NodeTreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<NodeTreeNodeData[]>>;
  getNodeTree: () => Promise<void>;
} {
  const [flatData, setFlatData] = useState<INodeTreeNodeEntity[]>([]);

  const [treeData, setTreeData] = useState<NodeTreeNodeData[]>([]);

  async function getNodeTree() {
    const res = await apiGetNodeTree();
    setFlatData(res.result);
  }

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

          nodeEntity: item,
          children: children.length > 0 ? children : undefined,
        };
      });
  }

  useEffect(() => {
    apiGetNodeTree().then((res) => {
      setFlatData(res.result);
    });
  }, []);

  useEffect(() => {
    const tree = buildTreeFromArray(flatData);
    setTreeData(tree);
  }, [flatData]);

  return { flatData, setFlatData, treeData, setTreeData, getNodeTree };
}

export function useGetNodeList() {
  const [nodeList, setNodeList] = useState<INodeEntity[]>([]);

  async function getNodeList() {
    const res = await apiGetNodeList();
    setNodeList(res.result);
  }

  useEffect(()=>{
    getNodeList()
  }, [])

  return nodeList;
}

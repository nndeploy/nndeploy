import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { useEffect, useState } from "react";
import { apiGetResourceTree } from "./api";

export function useGetTree(): [TreeNodeData[], React.Dispatch<React.SetStateAction<TreeNodeData[]>>]{
  const [treeData, setTreeData] = useState<TreeNodeData[]>([])

  useEffect(() => {
    apiGetResourceTree().then((res) => {
      setTreeData(res.result)
    })
  }, [])

  return [treeData, setTreeData]

}
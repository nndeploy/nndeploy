import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";
import { useEffect, useState } from "react";
import { apiGetNodeTree } from "./api";

export function useGetTree(){
  const [treeData, setTreeData] = useState<TreeNodeData[]>([])

  useEffect(() => {
    apiGetNodeTree().then((res) => {
      setTreeData(res.result)
    })
  }, [])

  return treeData

}
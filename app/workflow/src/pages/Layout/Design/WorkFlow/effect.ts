import { useEffect, useState } from "react";
import { apiGetWorkFlowBranch, apiGetWorkFlowTree } from "./api";
import { IWorkFlowTreeNodeEntity, WorkFlowTreeNodeData } from "./entity";

export function useGetWorkflowTree(): {
  flatData: IWorkFlowTreeNodeEntity[];
  setFlatData: React.Dispatch<React.SetStateAction<IWorkFlowTreeNodeEntity[]>>;

  treeData: WorkFlowTreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<WorkFlowTreeNodeData[]>>;
  getWorkFlowTree: () => Promise<void>;
} {
  const [flatData, setFlatData] = useState<IWorkFlowTreeNodeEntity[]>([]);
  const [treeData, setTreeData] = useState<WorkFlowTreeNodeData[]>([]);

  function buildTreeFromArray(
    data: IWorkFlowTreeNodeEntity[],
    parentId: string = ""
  ): WorkFlowTreeNodeData[] {
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

  async function getWorkFlowTree() {
    const res = await apiGetWorkFlowTree();
    setFlatData(res.result);
  }

  useEffect(() => {
    apiGetWorkFlowTree().then((res) => {
      setFlatData(res.result);
    });
  }, []);

  useEffect(() => {
    const tree = buildTreeFromArray(flatData);
    setTreeData(tree);
  }, [flatData]);

  return { flatData, setFlatData, treeData, setTreeData, getWorkFlowTree };
}

export function useGetWorkflowBranch(): {
  treeData: WorkFlowTreeNodeData[];
} {
  const [treeData, setTreeData] = useState<WorkFlowTreeNodeData[]>([]);

  function buildTreeFromArray(
    data: IWorkFlowTreeNodeEntity[],
    parentId: string = ""
  ): WorkFlowTreeNodeData[] {
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
    apiGetWorkFlowBranch().then((res) => {
      const tree = buildTreeFromArray(res.result);
      setTreeData(tree);
    });
  }, []);

  return { treeData };
}

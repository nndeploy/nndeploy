import { useEffect, useState } from "react";
import { apiGetResourceTree } from "./api";
import { IResourceTreeNodeEntity, ResourceTreeNodeData } from "./entity";

export function useGetTree(): {
  flatData: IResourceTreeNodeEntity[];
  setFlatData: React.Dispatch<React.SetStateAction<IResourceTreeNodeEntity[]>>;

  treeData: ResourceTreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<ResourceTreeNodeData[]>>;
} {
  const [flatData, setFlatData] = useState<IResourceTreeNodeEntity[]>([]);
  const [treeData, setTreeData] = useState<ResourceTreeNodeData[]>([]);

  function buildTreeFromArray(
    data: IResourceTreeNodeEntity[],
    parentId: string = ""
  ): ResourceTreeNodeData[] {
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
          // isLeaf: item.type,

          entity: item,
          children: children.length > 0 ? children : undefined,
        };
      });
  }

  useEffect(() => {
    apiGetResourceTree().then((res) => {
      setFlatData(res.result);
    });
  }, []);

  useEffect(() => {
    const tree = buildTreeFromArray(flatData);
    setTreeData(tree);
  }, [flatData]);

  return { flatData, setFlatData, treeData, setTreeData };
}

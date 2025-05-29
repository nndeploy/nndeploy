import { useEffect, useState } from "react";
import { apiGetResourceTree } from "./api";
import { IResourceEntity, ResourceTreeNodeData } from "./entity";

type NodeProcessFunc = (node: IResourceEntity) => ResourceTreeNodeData;

export function nodeProcess(node: IResourceEntity): ResourceTreeNodeData {
  return {
    key: node.id,
    label: node.name,
    isLeaf: node.isLeaf,
    children: node.children ? node.children.map(nodeProcess) : undefined,
    entity: node,
  };
}

export function processTree(
  tree: IResourceEntity[],
  nodeProcess: NodeProcessFunc
): ResourceTreeNodeData[] {
  return tree.map((item) => {
    return nodeProcess(item);
  });
}


export function useGetTree(): {
  flatData: IResourceEntity[];
  setFlatData: React.Dispatch<React.SetStateAction<IResourceEntity[]>>;

  treeData: ResourceTreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<ResourceTreeNodeData[]>>;
} {
  const [flatData, setFlatData] = useState<IResourceEntity[]>([]);
  const [treeData, setTreeData] = useState<ResourceTreeNodeData[]>([]);

  function buildTreeFromArray(
    data: IResourceEntity[],
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
          isLeaf: item.isLeaf,

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

  return {flatData, setFlatData, treeData, setTreeData};
}

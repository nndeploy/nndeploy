import { useEffect, useState } from "react";
import { apiGetResourceTree } from "./api";
import { IResourceTreeNodeEntity, IServerResourceFileEntity, IServerResourceResonseData, ResourceTreeNodeData } from "./entity";

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

  function transforServerDataToFlatData(serverResourceResonseData: IServerResourceResonseData) {

    let flatItems: IResourceTreeNodeEntity[] = [
      { id: 'images', "name": 'images', parentId: "", type: 'branch' },
      { id: 'videos', "name": 'videos', parentId: "", type: 'branch' },
      { id: 'models', "name": 'models', parentId: "", type: 'branch' }
    ]
    for (let fileType in serverResourceResonseData) {

      const files: IServerResourceFileEntity[] = serverResourceResonseData[fileType]
      for (let i = 0; i < files.length; i++) {
        let file = files[i]
        const flatItem: IResourceTreeNodeEntity = { 
          id: file.filename, name: file.filename, parentId: fileType, type: 'leaf', 
          entity: file
        
        }
        flatItems = [...flatItems, flatItem]
      }
    }
    return flatItems
  }

  useEffect(() => {
    apiGetResourceTree().then((res) => {

      let flatData = transforServerDataToFlatData(res.result)
      setFlatData(flatData);
    });
  }, []);

  useEffect(() => {
    const tree = buildTreeFromArray(flatData);
    setTreeData(tree);
  }, [flatData]);

  return { flatData, setFlatData, treeData, setTreeData };
}

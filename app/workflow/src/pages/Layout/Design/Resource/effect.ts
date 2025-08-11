import { useEffect, useState } from "react";
import { apiGetResourceTree } from "./api";
import { IResourceTreeNodeEntity, IServerResourceFileEntity, IServerResourceResonseData, ResourceTreeNodeData } from "./entity";

export function useGetTree(): {
  flatData: IResourceTreeNodeEntity[];
  setFlatData: React.Dispatch<React.SetStateAction<IResourceTreeNodeEntity[]>>;

  treeData: ResourceTreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<ResourceTreeNodeData[]>>;
  getResourceTree: () => void;

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

  // function transforServerDataToFlatData(serverResourceResonseData: IServerResourceResonseData) {

  //   let flatItems: IResourceTreeNodeEntity[] = [
  //     { id: 'images', "name": 'images', parentId: "", type: 'branch' },
  //     { id: 'videos', "name": 'videos', parentId: "", type: 'branch' },
  //     { id: 'models', "name": 'models', parentId: "", type: 'branch' }
  //   ]
  //   for (let fileType in serverResourceResonseData) {

  //     const files: IServerResourceFileEntity[] = serverResourceResonseData[fileType]
  //     for (let i = 0; i < files.length; i++) {
  //       let file = files[i]
  //       const flatItem: IResourceTreeNodeEntity = { 
  //         id: file.filename, name: file.filename, parentId: fileType, type: 'leaf', 
  //         file_info: file
        
  //       }
  //       flatItems = [...flatItems, flatItem]
  //     }
  //   }
  //   return flatItems
  // }

  async function getResourceTree(){
      const response =  await apiGetResourceTree()

      //let flatData = transforServerDataToFlatData(response.result)

      function getNodeById(id:string){
        return response.result.find(item=>item.id == id)!
      }

      if(response.flag != "success"){
        return 
      }

      response.result = response.result.map(item=>{
        return {
          ...item,
          parent_info: getNodeById(item.parentId)
        }
      })
      setFlatData(response.result);
    ;
  }

  useEffect(() => {
    getResourceTree()
  }, []);

  useEffect(() => {
    const tree = buildTreeFromArray(flatData);
    setTreeData(tree);
  }, [flatData]);

  return { flatData, setFlatData, treeData, setTreeData, getResourceTree };
}

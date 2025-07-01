import { useEffect, useState } from "react";
import { apiGetWorkFlowBranch, apiGetWorkFlowTree } from "./api";
import { IBusinessNode, IWorkFlowTreeNodeEntity, WorkFlowTreeNodeData } from "./entity";
import { TreeNodeData } from "@douyinfe/semi-ui/lib/es/tree";

export function useGetWorkflowTree(): {
 // flatData: IBusinessNode[];
  //setFlatData: React.Dispatch<React.SetStateAction<IBusinessNode[]>>;

  treeData: TreeNodeData[];
  setTreeData: React.Dispatch<React.SetStateAction<TreeNodeData[]>>;
  getWorkFlowTree: () => Promise<void>;
} {
 // const [flatData, setFlatData] = useState<IBusinessNode[]>([]);
  const [treeData, setTreeData] = useState<TreeNodeData[]>([]);

  // function buildTreeFromArray(
  //   data: IWorkFlowTreeNodeEntity[],
  //   parentId: string = ""
  // ): WorkFlowTreeNodeData[] {
  //   return data
  //     .filter((item) => {
  //       if (item.parentId == parentId) {
  //         return true;
  //       }
  //       return false;
  //     })
  //     .map((item) => {
  //       const children = buildTreeFromArray(data, item.id);

  //       return {
  //         key: item.id,
  //         label: item.name,

  //         entity: item,
  //         children: children.length > 0 ? children : undefined,
  //       };
  //     });
  // }

  async function getWorkFlowTree() {
    const res = await apiGetWorkFlowTree();
    const treeData = res.result.map(item=>{
      return {
        key: item.name_, 
        label: item.name_, 
        children: undefined
      }
    })

    //setFlatData(res.result);

    setTreeData(treeData)


  }

  useEffect(() => {
    getWorkFlowTree()
  }, []);

  // useEffect(() => {
  //   const tree = buildTreeFromArray(flatData);
  //   setTreeData(tree);
  // }, [flatData]);

  return { treeData, setTreeData,  getWorkFlowTree };
}

// export function useGetWorkflowBranch(): {
//   treeData: WorkFlowTreeNodeData[];
// } {
//   const [treeData, setTreeData] = useState<WorkFlowTreeNodeData[]>([]);

//   function buildTreeFromArray(
//     data: IBusinessNode[],
//     parentId: string = ""
//   ): WorkFlowTreeNodeData[] {
//     return data
//       .filter((item) => {
//         if (item.parentId == parentId) {
//           return true;
//         }
//         return false;
//       })
//       .map((item) => {
//         const children = buildTreeFromArray(data, item.id);

//         return {
//           key: item.id,
//           label: item.name,

//           entity: item,
//           children: children.length > 0 ? children : undefined,
//         };
//       });
//   }

//   useEffect(() => {
//     apiGetWorkFlowBranch().then((res) => {
//       const tree = buildTreeFromArray(res.result);
//       setTreeData(tree);
//     });
//   }, []);

//   return { treeData };
// }

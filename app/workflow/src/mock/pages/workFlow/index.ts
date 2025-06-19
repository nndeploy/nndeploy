import Mock, { templateOrFn } from "mockjs";

import { initialData } from "../../../pages/components/flow/initial-data";
import { IBusinessNode, IWorkFlowBranchEntity, IWorkFlowEntity, IWorkFlowTreeNodeEntity } from "../../../pages/Layout/Design/WorkFlow/entity";
import { businessContents, workFlows } from "./initalWorkFlow";
// mock方法,详细的可以看官方文档
const Random = Mock.Random;

interface MockItem {
  url: string | RegExp;
  type: "get" | "post";
  response: templateOrFn;
}

const branches: IWorkFlowBranchEntity[] = [
  {
    id: "image recognition",
    name: "image recognition",
    parentId: "",
  },
  {
    id: "dog recognition",
    name: "dog recognition",
    parentId: "image recognition",
  },
  {
    id: "Golden Retriever image",
    name: "Golden Retriever",
    parentId: "dog recognition",
  },
  {
    id: "German image shepherd",
    name: "German shepherd",
    parentId: "dog recognition",
  },

  {
    id: "voice recognition",
    name: "pattern recognition",
    parentId: "",
  },
  {
    id: "dog voice recognition",
    name: "dog recognition",
    parentId: "voice recognition",
  },
  {
    id: "Golden voice Retriever",
    name: "Golden Retriever",
    parentId: "dog voice recognition",
  },
  {
    id: "German voice shepherd",
    name: "German shepherd",
    parentId: "dog voice recognition",
  },
];

//const flows: IWorkFlowEntity[] = workFlows

export const workFlowHandler: MockItem[] = [
  {
    url: "/workflow/branch/save",
    type: "post",
    response: (options) => {
      const entity: IWorkFlowBranchEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {
          ...entity,
          id: entity.id ? entity.id : Random.guid(),
        },
      };
    },
  },
  {
    url: "/workflow/branch/delete",
    type: "post",
    response: (options) => {
      const entity: IWorkFlowBranchEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {},
      };
    },
  },
  {
    url: "/workflow/tree",
    type: "get",
    response: (options) => {
      const data: IWorkFlowTreeNodeEntity[] = [
        ...branches.map((item) => ({ ...item, type: "branch" as const })),
        ...businessContents.map((item) => ({ id: item.name_, name: item.name_, type: "leaf" as const , parentId: ""})),
      ];

      return {
        flag: "success",
        message: "成功",
        result: data,
      };
    },
  },

  {
    url: "/workflow/branch",
    type: "get",
    response: (options) => {
      const branches: IWorkFlowBranchEntity[] = [
        {
          id: "image recognition",
          name: "image recognition",
          parentId: "",
        },
        {
          id: "dog recognition",
          name: "dog recognition",
          parentId: "image recognition",
        },
        {
          id: "Golden Retriever image",
          name: "Golden Retriever",
          parentId: "dog recognition",
        },
        {
          id: "German image shepherd",
          name: "German shepherd",
          parentId: "dog recognition",
        },

        {
          id: "voice recognition",
          name: "pattern recognition",
          parentId: "",
        },
        {
          id: "dog voice recognition",
          name: "dog recognition",
          parentId: "voice recognition",
        },
        {
          id: "Golden voice Retriever",
          name: "Golden Retriever",
          parentId: "dog voice recognition",
        },
        {
          id: "German voice shepherd",
          name: "German shepherd",
          parentId: "dog voice recognition",
        },
      ];

      const data: IWorkFlowTreeNodeEntity[] = [
        ...branches.map((item) => ({ ...item, type: "branch" as const })),
      ];

      return {
        flag: "success",
        message: "成功",
        result: data,
      };
    },
  },

  {
    url: "/workflow/save",
    type: "post",
    response: (options) => {
      // const entity: IWorkFlowEntity = JSON.parse(options.body);

      // entity.designContent = entity.designContent? entity.designContent : { nodes: [], edges: [] }
      // const findIndex = flows.findIndex(item=>item.id == entity.id)

      // if(findIndex == -1){
      //    entity.id = Random.guid()
      //   flows.push(entity)
        
      // }else{
       
      //   flows[findIndex] = entity
      // }

      const businessContent: IBusinessNode = JSON.parse(options.body);
      const findIndex = businessContents.findIndex(item=>item.name = businessContent.name_)
      if(findIndex > -1){
        businessContents[findIndex] = businessContent
      }else{
        businessContents.push(businessContent)
      }
     

      return {
        flag: "success",
        message: "",
        result: businessContent
      };
    },
  },

  {
    url: "/workflow/get",
    type: "post",
    response: (options) => {
      const entity = JSON.parse(options.body);
  
      var find = businessContents.find(item=>item.name_ == entity.flowName)

      return {
        flag: "success",
        message: "",
        result: find
      };
    },
  },

  {
    url: "/workflow/delete",
    type: "post",
    response: (options) => {
      const entity: IWorkFlowEntity = JSON.parse(options.body);

       const findIndex = businessContents.findIndex((item) => item.name_ == entity.name);

      if (findIndex == -1) {
        return {
          flag: "error",
          message: "could not find this item",
          result: {},
        };
      } else {
        businessContents.splice(findIndex, 1);
      }
      return {
        flag: "success",
        message: "",
        result: {},
      };

    },
  },
];

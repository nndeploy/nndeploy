import Mock, { templateOrFn } from "mockjs";
import {
  IWorkFlowBranchEntity,
  IWorkFlowEntity,
  IWorkFlowTreeNodeEntity,
} from "../../../pages/Layout/Backend/WorkFlow/entity";
import { initialData } from "../../../pages/components/flow/initial-data";
// mock方法,详细的可以看官方文档
const Random = Mock.Random;

interface MockItem {
  url: string | RegExp;
  type: "get" | "post";
  response: templateOrFn;
}

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

      const flows: IWorkFlowEntity[] = [
        {
          id: "Golden  Retriever image 1",
          name: "image 1",
          parentId: "Golden Retriever image",
          content: initialData,
        },
        {
          id: "Golden  Retriever image 2",
          name: "image 2",
          parentId: "Golden Retriever image",
          content: initialData,
        },
      ];

      const data: IWorkFlowTreeNodeEntity[] = [
        ...branches.map((item) => ({ ...item, type: "branch" as const })),
        ...flows.map((item) => ({ ...item, type: "leaf" as const })),
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
      const entity: IWorkFlowEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {
          ...entity,
          id: entity.id ? entity.id : Random.guid(),
          content: entity.content ? entity.content : { nodes: [], edges: [] },
        },
      };
    },
  },

  {
    url: "/workflow/get",
    type: "post",
    response: (options) => {
      const entity: IWorkFlowEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {
          id: entity.id ? entity.id : Random.guid(),
          content: initialData,
        },
      };
    },
  },

  {
    url: "/workflow/delete",
    type: "post",
    response: (options) => {
      const resource: IWorkFlowEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {},
      };
    },
  },
];

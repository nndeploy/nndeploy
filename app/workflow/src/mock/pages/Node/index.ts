import Mock from "mockjs";
import {
  INodeBranchEntity,
  INodeEntity,
  INodeTreeNodeEntity,
} from "../../../pages/Node/entity";
import { MockItem } from "../../entity";

// mock方法,详细的可以看官方文档
const Random = Mock.Random;

const nodeBranches: INodeBranchEntity[] = [
  {
    id: "1",
    name: "begin",
    parentId: "",
  },
  {
    id: "2",
    name: "llm",
    parentId: "",
  },
  {
    id: "3",
    name: "branch",
    parentId: "",
  },
  {
    id: "3-1",
    name: "loop",
    parentId: "3",
  },
  {
    id: "3-2",
    name: "condition",
    parentId: "3",
  },
  {
    id: "4",
    name: "end",
    parentId: "",
  },
];

const nodes: INodeEntity[] = [];

export const nodeHandler: MockItem[] = [
  {
    url: "/node/branch",
    type: "post",
    response: (request) => {
      var params = JSON.parse(request.body);

      return {
        flag: "success",
        message: "成功",
        result: nodeBranches,
      };
    },
  },
  {
    url: "/node/tree",
    type: "get",
    response: (options) => {
      const data: INodeTreeNodeEntity[] = [
        ...nodeBranches.map((item) => ({ ...item, type: "branch" as const })),
        ...nodes.map((item) => ({ ...item, type: "leaf" as const })),
      ];

      return {
        flag: "success",
        message: "成功",
        result: data,
      };
    },
  },
  {
    url: "/node/branch/save",
    type: "post",
    response: (request: any) => {
      var entity = JSON.parse(request.body);

      const findIndex = nodeBranches.findIndex((item) => item.id == entity.id);

      if (findIndex == -1) {
        entity.id = Random.guid();
        nodeBranches.push(entity);
      } else {
        nodeBranches[findIndex] = entity;
      }

      return {
        flag: "success",
        message: "成功",
        result: entity,
      };
    },
  },
  {
    url: "/node/branch/delete",
    type: "post",
    response: (options) => {
      const entity: INodeBranchEntity = JSON.parse(options.body);

      const findIndex = nodeBranches.findIndex((item) => item.id == entity.id);

      if (findIndex == -1) {
        return {
          flag: "error",
          message: "could not find this item",
          result: {},
        };
      } else {
        nodeBranches.splice(findIndex, 1);
      }

      return {
        flag: "success",
        message: "",
        result: {},
      };
    },
  },

  {
    url: "/node/save",
    type: "post",
    response: (request: any) => {
      var entity = JSON.parse(request.body);

      const findIndex = nodes.findIndex((item) => item.id == entity.id);

      if (findIndex == -1) {
        entity.id = Random.guid();
        nodes.push(entity);
      } else {
        nodes[findIndex] = entity;
      }

      return {
        flag: "success",
        message: "成功",
        result: entity,
      };
    },
  },
  {
    url: "/node/delete",
    type: "post",
    response: (options) => {
      const entity: INodeEntity = JSON.parse(options.body);

      const findIndex = nodes.findIndex((item) => item.id == entity.id);

      if (findIndex == -1) {
        return {
          flag: "error",
          message: "could not find this item",
          result: {},
        };
      } else {
        nodes.splice(findIndex, 1);
      }

      return {
        flag: "success",
        message: "",
        result: {},
      };
    },
  },

  {
    url: "/node/page",
    type: "post",
    response: (options) => {
      const {
        parentId,
        currentPage,
        pageSize,
      }: { parentId: string; currentPage: number; pageSize: number } =
        JSON.parse(options.body);

      let finds: INodeEntity[] = [];
      if (parentId) {
        finds = nodes.filter((item) => item.parentId == parentId);
      } else {
        finds = nodes;
      }

      const offset = (currentPage - 1) * pageSize;

      const records = finds.slice(offset, pageSize);

      return {
        flag: "success",
        message: "",
        result: {
          records,
          total: finds.length,
        },
      };
    },
  },
];

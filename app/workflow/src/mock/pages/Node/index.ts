import Mock from "mockjs";
import { INodeBranchEntity } from "../../../pages/Node/entity";
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

export const nodeHandler: MockItem[] =[
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
];

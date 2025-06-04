import { templateOrFn } from "mockjs";

export interface MockItem {
  url: string | RegExp;
  type: "get" | "post";
  response: templateOrFn;
}
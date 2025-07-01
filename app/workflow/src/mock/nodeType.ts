import Mock from "mockjs";
import { initialData } from "../pages/components/flow/initial-data";

// mock方法,详细的可以看官方文档
const Random = Mock.Random;

export default [
  {
    url: "/nodeType/get",
    type: "post",
    response: (request:any) => {
      console.log('/nodeType/get')
       var params = JSON.parse(request.body)
      // const id = params
      //const id = "start"

      const nodeType = initialData.nodes.find((item) => {
        if (item.type === params.id) {
          return item
        }
      })
      return {
        flag: "success",
        message: "成功", 
        result: nodeType
      }
    }
  }
]

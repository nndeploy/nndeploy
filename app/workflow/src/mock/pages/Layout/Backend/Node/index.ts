import Mock from "mockjs";

// mock方法,详细的可以看官方文档
const Random = Mock.Random;

export default [
  {
    url: "/node/tree",
    type: "post",
    response: (request:any) => {
      var params = JSON.parse(request.body)
      console.log("/node/tree.............")
      return {
        flag: "success",
        message: "成功", 
        result: [
          {
            key: "1",
            label: "Node 1",
            children: [
              {
                key: "1-1",
                label: "Node 1-1",
                children: [
                  { key: "1-1-1", label: "Node 1-1-1" , type: 'start'},
                  { key: "1-1-2", label: "Node 1-1-2", type: 'start' },
                ],
              },
              {
                key: "1-2",
                label: "Node 1-2",
                children: [
                  { key: "1-2-1", label: "Node 1-2-1" , type: 'condition'},
                  { key: "1-2-2", label: "Node 1-2-2", type: 'loop' },
                ],
              },
            ],
          },
          {
            key: "2",
            label: "Node 2",
            children: [
              {
                key: "2-1",
                label: "Node 2-1",
                children: [
                  { key: "2-1-1", label: "Node 2-1-1" , type: 'llm'},
                  { key: "2-1-2", label: "Node 2-1-2" , type: 'end'},
                ],
              },
              {
                key: "2-2",
                label: "Node 2-2",
                children: [
                  { key: "2-2-1", label: "Node 2-2-1", type: 'comment' },
                  { key: "2-2-2", label: "Node 2-2-2" , type: 'group'},
                ],
              },
            ],
          },
        ],
      };
    },
  },
];

# nndeploy前端架构与实现文档

## 项目概览

nndeploy的前端基于 *react* 实现，提供一个基于图的工作流引擎, 包括节点的添加、删除、连接、属性设置等操作, 以及工作流的保存、加载、执行等操作.

在工作流方面我们选用字节的开源flowgram.ai, 虽然略显庞大, 但结构清晰, 提供丰富的拓展能力,  可进行定制化的开发. 基于rush的monorepo管理, 很好的处理了项目之间的依赖, 提升了开发效率.


### 模块划分

前端各个模块的功能如下：

```
workflow/src/
  ├── app.ts               # 入口文件, 路由配置
  ├── assets               # 静态资源文件
  ├──components            # flowgram.ai自带的组件库
  ├── context              # flowgram.ai自带的context, 及项目自定义的 流程context, 
  ├── hooks             
  |     |── useEditorProps.ts  # flowgram.ai自带的hooks
        |── use-is-sidebar.ts        # flowgram.ai自带的hooks, 用于判断当前是画布中的节点渲染, 还是侧边栏中的组件渲染
          
  ├── mock              # 开发时的测试数据
  ├── nodes             # flowgram.ai自带的节点库, 项目中未使用
  ├── pages             # 前端页面, 建议一个路由对应一个子目录
  |     |── Layout
  |     |      |── Design  # 设计器的layout页面, 包含顶部, 侧边导航栏, 画布
  |     |      |      |── header  # 顶部航栏
  |     |      |      |── sidebar # 侧边导航栏, 包含静态媒体资源(图片, 视频), 节点列表, 建好的工作流列表, 模板列表
  |     |      |      |── store,  # 全局状态管理
  |     |      |── Backend # 预留作为后端操作的layout  
  |     |── Home  # 系统首页, 包含用户的工作流及默认的工作流模板列表 
  |     |── components  
  |     |      |── flow       # 工作流编辑器 
  |     |      |── CodeBlock  # markdown友好显示组件
  |     |      |── json-schema-editor # json-schema编辑器, 用于编辑节点的属性
  |     |── NoMatch  # 404页面, 用于匹配不存在的路由
  |
  ├── request                # 发送ajax请求的函数
```

## 关键实现细节

### 日志与监控

* 画布底部有日志监控, 运行任务后通过 WebSocket 实时接收并展示后端日志，方便用户了解任务执行进度与状态。

### WebSocket通信

* 每个画布对应一个 WebSocket 连接，通过任务 ID 绑定，接收模型的下载进度, 以及模型的所有节点执行的状态, 日志 耗时信息. 

# chenze：SAM3 接入 nndeploy

本目录存放 SAM3 接入的设计、交接、开发约束与验证记录。

## 阅读顺序

1. [开发与维护边界](sam3_development_rules.md)：开始修改前阅读，仓库公共部分改动须审核；本项目非公共目录可直接维护。
2. [第一版 Graph 划分](sam3_graph_design.md)：三个推理 Graph、共享状态与内部可替换边界。
3. [完整能力与内部模块边界](sam3_module_boundaries.md)：图片/视频概念与实例能力、模块契约及已有缺口。
4. [API 交接框架](sam3_api_handoff.md)：完整 SAM3 业务入口及 SAM2 协议兼容要求。

5. [运行环境对齐](sam3_environment.md)：同一虚拟环境的依赖、动态库加载、服务兼容与原生推理验证。

6. [Graph 实现规划](sam3_implementation_plan.md)：契约检查点、分步实施、文件范围与验收条件（规划，未开始实现）。

7. [工作流实现与使用](sam3_workflow_usage.md) / [实现验收记录](sam3_graph_validation.md)：三 Graph、共享 Runtime、视频取消、JSON 恢复及通用 Server 验证。

## 工作范围

- 业务实现：`python/nndeploy/segment/sam3/`。
- 专用测试：`test/sam3/`。
- 项目文档：本目录。

第一版计划提供 `SAM3ImageConceptGraph`、`SAM3ImageInteractiveGraph`、`SAM3VideoGraph`。三 Graph 已实现并完成模型与通用 Server 验证，使用与限制见上述实现文档；编辑器交互部署、专用 API 和 SAM2 兼容仍待后续接入。

2026-11-19：归档已有设计，记录共享仓库维护边界，建立专用测试目录；已开展环境对齐，结果见运行环境文档；已实现三业务 Graph，验证结果和后续服务范围另见实现验收记录。

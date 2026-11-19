# SAM3 Graph 实现验收记录

日期：2026-11-19。代码在远程 `/home/chenze/nndeploy` 的 `chenze` 分支工作区，基线 HEAD 为 `b7286d190c61362553ddca1d3628a3913e1f124d`；本轮未 commit/push。

## 1. 实现范围

- 三条高层 Graph 和三类业务 Node；输入准备/推理/结果三段通过真实 Edge 连接。
- 显式 RuntimeOwner、共享图片特征、独立上下文、持续视频 session、版本与引用校验。
- 图片概念、多实例、独立交互提示组、候选与连续 logits 精修。
- 视频概念、点/mask、多对象、多关键帧、正反向/双向范围、结果读取、对象移除和生命周期。
- full/partial/fetch 三条路径的帧级取消，取消后要求 reset/close；视频帧结果增量写盘。
- 模块替换/观测入口、工作流 JSON 保存恢复、公共 Server graph_factory 接入。

实现与用法见 [工作流使用](sam3_workflow_usage.md)。本轮尚未完成 SAM3 专用 HTTP 接口、SAM2 兼容、公司存储/回调或完整编辑器交互部署；这些能力不能根据本记录标记完成。

## 2. 环境与资产

沿用 `/home/chenze/nndeploy-runtime/develop/.venv` 与 `20261119-sam3-environment/sam3-env.sh`。Python 3.12.15，torch 2.10.0+cu128，torchvision 0.25.0+cu128，triton 3.6.0，NumPy 1.26.4。使用空闲 L40S，测试进程各自通过 CUDA_VISIBLE_DEVICES 选择 GPU 1/3；进程内设备为 cuda。

权重引用 `/home/chenze/sam3-runtime/checkpoints/sam3.pt`。图片为 truck 及其并排双目标组合；视频为 bedroom 前 24 帧。模型/视频回归固定 seed=42、compile=False、bfloat16、torch CPU 线程数 4、cuDNN benchmark=False，不作为性能基准。

## 3. 已执行的验证

| 验证 | 实际结果 |
| --- | --- |
| CPU 契约测试 | 5 项通过：坐标与端点、RLE/bbox、结果版本/缺失帧、局部替换不外泄、未知参数拒绝 |
| DAG 扩展机制 | dict Edge、Creator、JSON 往返、加载后的 init/run/deinit 通过 |
| 图片 Graph | 概念结果、实例结果和连续精修与当前同环境直接原生调用逐值一致 |
| 图片特征复用 | 同上下文概念后精修没有额外 vision_encoder 调用；两图片交错操作不串状态 |
| 多实例/多提示组 | 并排两辆 truck 返回 2 实例；两个独立提示组及单/多候选通过 |
| 图片失败与引用 | 真正借入预测器特征后注入异常，临时状态被清理，后续调用成功；旧引用和释放后引用被明确拒绝 |
| 视频 Graph | 24 帧前向对象 ID、二值 mask 和 out_probs 与当前同环境原生基线逐值一致 |
| 视频提示与状态 | 文本、初始正概念框、文本+正/负框；点击正负点、mask、多对象、多关键帧、双向和会话隔离通过 |
| 范围边界 | 后向完整区间，以及首/末单帧的 forward/backward/both 均返回约定绝对帧 |
| 实际取消 | full（首输出前）、partial、fetch 三条路径均停止；取消后拒绝继续推理，reset 后恢复运行 |
| 只读结果 | read 不增加任何模型调用计数；未计算和空目标可区分，提示更新后旧结果引用失效 |
| 工作流恢复 | 图片概念/实例及视频 Graph 保存、加载和 owner 重绑后实际运行通过 |
| 公共 Server 接入 | 通过 FastAPI TestClient 对现有 Server 发起 5 次进程内 HTTP 请求；加载已保存的两条图片工作流并共用上下文，vision_encoder 共调用 1 次，历史任务快照保持正确 |
| 原生默认调用回归 | 44 组、209 数组与 2026-11-19 同环境原始 SAM3 基线逐值一致；新增包内取消/范围标记未设置时保持默认行为 |
| 静态和维护边界 | 包内/测试 Python AST、git diff --check 通过；公共组件未被本轮修改，既有 config.cmake 修改哈希保持不变 |

取消延迟只作为小样例的机制证据，不把一次测量视为服务时限承诺。CPU 模块耗时与调用次数用于说明真实调用链，不用于声称已经优化或提速。上一轮历史环境的数值差异仍单独保留，不覆盖其失败记录。

## 4. 产物与复现

私有记录根目录：`/home/chenze/nndeploy-runtime/20261119-sam3-graphs/`。

| 文件 | 内容 |
| --- | --- |
| `before.json` | 开工 HEAD、已有修改与关键文件哈希 |
| `p0-dag.log` | DAG 原生扩展与序列化探针 |
| `contracts.log` | CPU 契约用例 |
| `image/report.json`、`image.log` | 最终图片验证与模块调用/张量元信息 |
| `video/report.json`、`video.log` | 最终视频验证、取消路径与模块元信息 |
| `server/report.json`、`server.log` | 公共 Server 运行已保存 Graph 的验证 |
| `native-regression.json` | 209 数组严格对照摘要 |
| `native-adaptation.json`、`native-before/` | 包内原生适配入口及修改前副本 |
| `final-summary.json` | 本轮完成状态、边界核对和限制 |

各测试命令见 [工作流使用](sam3_workflow_usage.md)。权重、运行结果、缓存与大文件不进入源码提交。

## 5. 下一阶段

工作流代码与通用 Server 工厂已具备实测基础。后续核对更新后的部署机制，完成 owner 生命周期、会话路由、取消控制通道及编辑器结果展示，再交付完整 SAM3 服务和 SAM2 兼容门面。原生纯负概念提示、跨 Worker 状态共享、持久化恢复和引擎优化保持未交付状态。

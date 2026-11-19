# SAM3 Graph 使用与验证

日期：2026-11-19。实现基线：`chenze` 分支 `b7286d1` 之后的未提交工作区。原生 SAM3、运行环境来源见 [环境对齐](sam3_environment.md)。

## 1. 当前交付

已实现三条 Python Graph、共享图片 Runtime、持续视频会话、结果引用与真实协作取消。三个 Graph 均使用已有 nndeploy Node/Graph/Edge 和 Creator 机制。保存的 JSON 可以通过现有 Graph 加载入口恢复，然后绑定同一个显式 owner。

| Graph | 默认推理操作 | 能力 |
| --- | --- | --- |
| `SAM3ImageConceptGraph` | `concept` | 文本、同图正负概念框、组合、连续查询、阈值更新、多实例、空结果 |
| `SAM3ImageInteractiveGraph` | `interact` | 正点/负点、实例框、点框组合、多提示组、多候选、概念后精修、低分辨率 logits 精修 |
| `SAM3VideoGraph` | 必须指定 `op` | 会话、概念、对象点/mask、多目标、多关键帧、传播/读取、移除、重置/关闭和取消 |

图片两 Graph 共用 owner.image；视频所有操作共用 owner.video。默认是原生 PyTorch、`compile=False`，本地显式权重，不下载模型。视频当前要求 CUDA/bfloat16；图片支持配置 float32/bfloat16。device 使用当前进程可见设备编号。

已验证保存的图片工作流通过公共 Server 的 graph_factory、JSON Edge 和 thread Worker 运行，两条 Graph 共用同一图片特征。本文不宣称已完成可视化编辑器全部交互控件、独立业务 HTTP API 或 SAM2 协议兼容；这些接入仍按 [API 交接框架](sam3_api_handoff.md) 后续实施。

## 2. 运行环境

以下命令在远程 Linux 中执行：

```bash
source /home/chenze/nndeploy-runtime/20261119-sam3-environment/sam3-env.sh
export CUDA_VISIBLE_DEVICES=3  # 按实际空闲资源选择
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
```

实现代码全部位于 `python/nndeploy/segment/sam3/`，本轮没有新增公共依赖。权重、JSON 工作流、视频逐帧结果和日志均保留在仓库外。运行中的旧 Web 服务未重启。

## 3. 图片概念后精修

以下是 Python 使用示例，checkpoint、输入和结果目录按实际运行设置：

```python
from nndeploy.segment.sam3 import ModelConfig, RuntimeOwner
from nndeploy.segment.sam3.graphs import (
    SAM3ImageConceptGraph, SAM3ImageInteractiveGraph,
)
from nndeploy.segment.sam3.workflow import execute

config = ModelConfig('/home/chenze/sam3-runtime/checkpoints/sam3.pt', device='cuda')
with RuntimeOwner(config, '/home/chenze/nndeploy-runtime/my-sam3/results') as owner:
    concept = SAM3ImageConceptGraph('concept', owner=owner)
    interact = SAM3ImageInteractiveGraph('interact', owner=owner)
    concept.init()
    interact.init()
    try:
        created = execute(concept, {
            'op': 'create', 'image': '/home/chenze/sam3/assets/images/truck.jpg',
        })
        assert created['ok'], created
        context = created['context_id']
        found = execute(concept, {'context_id': context, 'text': 'truck'})
        assert found['ok'] and found['instances'], found
        refined = execute(interact, {
            'context_id': context,
            'groups': [{
                'id': 'selected', 'multimask': True,
                'selection': {'result_id': found['result_id'], 'index': 0},
            }],
        })
        assert refined['ok'], refined
        # Use this result's candidate as the next low-resolution logits input.
        again = execute(interact, {
            'context_id': context,
            'groups': [{
                'id': 'selected', 'multimask': False,
                'refine': {
                    'result_id': refined['result_id'],
                    'group_id': 'selected', 'candidate': 0,
                },
            }],
        })
        assert again['ok'], again
        execute(interact, {'op': 'release', 'context_id': context})
    finally:
        concept.deinit()
        interact.deinit()
```

调用方应逐次检查 `ok`。业务校验失败返回 `ok=false` 与 `error.code/message`；模型加载和未预期的执行异常仍由 Graph/外层执行器报告，不能把它们当作成功结果。关闭借用 Graph 不会销毁 Runtime，owner 退出时统一释放资源。

### 图片请求规则

| 项目 | 规则 |
| --- | --- |
| 图片输入 | 本地路径、PIL 图片，或 HWC RGB uint8 ndarray；路径/PIL 按 EXIF 方向转正后处理 |
| 上下文 | `op=create` 返回 context_id；query/clear/release 操作复用该 ID；推理输入必须在 image 与 context_id 中二选一 |
| 一次性调用 | 直接传 image 自动建立并释放临时上下文；返回 context_id/result_id 为 null；连续精修使用显式上下文 |
| 坐标 | 原图像素点 `[x,y]`，框为 xyxy；mask 外接框右下端点排他；模型预测框另记 model_bbox_xyxy |
| 概念 | `text`、`boxes=[{box:[x0,y0,x1,y1],positive:true/false}]`；默认 update=replace；append 只追加框或更新 threshold |
| 清提示 | `op=clear` 清提示与当前结果，保留视觉特征；阈值保存于各自上下文，范围 [0,1] |
| 实例组 | `groups` 非空；每组 id 为唯一字符串，支持 points/labels、box、selection、refine 或 mask_logits，multimask 为 bool |
| 概念选择 | selection 只接受当前概念结果和实例 index；从其模型框进入实例分支，不将概念概率冒充 logits |
| logits | refine 指向当前结果/提示组/候选；或传实际 mask_logits（1,Hlow,Wlow）浮点数据；Hlow/Wlow 由当前模型校验，输出包含 logits_shape |
| 输出 | 概念是多个实例；实例交互是多个组各自的候选；mask 为压缩 COCO RLE，score_kind 区分 concept/mask_quality |
| 调试 | include_raw=true 可返回 CPU ndarray，适合原生数值对照，不用于 JSON 服务响应 |

纯负概念框、跨图概念示例、真正批量 GPU 推理未承诺支持；有文本或正概念框时可使用负框。多个独立图片/提示组先按正确顺序执行。

提示更新后依赖上轮提示的引用失效。调用方已取走的结果快照仍可独立使用，但不能把旧 result_id 再当作当前精修输入。外部修改结果列表/调试数组不会改变 Runtime 内部保存的选择与精修数据。

## 4. 视频操作

```python
from nndeploy.segment.sam3.graphs import SAM3VideoGraph
from nndeploy.segment.sam3.workflow import execute

graph = SAM3VideoGraph('video', owner=owner)
graph.init()
try:
    started = execute(graph, {'op': 'start', 'source': '/absolute/path/to/video.mp4'})
    assert started['ok'], started
    session = started['session_id']
    prompted = execute(graph, {
        'op': 'concept', 'session_id': session,
        'frame': 0, 'text': 'person', 'restart': True,
    })
    assert prompted['ok'], prompted
    tracked = execute(graph, {
        'op': 'propagate', 'session_id': session,
        'start': 0, 'end': started['num_frames'], 'direction': 'forward',
    })
    assert tracked['ok'], tracked
    frames = execute(graph, {
        'op': 'read', 'session_id': session,
        'result_id': tracked['result_id'], 'frames': [0, 1, 2],
    })
    execute(graph, {'op': 'close', 'session_id': session})
finally:
    graph.deinit()
```

该片段接续上一节的 owner；独立运行时需要自行创建 RuntimeOwner。视频 source 为原生支持的视频文件或顺序帧目录，自动下载/上传属于后续服务编排。

| op | 输入与状态语义 |
| --- | --- |
| start/query | source 建立会话；查询返回尺寸、帧数、版本、对象 ID 和状态 |
| concept | frame、text/boxes，必须 restart=true；概念更换会清原有跟踪任务。初始概念框最多一个；纯负视觉查询暂拒绝 |
| points | frame、object_id、points、labels；像素点，labels=0/1；同对象同帧新请求替换该次点击集合，不提供静默无效的 clear_old_points |
| mask | frame、object_id、原帧尺寸二值 ndarray 或压缩 RLE；可创建/修正对象及多个关键帧；同对象同帧后一次 mask 替换前一次输入 |
| propagate | start/end 为绝对帧半开区间；direction 为 forward/backward/both。两个方向均覆盖约定区间；双向先前向后后向，重复帧以后一次实际输出为准 |
| read | result_id 与最多 128 个 frames；只读本轮已完成结果，缺失帧返回 availability=not_computed；完成但无目标为 complete + 空 objects |
| remove | object_id 必须存在；移除后结果版本失效，后续传播按原控制器更新 |
| reset | 保留媒体，清提示/对象/跟踪，递增版本；旧结果引用不再是当前结果 |
| close | 关闭会话；不销毁其他会话使用的模型 |
| cancel | 仅请求停止活动传播；若 Graph 正在执行，应走下述独立控制路径 |

对象 ID 由概念发现产生，或由实例点/mask 请求指定。对象标识只在所属会话内有意义。当前高层没有视频“指定对象的实例框修正”，不能用概念 box 冒充。

建立会话后必须先加入概念或实例提示再传播；否则返回 prompt_required，避免隐式使用未初始化的提示状态。

视频 `objects[].score` 保留原生 out_probs，即 obj_id_to_score；使用 score_kind=object_score，不把它宣称为 tracker mask 质量。当前高层没有返回独立 tracker 分数，因此未虚构该字段。

### 取消与完成结果

同一个 Runtime 的有状态推理先串行。同一 Graph 实例不允许并发 run。长传播中由另一个控制线程/控制入口调用：

```python
reply = owner.video.cancel(session_id)
# reply.status == 'cancellation_requested' only acknowledges the request.
```

控制路径不等待推理锁，仅置停止标记。full（包括首个 yield 之前）、partial 和 fetch 循环会在帧边界检查；已提交的当前 CUDA 工作允许完成。传播 Graph 返回后通过 status/result.status 判断是否已 cancelled。

取消/执行失败后的会话必须 reset 或 close 后再执行推理。取消前已提交的帧保留；尚未提交或只存在于原生 hotstart 缓冲的帧不会冒充完成结果。取消后的读取仍可访问该版本已提交帧。通用 Server 的取消 HTTP 路由尚未与这个控制路径做专用适配，不能将排在长任务后面的普通 Graph cancel 请求当作即时取消。

逐帧 RLE 结果增量写入 owner.result_dir，各结果独立目录记录 manifest 和帧 JSON。Graph 不输出活 generator，也不累计全部 GPU mask。原生 tracker/视觉/帧缓存仍按原逻辑存在，本实现不宣称总体显存恒定。结果索引为当前 owner 的内存状态；磁盘文件保留但跨进程恢复引用尚未实现。

## 5. 工作流保存、恢复与现有 Server

导入 `nndeploy.segment.sam3.workflow` 会显式注册本包 Node/Graph，普通导入 sam3 包仍保持懒加载。

```python
from nndeploy.segment.sam3.workflow import load_workflow, server_factory

concept.save_file('/home/chenze/nndeploy-runtime/my-sam3/concept.json')
loaded = load_workflow('/home/chenze/nndeploy-runtime/my-sam3/concept.json', owner)
loaded.init()
# execute(loaded, request)
loaded.deinit()
```

JSON 保存 model_config 等稳定配置，不保存 GPU 模型、上下文、session、精修数据或凭证。不同配置的 owner 会被拒绝绑定。已有上下文只在原 owner 还活着时可延续，加载 JSON 本身不会恢复已关闭进程的状态。

无权重加载地生成三个工作流：

```bash
python -B -m nndeploy.segment.sam3.workflow \
  --checkpoint /home/chenze/sam3-runtime/checkpoints/sam3.pt \
  --result-dir /home/chenze/nndeploy-runtime/my-sam3/results \
  --export-dir /home/chenze/nndeploy-runtime/my-sam3/workflows
```

命令行也支持 `--kind concept|interact|video --requests /absolute/requests.jsonl --output /absolute/responses.jsonl`，或 `--workflow /absolute/workflow.json`。每个请求一行 JSON，同一进程保留 owner。跨两类图片 Graph 的联动使用 Python owner 或下述公共 Server 工厂。

复用现有 Server 时使用 `graph_factory=server_factory(path, owner)`，输入/输出均为 type=json，边名 request/result，execution.isolation=thread；两个图片入口传同一 owner。具体可运行范例见 `test/sam3/check_server_workflows.py`。进程隔离 Worker 不能直接传递这个 owner；跨 Worker 路由与编辑器的状态化部署策略留到服务接入阶段核对。

## 6. 内部模块与测量边界

NativeTorch 自然入口通过 `modules.boundary` 接入 operation-local Modules。保留 vision_encoder、tokenizer、text_encoder、concept_prompt/encoder/decoder/mask_head、instance_segmentor/heads、memory_attention/encoder、association、tracking_plan/commit、视频检测/帧/后处理等真实调用边界。

`Modules(replacements={name: replacement}, measure=True)` 中 replacement 的签名为 `(native_function, *args, **kwargs)`，包含原生方法 self；默认委托原函数。替换只对当前 owner 的操作上下文生效，不修改公共函数或权重键。后端替换仍须满足该原生入口的完整特征、坐标、dtype/device、可变性与状态提交契约，不能以一个 embedding 代替多尺度特征。

measure=True 记录每模块 CPU 墙钟时间和最近一次有界张量结构（shape/dtype/device），不读取张量值、不逐模块同步 CUDA。默认保留调用计数，`observe=False` 可关闭观测；替换仍然生效。图片上下文命中、视频 full/partial/fetch 分别计数。已有原生 profiler 区域保留，可后续用正式 profiling 工具采集 GPU 时间。

这些 CPU 时间包含嵌套/等待，不是 GPU 内核耗时，也不能相加推导 Graph 总时长。`cpu_wall_seconds` 为当前推理 Node 的一次请求时间，不包含外部队列和 HTTP 等待。此轮不报告性能提升。

当前测试模型的实测特征示例（不是对任意模型版本的静态 shape 保证）：

| 边界 | 实测输入/输出及替换约束 |
| --- | --- |
| VisionEncoder | 输入 NCHW `[1,3,1008,1008]` float32；主特征 `[1,256,72,72]`，FPN 和位置编码含 288/144/72 三尺度 bfloat16，另外保留完整 sam2_backbone_out；processor 还会对实例高分辨率特征做 conv_s0/conv_s1 投影，替换时必须保留该顺序 |
| 图片实例 | 输出 `M×Horiginal×Woriginal` mask、`M` 质量分数、`M×288×288` logits；M 是候选数，多个提示组另行索引；候选数据复制到 CPU，精修所需数据由 ImageContext 持有 |
| MemoryAttention | 当前特征/位置为 `[5184,B,256]`，历史 cond/non-cond 输出和帧方向共同决定读取内容；示例输出 `[B,256,72,72]` float32，不能无条件压回 bfloat16 |
| MemoryEncoder | 输入包含投影后的 32/64/256 通道多尺度特征、最终高分辨率 mask、object score 和状态元数据；示例输出 `[B,64,72,72]` bfloat16 memory 及位置编码，由原控制器按原时序提交 |

上述特征由 Runtime/原生状态持有，不作为公共 JSON。输入可能包含由上下文复用的缓冲区，替换后端不能任意原地改写；输出须在调用者消费的 CUDA stream 上有正确依赖。首版没有跨 stream 的异步后端或已有 memory 的后端迁移，切换实现应在新 owner/新会话边界进行并重新验证。

## 7. 专用验证与记录

在第 2 节环境中执行：

```bash
python -B /home/chenze/nndeploy/test/sam3/test_contracts.py
python -B /home/chenze/nndeploy/test/sam3/check_graphs.py \
  --phase image --checkpoint /home/chenze/sam3-runtime/checkpoints/sam3.pt \
  --image /home/chenze/sam3/assets/images/truck.jpg \
  --output /home/chenze/nndeploy-runtime/my-sam3/image-check
python -B /home/chenze/nndeploy/test/sam3/check_graphs.py \
  --phase video --checkpoint /home/chenze/sam3-runtime/checkpoints/sam3.pt \
  --frames /home/chenze/nndeploy-runtime/20261116-sam3-migration/frames \
  --output /home/chenze/nndeploy-runtime/my-sam3/video-check
python -B /home/chenze/nndeploy/test/sam3/check_server_workflows.py \
  --checkpoint /home/chenze/sam3-runtime/checkpoints/sam3.pt \
  --image /home/chenze/sam3/assets/images/truck.jpg \
  --output /home/chenze/nndeploy-runtime/my-sam3/server-check
```

图片回归当前固定 truck 示例，视频回归当前固定 bedroom 的 24 帧；脚本中的提示与帧号用于这些验收资产，不代表任意输入内容都应产生相同对象。所有权重、资产、设备和输出位置显式传参。

本轮真实结果汇总见 [Graph 实现验收记录](sam3_graph_validation.md)。日志和数值产物位于 `/home/chenze/nndeploy-runtime/20261119-sam3-graphs/`，不加入源码仓库。

包内原生文件只增加可选模块边界、取消检查和 workflow-only 后向起始帧适配；普通原生调用的语义保留。迁移时的 MIGRATION.json 是历史源码快照清单，当前集成修改不会伪装成仍与旧清单逐文件相同；本轮差异与回归单独记录。

# e_dit

基于 nndeploy DAG 的 DiT 推理模块。首版验收模型为 EraserDiT，运行环境为 Linux + NVIDIA CUDA。
模型计算、视频窗口、缓存、量化、融合、卸载和分布式实现迁自 EraserDiT；原 CLI、HTTP 服务和 WebUI 不包含在本包中。
原项目许可证见 `LICENSE.EraserDiT`，重构后的文件布局见下文。

## 数据与生命周期

- `DiTParams`：nndeploy Param，保存模型路径、组件和优化配置，可序列化。
- `DiTResources`：持有常驻权重、组件配置和通信资源，不承担请求执行。
- `Req`：请求级状态，包含采样参数、随机数生成器、条件和中间张量；由 `nodes/schedule_batch.py` 定义，包入口提供延迟导出。
- `build_graph()`：按算法创建完整可配置模型图。由原生 `dag.Node`、`dag.Graph`、`dag.Loop` 组成，统一负责请求执行与循环调度。

Edge 传递共享 State 槽位；Tensor 存在请求 State 中。每次执行结束清空槽位，避免图持有上一请求的 Tensor。
窗口循环和去噪循环均为原生 Loop，循环体的 Node 由 nndeploy DAG 执行。视频窗口规划和提交语义与原实现一致。

```text
EraserDiT (Graph，输入 video / mask / sampling，输出视频 bytes)
└─ InferenceGraph（预热、随机种子、融合上下文）
   └─ RequestGraph
      ├─ PrepareRequestNode
      ├─ WindowLoop
      │  ├─ WindowValidation → TextEncoding → Preprocess
      │  ├─ ConditionEncoding → LatentPreparation → TimestepPreparation
      │  ├─ Denoising (Loop)
      │  │  └─ DiTPredictNode → SchedulerStepNode
      │  └─ Decoding → WindowPostprocess → WindowCommitSync
      └─ FinalizeOutputNode
```

`DiTPredictNode` 完成 DiT 前向、CFG 与缓存策略；`SchedulerStepNode` 完成调度器更新和采样步进度。
Transformer block 和模型优化策略属于算法实现；通用 Attention、量化和融合 kernel 属于公共算子层。
上述嵌套节点及连线由 Python 代码构建，算法通过 `build_graph()` 或 `build_workflow()` 创建完整拓扑。

源码布局：`algorithms/eraserdit/` 集中模型、算法配置、计算节点、请求图、视频擦除流程及优化策略。
公共 `nodes/` 提供 DAG 基类，`layers/quantization/` 提供 INT8/FP8 算子，
`distributed/process_pool.py` 管理通用计算进程，算法目录中的 `parallel/workers.py` 定义具体计算协议。
算法内部旧导入路径已删除。公共工厂根据模型注册表创建算法图。
算法通过 `algorithms/base.py` 中的 `AlgorithmAdapter` 接口扩展。
不再提供 `pipelines/`、`PipelineStage`、Session 或独立执行器。

首版提供完整模型节点，不承诺编辑器中的任意阶段重组。模型加载使用 PyTorch；nndeploy 其他推理后端不自动适用于本模块。

## 环境

先构建可运行的 nndeploy 原生绑定，再在仓库的 `python/` 目录执行 `python -m pip install -e .`。
普通安装现在默认包含 DiT 计算库和所有 Server 扩展依赖，不再需要指定 `[e-dit]` 或 `[server-*]`。
原来的 extra 名称保留为空别名，已有安装命令仍然可用。
这些默认版本仍受各依赖支持的 Python、操作系统和硬件范围限制。

`algorithms/eraserdit/requirements.txt` 是 EraserDiT 的特定环境依赖配置（Python 3.10、Torch 2.6.0+cu126、Triton 3.2.0），
包含默认安装之外的优化、测试和测量工具。普通安装不加载它；选择 EraserDiT 安装时，会应用其中所有依赖。
其中 FlashAttention 等源码扩展需要先安装匹配的 Torch 和构建工具，再按其构建要求安装。
FFmpeg / ffprobe 需要在 PATH 中。FlashAttention、SageAttention、bitsandbytes 按配置选装，SDPA 路径不要求安装它们。
模型权重位于外部目录，不复制进 Python 包。
建议在启动 Python 前设置 `NUMPY_MADVISE_HUGEPAGE=0`，与原 EraserDiT 入口默认值一致，减少共享机器上的大页分配/整理开销；显式设置的值仍由调用者决定。

### 算法依赖覆盖

在仓库根目录执行以下命令，会自动选择 EraserDiT 依赖文件、使用其中的 CUDA 下载源，并替换默认版本：

```bash
python python/install.py nndeploy/e_dit/algorithms/eraserdit/requirements.txt --editable
```

支持传入依赖文件或目录；传入目录时自动读取其中的 `requirements.txt`。
相对路径以 `python/install.py` 所在目录为基准，与命令执行目录无关，也支持绝对路径。例如：

```bash
python python/install.py /absolute/path/to/eraserdit --editable
```

仍支持 `--algorithm eraserdit` 简写，与直接传文件路径二选一。
省略 `--editable` 可执行普通源码安装。脚本使用当前 Python 解释器的 pip，因此请先激活目标虚拟环境。
它在同一次 pip 安装中传入算法 requirements 和 nndeploy 源码；`setup.py` 在依赖解析前替换默认要求，
最终包元数据也记录替换后的版本。已有库版本不符合算法要求时，由 pip 升级或降级。
后续安装其他算法时，传入对应的依赖文件路径。文件不存在时直接报错，不会静默回退。
不会在运行模型时自动调用 pip；安装完成后需重启服务和计算进程。

算法有不同的依赖要求时，在 `algorithms/<算法名>/requirements.txt` 中只列出需要替换或新增的库，
每行使用一个 PEP 508 依赖声明。例如，选择 CUDA 12.6 的 Torch 构建：

```text
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.6.0+cu126
torchvision==0.21.0+cu126
```

上述安装脚本也可以等价地写成以下命令，在仓库的 `python/` 目录执行：

```bash
NNDEPLOY_DIT_ALGORITHM=eraserdit python -m pip install -e . \
  -r nndeploy/e_dit/algorithms/eraserdit/requirements.txt
```

也可继续通过 `NNDEPLOY_REQUIREMENTS_OVERRIDE` 指定任意覆盖文件路径；与 `NNDEPLOY_DIT_ALGORITHM` 二选一。
自定义覆盖路径可以是绝对路径；相对路径以 `python/setup.py` 所在目录为基准。
同名库按规范化名称匹配，算法声明**完整替换**默认版本要求（包括 extras 和环境标记），新增库追加，未列出的库保持默认。
一个库的多条环境标记声明会一起替换默认项，因此需要覆盖所有目标平台。
支持空行、整行注释和空白后的行尾注释，以及 `--index-url`、`--extra-index-url`、`--find-links`、`--trusted-host`、`--no-index`。
索引选项由安装脚本通过 pip 的 `-r` 自动传入，不写入包元数据。
直接设置环境变量时，也需要将同一文件通过 `-r` 传给 pip，或在 pip 命令中指定索引；环境变量本身不会配置 pip 下载源。
覆盖文件暂不支持嵌套 `-r`、`-c`、可编辑依赖和其他 pip 指令。

覆盖发生在源码安装或构建 wheel 时，最终依赖写入包元数据，pip 可以据此升级或降级已有库。
仅对预构建 wheel 设置这些变量不会修改其元数据；切换算法时需要通过安装脚本重新安装源码或重新构建 wheel。
每次重装都需指定所选算法或覆盖文件，未设置变量会恢复默认依赖要求。
覆盖只替换 nndeploy 自身的要求，其他依赖之间的冲突仍由 pip 报错，需要在算法文件中提供相互兼容的版本组合。
安装后执行 `python -m pip check`，并重启服务和计算进程。
同一个 Python 环境只提供一套库版本；需要同时运行版本互不兼容的算法时，使用独立虚拟环境或容器分别部署。

## Python

```python
from pathlib import Path
from nndeploy.e_dit import DiTParams, build_graph

params = DiTParams(
    '/models/EraserDiT',
    device='cuda', weight_dtype='bf16',
    dit_cpu_offload=True,
    text_encoder_cpu_offload=True,
    vae_cpu_offload=True,
    pipeline={'vae_low_memory': True},
    attention_backend='sdpa',
)
graph = build_graph(params)
try:
    graph.init()
    video = graph.infer(
        '/data/input.mp4', '/data/mask.mp4',
        prompt='There is a bridge over the lake.',
        seed=42, num_inference_steps=50, guidance_scale=3., strength=.8,
        runtime_mode='windowed_streaming', streaming_cache_dtype='uint8',
    )
    Path('/data/result.mp4').write_bytes(video)
finally:
    graph.deinit()
```

再次调用 `infer` 复用模型，采样参数通过输入端口传递，不重新初始化权重。
在 `DiTParams` 中设置 `warmup=True, warmup_steps=1` 可让首次请求执行预热；后续请求复用已预热模型。
预热保留取消支持，不提前推进正式请求的进度；本次预热摘要可通过 `graph.model.last_warmup` 查看。
Server 的取消与进度通过 nndeploy DAG 执行上下文传入请求 State；`DiTResources` 不提供 `run` 方法。

实例配置见 `config/server_args.py`，模型配置见 `algorithms/eraserdit/config.py` 的 `EraserDiTPipelineConfig`，请求参数见同文件的 `EraserDiTEraseSamplingParams`。
公共环境开关使用 `NNDEPLOY_DIT_*`，EraserDiT 专属开关使用 `NNDEPLOY_DIT_ERASERDIT_*`，Wan 视频编辑开关使用 `NNDEPLOY_DIT_WAN_VIDEOEDIT_*`。不再读取旧环境变量名称。
`pipeline`、`EraserDiTPipelineConfig`、`pipeline_class_name` 是模型配置字段或类型，不对应独立 Pipeline 执行实现。

## 工作流与服务

```python
from functools import partial
from nndeploy.server import Server
from nndeploy.e_dit import build_graph, server_config

# params 为上例中的 DiTParams；进程工厂只携带可序列化配置。
server = Server(
    name='eraserdit', data_dir='resources/eraserdit',
    graph_factory=partial(build_graph, params.to_dict()),
    config=server_config(),
)
# 放在 if __name__ == '__main__': 内启动，便于 spawn 子进程导入。
server.run(host='127.0.0.1', port=8000)
```

服务复用 nndeploy `/v1/server/eraserdit/deploy`、文件接口、任务记录和事件流。请求示例：

```json
{
  "inputs": {
    "video": {"path": "/data/input.mp4"},
    "mask": {"path": "/data/mask.mp4"},
    "sampling": {"prompt": "There is a bridge over the lake.", "seed": 42}
  }
}
```

`video` / `mask` 也接受上传后的文件引用；输出 `video.file_id` 可通过 nndeploy 文件接口下载。
采样步和窗口进度以事件中的 `progress` 对象提供。长视频需要按文件大小、排队时间和执行时间配置 Server 限额。
默认使用隔离进程，并开启 `execution.allow_child_processes`，以允许 NCCL 子进程。EraserDiT 的 `shutdown_timeout_seconds` 为 180 秒，用于释放多卡权重和关闭计算进程。
取消或超时会停止隔离 Worker 及计算子进程组，下次请求重新加载模型；Python 直接调用可通过 cancellation token 在模型安全检查点协作取消。

所有算法工作流均通过代码构建。`build_graph()` 提供推理端口；
需要包含输入、采样配置和文件输出的完整工作流时，使用 `build_workflow()`：

```python
from nndeploy.base import StatusCode
from nndeploy.e_dit import build_workflow
workflow = build_workflow(
    params, video='/data/input.mp4', mask='/data/mask.mp4',
    output='/data/result.mp4', sampling={'prompt': 'There is a bridge over the lake.'},
)
try:
    status = workflow.init()
    if status != StatusCode.Ok:
        raise RuntimeError(f'Workflow initialization failed: {status}')
    status = workflow.run()
    if status != StatusCode.Ok:
        raise RuntimeError(f'Workflow execution failed: {status}')
finally:
    workflow.deinit()
```

该工作流在代码中连接视频路径、mask 路径、采样输入、完整模型和视频输出五个节点。
Wan、VideoEdit、LTX-2 和 VSR 同样通过各自的 `workflow.py` 构建。
服务请求使用上面的 `build_graph` 端口绑定，保持请求参数与模型配置分离。
Server 依照 `model_path` 的模型资源声明管理模型目录。
编辑器使用同一个 `nndeploy.e_dit.EraserDiT` 节点。

## 验证

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 python -m pytest test/e_dit -q
# 原仓库对照；未设置源码路径时相应测试会跳过。
ERASERDIT_SOURCE=/path/to/EraserDiT VIDEOEDIT_SOURCE=/path/to/VideoEdit/sglang \
  PYTHONPATH=python OMP_NUM_THREADS=1 python -m pytest \
  test/e_dit/test_eraserdit_upstream.py \
  test/e_dit/videoedit/test_upstream_orchestration.py -q
```

测试从 nndeploy 仓库根目录运行。EraserDiT 对照覆盖相同小模型权重的 Transformer
FP32/BF16 输出；VideoEdit 对照覆盖原去噪方法与 DAG Loop 的逐步预测及视频输出，
共用模型和媒体实现，不等同于独立完整 SGLang 推理验收。
VSR 真实权重独立进程对照入口为 `test/e_dit/vsr/align_sglang.py`，
用 `--help` 查看原仓库、权重和输出目录参数。
GPU 回归在启动 Python 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，避免测试间启用确定性后影响已有 cuBLAS 上下文。
GPU 测试的跳过不代表通过。

公共资源由 `DiTResources` 管理，算法通过 `AlgorithmAdapter` 组合原生 DAG 节点。

## LTX-2 算法

新增 `model='ltx2'` 与 `model='ltx2_two_stage'`，支持文生音视频、首帧条件生成及两阶段上采样/refinement。
模型、计算节点和配置位于 `algorithms/ltx2/`；继续使用原生 DAG。
使用方法、组件路径、LoRA、支持边界及验证命令见 [LTX-2 说明](algorithms/ltx2/README.md)。

### Wan

新增 Wan 2.1/2.2、I2V/TI2V、DMD/Self-Forcing 和 VideoEdit 算法入口。
它们共享组件加载、执行上下文和原生 DAG/Loop；VideoEdit 支持按窗口直接输出文件。
模型名、用法、依赖和验证边界见 [Wan 使用说明](algorithms/wan/README.md)。

## Wan VSR 视频超分

`model='wan_vsr'` 接入单步视频恢复、流式分块、颜色校正、音轨保留和多卡 tile 推理。
使用 `build_graph()`、`build_workflow()` 与 `server_config(model='wan_vsr')`；
长视频可用 `graph.infer_file(video, output, target_resolution='1080x1920')` 直接写文件。
HTTP 服务通过公共 `python -m nndeploy.server --api-only --services services.json` 启动，
或向已有 Server 调用 `add_vsr(server, params)` 注册；不再提供算法侧 `create_server()`。
配置与联合部署示例见 [公共服务入口](../server/README.md)。

## 公共组件契约

无需权重即可查询模型声明；`enforced=false` 表示仍由该算法已有校验器检查，
空能力列表不代表支持所有设备或后端。能力组合仍可能受算法附加约束限制。

```python
from nndeploy.e_dit import describe_model

spec = describe_model('wan_i2v')
print(spec['capabilities'])
print(spec['components'])
```

- `config/capabilities.py`：Wan/LTX 的设备、后端和运行功能前置校验；错误包含 `algorithm`、`parameter`、`value`、`reason`。
- `loader/contracts.py`：组件必需性、依赖顺序与加载报告。`resources.loading_report` 中每条记录可调用 `to_dict()`；记录已加载、注入、跳过或失败以及耗时、参数量、设备与 dtype。当前报告覆盖公共加载器，算法 hook 内额外加载的组件尚未纳入。
- `config/request_validation.py`：生成请求公共字段校验；算法继续校验尺寸对齐、条件和特殊采样规则。
- `models/schedulers/request.py`：每请求复制 scheduler 并初始化 schedule；不修改模板。算法保留更新公式、prediction type 转换和特殊 schedule。
- `utils/random.py`：Wan/LTX 共用独立随机数生成器，保持历史 `seed + 样本索引` 规则，不修改全局 RNG。
- `memory/component_use.py`：整组件 eager 执行的驻留上下文，支持嵌套、异常卸载和组件级互斥；不用于已有分层卸载/FSDP 管理器接管的模块。默认保持权重 dtype，autocast 必须显式指定。
- `nodes/conditioning.py`：图像编码和 VAE posterior mode 编码操作，已接入 Wan；归一化、mask 拼接仍属于算法。公共文本编码接入同一驻留上下文，供 Wan/LTX 使用。
- `video/media.py`：编码边界统一为 FHWC RGB，接受 uint8 或有限浮点数（编码时截断至 `[0,1]`）；音频为 CN，采样率必须为正整数。帧率用 Fraction 保留，视频字节接口保持兼容。

这些接口是公共基础建设的首批实现，不代表全部需求已完成。
测试见 `test/e_dit/test_common_contracts.py`。后续工作包括统一媒体结果、视频流与窗口基础设施、缓存失效协议、后端选择及性能指标；共享 CPU 权重、插帧和超分节点按实际场景推进。

## 当前接口

只支持当前导入路径，不再提供旧模块或名称的转发：

- 公共入口：`DiTParams`、`DiTResources`、`Req`、`build_graph()`、`build_workflow()`。
- EraserDiT 具体类型从 `algorithms.eraserdit.graph`、`algorithms.eraserdit.config` 导入。
- 分布式运行时从 `parallel.runtime` 导入。
- LTX 模型名为 `LTX2VideoTransformer3DModel`，不再导出 `LTXModel`。
- 内存事件时间标识使用 `timestamp`，不再提供拼写错误的 `timestemp`。

旧环境变量不再生效：原公共 `MGERASE_*` / `ERASERDIT_DETERMINISTIC` 开关使用对应的
`NNDEPLOY_DIT_*` 名称；算法专属 `MGERASE_*` 使用 `NNDEPLOY_DIT_ERASERDIT_*`；
`SGLANG_VIDEOEDIT_*` 使用 `NNDEPLOY_DIT_WAN_VIDEOEDIT_*`。
现有工作流中的节点 key 仍是当前节点的注册名称，正常保留。

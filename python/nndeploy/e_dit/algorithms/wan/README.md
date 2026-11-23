# Wan 算法

源自 SGLang `multimodal_gen`。
迁移代码保留源文件的 Apache-2.0 声明。
运行不依赖 SGLang。公共执行入口是 nndeploy 原生 Graph、Node 和 Loop。

## 模型入口

| `DiTParams.model` | 源配置/用途 |
|---|---|
| `wan` | Wan 2.1 T2V 1.3B / 480P |
| `wan_t2v_720p` | Wan 2.1 T2V 14B / 720P |
| `wan_i2v`、`wan_i2v_720p` | Wan 2.1 I2V，包含 CLIP 图像条件 |
| `wan_fun` | Wan 2.1 Fun 1.3B InP 的源 I2V 配置 |
| `wan_dmd`、`wan_turbo` | FastWan / TurboWan 文生视频 DMD 步表 |
| `wan_i2v_dmd` | TurboWan 2.2 I2V 双模型 DMD |
| `wan2_2_ti2v`、`wan2_2_ti2v_dmd` | Wan 2.2 TI2V 5B，首帧约束及展开 timestep |
| `wan2_2_t2v`、`wan2_2_i2v` | Wan 2.2 A14B 高/低噪声模型切换 |
| `wan_causal_dmd` | Self-Forcing，分块因果注意力及 KV/text 缓存 |
| `wan_videoedit` | 本地 VideoEdit 扩展，掩码、参考图、窗口传播、动态 CFG、回填 |
| `wan_videoedit_dmd` | VideoEdit 蒸馏权重，源网关的固定四步、guidance=1、无动态 CFG/负向提示词策略 |

每个入口必须对应其训练权重，不能仅修改 model 名称复用不匹配的 checkpoint。
使用 Diffusers 目录布局和 `model_index.json`。双模型目录需包含 `transformer_2`。
模型结构以组件的 `config.json` 为准，权重加载严格检查名称和形状。

## 生成

```python
from pathlib import Path
from nndeploy.e_dit import DiTParams, build_graph

params = DiTParams(
    model="wan", model_path="/models/Wan2.1-T2V-1.3B-Diffusers",
    device="cuda:0", dit_cpu_offload=True,
    text_encoder_cpu_offload=True, vae_cpu_offload=True,
)
graph = build_graph(params)
try:
    graph.init()
    video = graph.infer(
        "A quiet river at sunrise.", seed=10,
        height=480, width=832, num_frames=81, num_inference_steps=30,
    )
    Path("wan.mp4").write_bytes(video)
finally:
    graph.deinit()
```

I2V/TI2V 使用 `image_path="/data/first_frame.png"`。I2V 的目标尺寸按请求指定，图像以
Lanczos 缩放。普通 Wan 帧数为 `4*n+1`，宽高需满足 VAE 与 patch 的整除约束。
`infer_batch(prompt=[...])` 返回 `OutputBatch`，视频数组形状为 `BFHWC`，值域 `[0,1]`。
MP4 输出端口接受单个结果。随机数生成器、调度器及因果缓存按请求隔离。

LoRA 使用 `DiTParams(lora_path="/models/adapter.safetensors", lora_scale=1.0, ...)`，
通过可撤销的 Linear hook 应用，不修改基础权重。Wan 2.1 非 DMD 模型可在请求中设置
`enable_teacache=True`，使用源配置的拟合系数；未校准的 Wan 2.2 和 DMD 请求明确拒绝此选项。

## VideoEdit

```python
params = DiTParams(
    model="wan_videoedit", model_path="/models/VideoEdit-diffusers-model",
    device="cuda:0", dit_cpu_offload=True, text_encoder_cpu_offload=True,
    image_encoder_cpu_offload=True, vae_cpu_offload=True,
)
graph = build_graph(params)
try:
    graph.init()
    result_path = graph.infer_file(
        "edited.mp4", prompt="A red jacket", seed=10,
        video_input_path="/data/input.mp4",
        mask_input_path="/data/masks.npy",
        reference_image_path="/data/reference.png",
        infer_len=49, overlap=5, bridge_overlap=5,
        num_inference_steps=40, use_clip=False,
    )
finally:
    graph.deinit()
```

`infer_file` 按窗口读取和写出，保留输入帧率，并按 `preserve_audio=True` 保留原音轨。
`infer`/`infer_batch` 会收集完整视频到内存；长视频优先使用 `infer_file`。
掩码支持源实现的视频、NumPy 和 COCO 表达；支持参考帧双向传播、窗口 overlap、裁剪、
掩码稳定化和 paste-back。VideoEdit 的 VAE 和文本编码使用源算法指定的原生
Diffusers `AutoencoderKLWan` 和 Transformers `UMT5EncoderModel`。

`use_clip=True` 需要 `image_encoder`；`clip_preprocess="diffuser"` 还需要
`image_processor`。默认查找模型目录同名子目录，也可通过 `component_paths` 指定。
每个窗口内部使用独立的原生 DAG，去噪的每一步通过 Loop 执行并检查取消。
视频工具使用 `NNDEPLOY_DIT_WAN_VIDEOEDIT_*` 环境变量，不再读取旧前缀，不需要安装 SGLang。

### 文件输出和命令入口

`graph.infer_files("edited.mp4", save_crop_only=True, ...)` 返回主视频和
`edited_crop_only.mp4` 两个路径。裁剪输出需要文件模式，输出路径不得覆盖视频、掩码或参考图。
`decode_mode="stream"` 按窗口解码；`"eager"` 预读视频。两者均通过原生 DAG 执行。

```bash
python -m nndeploy.e_dit.algorithms.wan.videoedit_cli \
  --model-path /models/VideoEdit --video-input-path input.mp4 \
  --mask-input-path masks.npy --reference-image-path reference.png \
  --prompt "A red jacket" --output edited.mp4 --no-use-clip --save-crop-only
```

使用 DMD 权重时添加 `--variant dmd`，或设置 `DiTParams(model="wan_videoedit_dmd", ...)`。
该入口按源策略覆盖步数、guidance、dynamic_cfg 和 negative_prompt；不能使用普通权重替代。
其他采样与运行配置分别由 `--sampling-json` 和 `--runtime-json` 指定。

公共视频对比工具逐帧计算 SSIM、PSNR、MSE 和 MAE，可用于迁移结果验收：

```bash
python -m nndeploy.e_dit.video.compare \
  --reference reference.mp4 --candidate edited.mp4 --report-json compare.json
```

HTTP 服务使用下述算法独立入口；原生图入口仍负责单设备推理，不包含多进程共享权重。

## 工作流和依赖

工作流统一通过 Python 代码构建，`workflow.py` 使用 `Graph`、`Edge` 和算法节点连接输入、模型与视频输出。
`wan_workflow.py`、`wan_videoedit_workflow.py` 提供完整示例，使用前修改模型路径、采样输入和输出路径：

```bash
python -m nndeploy.e_dit.algorithms.wan.wan_workflow
python -m nndeploy.e_dit.algorithms.wan.wan_videoedit_workflow
```

示例中的 `create_workflow()` 只构图，`main()` 负责初始化、执行和释放资源。
其他 Wan 变体使用 `nndeploy.e_dit.build_workflow(params, sampling={...}, output="result.mp4")`，
通过 `DiTParams.model` 选择算法，内部节点和连线由代码创建。
服务 schema 可由算法适配器的 `server_config()` 获取，与现有服务执行控制共享取消和进度接口。

安装 `nndeploy[e-dit]`，VideoEdit 文件输出还需要 PATH 中的 `ffmpeg` 和 `ffprobe`。
新增公共能力包括独立 cos/sin RoPE、时间嵌入接口、UniPC/FM 调度器、时序相似度缓存、
图像编码器 offload 选项及分数帧率 MP4 封装。`repack.py` 保留官方格式 checkpoint 的转换工具。

## 验证范围

支持单设备 CPU/CUDA、SDPA、FP32/BF16 和组件级 CPU offload。多卡、稀疏注意力、
量化、FSDP、layerwise offload、编译及原生 Wan VAE 空间 tiling 未接入该图入口，
相关配置会提前报错。VideoEdit 使用的 Diffusers VAE tiling 可用。
转换工具能重命名量化 checkpoint，但该图入口不运行量化权重。

验证以小尺寸真实 Transformer/VAE、完整小 checkpoint 加载、原生 DAG、CPU/CUDA、
JSON roundtrip 和流式窗口测试为依据，测试位于 `test/e_dit/`。
尚未完成真实 1.3B/5B/14B checkpoint 的生成质量、多卡和长视频性能验收。

## VideoEdit 服务

`server/` 提供原 VideoEdit 的业务协议，复用 `nndeploy.server` 的任务表、Worker、
文件存储、鉴权和回调投递。没有另建任务队列或 FastAPI 应用，也不依赖 SGLang。

```python
from nndeploy.server import Server
from nndeploy.e_dit import DiTParams
from nndeploy.e_dit.algorithms.wan.server import add_videoedit

server = Server(resources="./resources",
                storage_config={"max_bytes": 5 * 1024**3,
                                "media_max_duration_seconds": 86400, "media_max_dimension": 16384})
add_videoedit(server, {
    "normal": DiTParams(model="wan_videoedit", model_path="/models/VideoEdit", device="cuda"),
    "dmd": DiTParams(model="wan_videoedit_dmd", model_path="/models/VideoEdit-DMD", device="cuda"),
})
server.run(host="127.0.0.1", port=30000)
```

生产环境使用上面的独立进程 Worker；Linux 下同时管理 FFmpeg 子进程组。
normal 和 DMD 常驻同一 Worker、共用一个接收名额，不排队，繁忙返回
`{"code":2,"message":"A task is running."}`。按显存配置 `DiTParams` 的组件卸载选项；
DMD 必须使用对应权重，并固定四步、guidance=1、关闭 dynamic_cfg 和 negative_prompt。
单个后端初始化失败会记录日志；normal 可用而 DMD 不可用时继续提供 normal 服务。

也可以用公共 Server CLI 启动。将模型配置放入 `services.json`：

```json
{
  "services": [{
    "algorithm": "videoedit",
    "name": "videoedit",
    "options": {"models": {"normal": {"model_path": "/models/videoedit"}}}
  }]
}
```

```bash
python -m nndeploy.server --api-only --services services.json \
  --resources ./resources --host 127.0.0.1 --port 30000
```

安装 `nndeploy[e-dit,server]`，对象存储另安装 `nndeploy[server-s3]`，PATH 中需有
`ffmpeg`、`ffprobe`。文件大小、配额及保留策略统一由公共 Server 管理；默认上限为
16 MiB、600 秒、4096 像素，需要按视频大小通过公共 `--storage-config` 显式配置。
同一份服务配置可同时注册 VSR，共享数据库、存储和生命周期，详见
[公共服务入口](../../../server/README.md)。算法包不再提供独立 CLI。

### HTTP 合约

- `POST /v1/videos/repairs`：提交。
- `GET /v1/videos`：`after`、`limit`（1–100）、`order` 分页。
- `GET /v1/videos/{task_id}`、`GET /v1/videos/{task_id}/progress`：查询。
- `DELETE /v1/videos/{task_id}`：取消。
- `GET /v1/videos/{task_id}/content`：下载主视频。
- `GET /health`：后端可用性与任务统计；`GET /admin/queue`：状态筛选和全量统计。

默认 `contract="gateway"` 对齐原双服务网关：允许自动生成 taskId、返回
`code/message/task_id/status/variant`，非法请求 HTTP 400、重复 ID HTTP 409、后端不可用 HTTP 503。
`contract="backend"` 对齐原后端：要求 taskId，素材准备成功后返回
`{"code":0,"message":"ok"}`，准备或校验失败返回 code=1 与 reason，并保存失败任务以供查询/回调。
单模型 backend 自动选择其配置模型。两个合约在一个 Server 中选择其一，避免路由冲突。

`VideoRepairRequest` 保留源字段、默认值、显式 camelCase 别名优先级和废弃字段拒绝规则：

```json
{
  "taskId": "edit-001",
  "model": "videoedit-dmd",
  "prompt": "A red jacket",
  "videoUrl": "/data/input.mp4",
  "maskUrl": "/data/masks.npy",
  "referenceImageUrl": "/data/reference.png",
  "callbackUrl": "http://callback-host/results",
  "timeout": 1800
}
```

同一个 repairs 接口也接受 `{"inputs": {...}, "parameters": {...}}`，或
`inputs.request` 包装；parameters 合并到业务请求后再做同样校验，也接受单数 parameter。
通用协议字段转换工具在 `nndeploy.server.request_mapping`。原业务合约应调用 repairs 路由；
普通 `/v1/server/{name}/deploy` 保留公共 Server 的返回模型。

### 生命周期和结果

外部 ID、任务状态、进度、回调状态在公共数据库中保存。接收名额由数据库事务保护，
HTTP 断开不会取消已接收任务；取消或截止时间到达后，等待计算退出再释放名额。
内部 cancelled/expired 在业务接口呈现为 failed / `Request timed out.`，与原服务一致。
重启后可查询历史任务、下载保留的文件、继续未完成的回调投递；中断的推理遵循公共 Server
恢复规则，不把重启视为模型断点续算。

本地输入、HTTP(S)、S3 对象都支持；下载与结果入库按块处理，不把整段视频读成 bytes。
主视频和 `_crop_only` 副产物一起登记到公共文件存储。未指定 output_path 时，返回受管文件
的本地 file_path；远程公共存储返回 content 路径。显式 output_path 保留原路径语义。
设置 minioConfig 或 `add_videoedit(storage=VideoRepairMinioConfig(...))` 可使用源对象存储交付：
保留扩展名替换、COS prefix、output_bucket 和 output_object_key 规则，回调优先返回对象 key。
backend 合约中的远程产物通过返回的 URL 访问，content 返回 HTTP 400。

请求凭据入库前用 Fernet 加密，默认密钥位于 data-dir/request-secrets/request.key，权限 0600。
多副本部署共享数据库、Redis namespace、data-dir（包括 .server-id 与受管文件）和模型挂载，
以便任一副本继续查询与投递回调；用 `secret_key_env` / `--secret-key-env`
配置同一个 Fernet 密钥，并保持模型与服务配置一致。CLI 的 `--server-config` 接收
Server 构造参数 JSON，`--storage-config` 接收 VideoRepairMinioConfig JSON。

进度回调每 5 秒至多一次；终态最多投递 3 次，间隔 1、2 秒。负载保留
status/progress/reason/output，其中 output 为 JSON 字符串。投递失败不会改写推理结果。
任务查询及下载沿用 subject 所有权；开启鉴权时 admin/queue 要求管理员。
可通过 audit_directory / --audit-directory 保存脱敏的请求、实际配置和推理阶段审计；
任务最终成功与否以公共任务表为准（还包括存储交付）。

插帧和超分迁自源 RIFE、Real-ESRGAN 实现，按相邻帧/单帧执行；可配置本地模型路径，
缺省按源逻辑下载权重。视频编码沿用源 VideoEdit 的参考 codec、bitrate、色彩元数据优先规则，
不会用通用 output_quality/output_compression 覆盖参考视频编码。

### 服务验证

`test/e_dit/videoedit/test_videoedit_api.py` 和 `test_videoedit_delivery.py` 覆盖独立进程、
线程 Worker、normal/DMD、忙碌拒绝、重复 ID、取消/超时、重启查询、权限、分页、
进度/终态回调与重试、对象存储交付、流式后处理及小尺寸真实 Transformer/VAE HTTP 推理。
对象存储测试使用替身客户端；插帧/超分流程测试不下载真实权重。
`test_server_protocol.py` 设置 `SGLANG_VIDEOEDIT_SOURCE` 为源 multimodal_gen 目录后，
直接对照源 schema、默认值、解析和回调纯函数。生产权重画质、真实云存储及长视频压测仍需部署环境验收。

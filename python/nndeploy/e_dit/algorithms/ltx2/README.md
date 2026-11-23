# LTX-2

迁移自本地 SGLang `multimodal_gen`，提供 `ltx2` 和 `ltx2_two_stage` 两个算法入口。
两者均支持文本条件、首帧图片条件和联合音视频生成。源码中没有独立的旧版 LTX-Video 生成 Pipeline；
`LTX2I2VPipelineConfig` 已保留，图片条件通过同一算法的 `image_path` 使用。

迁移内容包括 AV Transformer、视频 VAE、音频 VAE、Gemma 配置与文本归一化、Text Connectors、
vocoder、latent upsampler、LTX float32 FlowMatch scheduler，以及单阶段/两阶段生成计算。
模型实现源自 SGLang `multimodal_gen`，保留源码中的许可证声明。

## 使用

安装 `nndeploy[e_dit]` 和可用的原生绑定。模型使用本地 diffusers 布局，根目录需包含
`model_index.json`，组件目录为 `transformer`、`text_encoder`、`tokenizer`、`scheduler`、
`vae`、`audio_vae`、`vocoder`、`connectors`。默认使用 BF16 Transformer/Gemma/connector 和 FP32 音频组件。

```python
from pathlib import Path
from nndeploy.e_dit import DiTParams, build_graph

params = DiTParams(
    model='ltx2',
    model_path='/models/LTX-2',
    device='cuda:0',
    dit_cpu_offload=True,
    text_encoder_cpu_offload=True,
    vae_cpu_offload=True,
)
graph = build_graph(params)
try:
    graph.init()
    video = graph.infer(
        'A river flowing through a forest, with birds singing.',
        seed=10, height=512, width=768, num_frames=121,
        num_inference_steps=40, guidance_scale=4.0, fps=24,
        # image_path='/data/first_frame.png',  # 可选首帧条件
    )
    Path('ltx2.mp4').write_bytes(video)
finally:
    graph.deinit()
```

`infer()` 返回包含音轨的 MP4 bytes。`generate_audio=False` 只关闭音频解码与输出；
模型内部仍保留联合 AV 采样需要的音频 latent。单个 MP4 输出端口只接受单结果。
批量调用 `graph.infer_batch(prompt=[...], ...)` 返回 `OutputBatch`，其中 `output` 为
`[B,F,H,W,C]`、范围 `[0,1]` 的 NumPy 视频，`audio` 为 CPU waveform tensor。
请求参数不修改常驻 scheduler；每个请求及 refinement 阶段使用独立副本。

两阶段入口：

```python
params = DiTParams(
    model='ltx2_two_stage', model_path='/models/LTX-2', device='cuda:0',
    dit_cpu_offload=True, text_encoder_cpu_offload=True, vae_cpu_offload=True,
    component_paths={
        'spatial_upsampler': '/models/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors',
        'distilled_lora': '/models/LTX-2/ltx-2-19b-distilled-lora-384.safetensors',
    },
    # lora_path='/models/style.safetensors', lora_scale=0.8,
)
```

若模型目录包含上述文件或 `latent_upsampler/`，会自动解析两阶段组件路径。
流程为半分辨率生成 → latent 上采样 → distilled LoRA 激活 → 三步 refinement → 解码。
可选基础 LoRA 在两个阶段生效，distilled LoRA 只在第二阶段生效；切回第一阶段不会污染基础权重。
支持成对的 `lora_A/lora_B`、`lora_down/lora_up` safetensors 线性层权重；未知格式、缺失配对和形状不匹配报错。

单阶段空间尺寸需为 VAE 压缩倍数（通常 32）的整数倍；两阶段需为 64 的整数倍。
帧数需满足 `8*n+1`（具体以 checkpoint 的 temporal compression 为准）。图片条件目前要求单结果。

## DAG、工作流与服务

所有计算由 nndeploy 原生 Node/Graph/Loop 执行，完整内部拓扑由 Python 代码构建。
每个去噪步由 `GeneratorStepNode` 执行，支持步间取消和进度回调；中断会关闭计算生成器并释放请求引用。

```python
from nndeploy.base import StatusCode
from nndeploy.e_dit import build_workflow
workflow = build_workflow(
    params, output='ltx2.mp4',
    sampling={'prompt': 'A river at sunrise.', 'seed': 10},
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

服务复用 nndeploy Server，配置入口 `server_config(model='ltx2')` 或 `server_config(model='ltx2_two_stage')`。
输入为 `sampling` JSON，输出为 `video` 文件；首帧条件使用 sampling 中 worker 可访问的本地 `image_path`。

## 支持边界与验证

当前启用单 CPU/CUDA 设备、SDPA、组件级 CPU offload、LoRA 和两阶段生成。
多卡 SP/TP/CFG、量化、编译、融合及自动 warmup 尚未完成 LTX 集成验证，配置时明确拒绝。
迁入的模型保留原始并行计算代码，不意味着本次已接通 SGLang 的分布式运行时。
完整 19B 真实权重与长视频性能尚未验收，微型模型测试不代表生成质量验收。

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m pytest test/e_dit/test_current_api.py -k ltx -q
```

上述测试覆盖 LTX-2 单阶段、两阶段构图和序列化，以及微型 Transformer 的有限输出与重复执行一致性。

`ltx2_workflow.py` 与 `ltx2_two_stage_workflow.py` 是单阶段、两阶段的完整代码示例。
使用前修改模型目录、采样参数和输出路径，然后运行：

```bash
python -m nndeploy.e_dit.algorithms.ltx2.ltx2_workflow
python -m nndeploy.e_dit.algorithms.ltx2.ltx2_two_stage_workflow
```

示例中的 `create_workflow()` 只构图，`main()` 负责初始化、执行和释放资源。

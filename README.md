[English](README_EN.md) | 简体中文

<h3 align="center">
nndeploy：基于工作流的多端AI推理部署框架
</h3>

<p align="center">
<a href="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml/badge.svg" alt="Linux" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml/badge.svg" alt="Windows" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml/badge.svg" alt="Android" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml/badge.svg" alt="macOS" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml/badge.svg" alt="iOS" style="height: 16px;">
</a>
 <a href="https://pepy.tech/projects/nndeploy">
  <img src="https://static.pepy.tech/personalized-badge/nndeploy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" style="height: 16px;">
</a>
</p>

<p align="center">
<a href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"><b>文档</b></a> 
| <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>微信交流群</b></a> 
<!-- | <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>知乎</b></a>  -->
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
<!-- | <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>哔哩哔哩</b></a>  -->
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow.gif" width=100%>
  </picture>
</p>

---

## 最新动态

- [2025/05/29]🔥nndeploy 开源团队和昇腾官方合作的推理框架免费课程上线拉，适合想入门和提升 AI 推理部署能力的同学。[昇腾平台学习](https://www.hiascend.com/developer/courses/detail/1923211251905150977) | [B 站学习](https://space.bilibili.com/435543077?spm_id_from=333.788.0.0)

---

## 介绍

nndeploy 是一款基于工作流的多端 AI 推理部署框架，致力于为用户提供简单易用、高性能的 AI 落地体验。基于 nndeploy，开发者可以轻松地从训练算法仓库开发出指定平台和硬件所需的 SDK，大幅节省开发时间。此外，框架已部署包括大语言模型（LLM）、AIGC 生成、换脸、目标检测、图像分割等众多主流 AI 模型，让您开箱即用。

### **简单易用**

- **可视化工作流**：通过拖拉拽操作就能部署 AI 算法，前端可视化调节 AI 算法的所有节点参数，快速预览算法调参后的效果
- **自定义节点**：支持 Python/C++自定义节点，无需前端代码，无缝集成到可视化界面
- **算法组合**：灵活组合不同算法，快速构建创新 AI 应用
- **一键部署**：搭建完成的工作流可一键导出为 JSON 配置文件，支持 Python/C++ API 直接调用，实现从开发环境到生产环境的无缝衔接，全面支持 Linux、Windows、macOS、Android、iOS 等平台。

### **高性能**

- **13 种推理引擎无缝集成**：一套工作流，多端部署。通过零抽象成本接入了 13 种主流推理框架，覆盖云端、桌面、移动、边缘等全平台

  | 推理框架                                                                         | 适用场景                 | 状态 |
  | :------------------------------------------------------------------------------- | :----------------------- | :--- |
  | [PyTorch](https://pytorch.org/)                                                  | 研发调试、快速原型       | ✅   |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime)                          | 跨平台推理               | ✅   |
  | [TensorRT](https://github.com/NVIDIA/TensorRT)                                   | NVIDIA GPU 高性能推理    | ✅   |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino)                          | Intel CPU/GPU 优化       | ✅   |
  | [MNN](https://github.com/alibaba/MNN)                                            | 阿里推出的移动端推理引擎 | ✅   |
  | [TNN](https://github.com/Tencent/TNN)                                            | 腾讯推出的移动端推理引擎 | ✅   |
  | [ncnn](https://github.com/Tencent/ncnn)                                          | 腾讯推出的移动端推理引擎 | ✅   |
  | [CoreML](https://github.com/apple/coremltools)                                   | iOS/macOS 原生加速       | ✅   |
  | [AscendCL](https://www.hiascend.com/zh/)                                         | 华为昇腾 AI 芯片推理框架 | ✅   |
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) | 瑞芯微 NPU 推理框架      | ✅   |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)   | 高通骁龙 NPU 推理框架    | ✅   |
  | [TVM](https://github.com/apache/tvm)                                             | 深度学习编译栈           | ✅   |
  | [自研推理框架](docs/zh_cn/inference/README_INFERENCE.md)                         | 定制化推理需求           | ✅   |

- **并行优化**：支持串行、流水线并行、任务并行等执行模式
- **内存优化**：零拷贝、内存池、内存复用等优化策略
- **高性能优化**：内置 C++/CUDA/Ascend C/SIMD 等优化实现的节点

### **开箱即用的算法**

已部署模型列表，并制作 100+节点，我们将持续部署更多高价值的 AI 算法，打造丰富的多端 AI 生态，满足各类行业与创新应用需求。如果你有需要部署的算法，请通过[issue](https://github.com/nndeploy/nndeploy/issues)告诉我们

| 应用场景       | 可用模型                                                                                                                                                                                                     | 支持平台                        | 备注                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------- |
| **大语言模型** | **QWen-0.5B**                                                                                                                                                                                                | Linux/Windows/macOS/Android/iOS |                                                     |
| **图片生成**   | Stable Diffusion 1.5, Stable Diffusion XL, Stable Diffusion 3, HunyuanDiT, Kandinsky, Wuerstchen, Stable Cascade, PixArt Alpha, PixArt Sigma, Sana, AuraFlow, Flux, Lumina, CogView3 Plus, CogView4 等等模型 | Linux/Windows/macOS             | 支持文生图、图生图、图像修复，基于**diffusers**实现 |
| **换脸**       | **deep-live-cam**                                                                                                                                                                                            | Linux/Windows/macOS             |                                                     |
| **目标检测**   | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx**                                                                                                                                                           | Linux/Windows/macOS/Android/iOS |                                                     |
| **目标追踪**   | FairMot                                                                                                                                                                                                      | Linux/Windows/macOS/Android/iOS |                                                     |
| **图像分割**   | RBMGv1.4, PPMatting, **Segment Anything**                                                                                                                                                                    | Linux/Windows/macOS/Android/iOS |                                                     |
| **分类**       | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet                                                                                                                                   | Linux/Windows/macOS/Android/iOS |                                                     |
| **API LLM**    | OPENAI, DeepSeek, Moonshot                                                                                                                                                                                   | Linux/Windows/macOS             |                                                     |
| **API AIGC**   | OPENAI                                                                                                                                                                                                       | Linux/Windows/macOS             |                                                     |

## 性能加速

测试环境：Ubuntu 22.04，CPU：12th Gen Intel(R) Core(TM) i7-12700，GPU：RTX3060；为确保测试准确性，移除了非必要的图像编码对全局性能的影响。

### 流水线并行加速

YOLOv11s端到端总耗时，串行与流水线并行耗时对比

![yolov11s_performance](docs/image/workflow/yolo_performance.png)

| 运行方式\推理引擎 | ONNXRuntime | OpenVINO  | TensorRT  |
| ----------------- | ----------- | --------- | --------- |
| 串行执行          | 112.298 ms  | 87.459 ms | 67.642 ms |
| 流水线并行        | 54.05 ms    | 45.83 ms  | 36.76 ms  |
| 性能提升          | **51.9%**   | **47.6%** | **45.65%** |

### 任务并行加速

多个组合任务(分割RMBGv1.4+检测YOLOv11s+分类ResNet50)的端到端总耗时，串行与任务并行耗时对比

![rmbg_yolo_resnet.png](docs/image/workflow/rmbg_yolo_resnet.png)

| 运行方式\推理引擎 | ONNXRuntime | OpenVINO   | TensorRT  |
| ----------------- | ----------- | ---------- | --------- |
| 串行执行          | 652.561 ms  | 708.008 ms | 86.813 ms |
| 任务级并行        | 608.684 ms  | 459.870 ms | 72.258 ms |
| 性能提升          | **6.7%**    | **35.0%**  | **16.8%** |

## 快速开始

### 安装

```bash
pip install --upgrade nndeploy
```

### 启动可视化工作流

```bash
# 方法一：仅使用内置节点
nndeploy-app --port 8000

# 方法二：使用用户自定义节点
nndeploy-app --port 8000 --plugin plugin1.py plugin2.py
```

- 命令参数说明
  - `--port`：指定 Web 服务端口号（默认为 8000）
  - `--plugin`：加载用户自定义插件文件（可选参数，如果没有该参数，仅使用内置节点）
    - Python 插件：参考[Python 插件模板写法](template/python/template.py)
    - C++插件：参考[C++插件模板写法](template/cpp/template.h)
    - 可以同时加载多个插件：`--plugin plugin1.py plugin2.so`

启动成功后，打开 http://localhost:8000 即可访问工作流界面。

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="quick_start.gif">
    <img alt="nndeploy" src="docs/image/quick_start.gif" width=100%>
  </picture>
</p>

### 导出工作流并执行

在可视化界面中完成工作流的搭建后，可将其保存为 JSON 文件（例如 workflow.json），然后可以使用以下命令执行该工作流：

```bash
nndeploy-run-json --json-file workflow.json --plugin plugin.py
```

- API 加载运行 JSON 配置文件
  - [Python API 示例代码](python/nndeploy/dag/run_json.py)
  - [C++ API 示例代码](framework/include/nndeploy/dag/graph_runner.h)

> 需要 Python 3.10 及以上版本。默认包含 PyTorch 和 ONNXRuntime 两个推理后端，如需使用更多推理后端（如 TensorRT、OpenVINO、ncnn、MNN 等），请采用开发者模式

> 使用`nndeploy-clean`可清理过期的后端资源。

### 文档

- [如何构建](docs/zh_cn/quick_start/build.md)
- [如何获取模型](docs/zh_cn/quick_start/model.md)
- [如何执行](docs/zh_cn/quick_start/example.md)
- [Python 快速开始](docs/zh_cn/quick_start/python.md)
- [可视化工作流快速开始](docs/zh_cn/quick_start/workflow.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [C++插件开发手册](docs/zh_cn/quick_start/plugin.md)
- [Python++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)
- [Python 插件开发手册](docs/zh_cn/quick_start/plugin_python.md)

## 保持领先

在 GitHub 上给 nndeploy Star，并立即收到新版本的通知。

<img src="docs/image/star.gif">

## 下一步计划

- [工作流生态](https://github.com/nndeploy/nndeploy/issues/191)
- [端侧大模型推理](https://github.com/nndeploy/nndeploy/issues/161)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)
- [架构优化](https://github.com/nndeploy/nndeploy/issues/189)

## 联系我们

- 当前 nndeploy 正处于发展阶段，如果您热爱开源、喜欢折腾，不论是出于学习目的，抑或是有更好的想法，欢迎加入我们。

- 微信：Always031856（欢迎加好友，进活跃的 AI 推理部署交流群，备注：nndeploy\_姓名）

  <img src="docs/image/wechat.jpg" width="225px">

## 致谢

- 感谢以下项目：[TNN](https://github.com/Tencent/TNN)、[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)、[opencv](https://github.com/opencv/opencv)、[CGraph](https://github.com/ChunelFeng/CGraph)、[CThreadPool](https://github.com/ChunelFeng/CThreadPool)、[tvm](https://github.com/apache/tvm)、[mmdeploy](https://github.com/open-mmlab/mmdeploy)、[FlyCV](https://github.com/PaddlePaddle/FlyCV)、[oneflow](https://github.com/Oneflow-Inc/oneflow)、[flowgram.ai](https://github.com/bytedance/flowgram.ai)、[deep-live-cam](https://github.com/hacksider/Deep-Live-Cam)。

- 感谢[HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy)推荐

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## 贡献者

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)

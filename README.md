
[English](README_EN.md) | 简体中文

<h3 align="center">
nndeploy：你本地的AI工作流
</h3>

<p align="center">
<a href="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml/badge.svg" alt="Linux">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml/badge.svg" alt="Windows">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml/badge.svg" alt="Android">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml/badge.svg" alt="macOS">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml/badge.svg" alt="iOS">
</a>
</p>

<p align="center">
<a href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"><b>文档</b></a> 
| <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>公众号</b></a> 
| <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>知乎</b></a> 
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
| <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>哔哩哔哩</b></a> 
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow.png" width=100%>
  </picture>
</p>

你只需使用的Python/C++编写算法逻辑，无需掌握任何前端技术，就能快速构建你的可视化AI工作流。

支持将搭建的工作流一键导出为JSON文件，并提供Python/C++ API接口来调用该JSON文件，让你轻松将AI应用部署到云服务器、桌面应用、移动设备、边缘计算设备等各种生产环境中。

框架内置了多种业界主流的高性能推理引擎、丰富的节点和深度优化策略，帮助程序员将AI算法创意快速转化为高性能的企业级生产应用。

---

<!-- ## 最新动态
- [2025/08/01] 🔥 **可视化工作流**：告别复杂的代码编写！通过简单的拖拽操作，你就能快速搭建专业的AI应用。无论你是技术小白还是经验丰富的开发者，都能在几分钟内完成AI算法的部署。[立即体验丰富的工作流模板：换脸、LLM对话、AI绘画、目标检测、图像分割等应用](https://github.com/nndeploy/nndeploy-workflow)
- [2025/07/20] 🔥 **Python API**：只需几行代码，就能将你的AI模型部署到手机、电脑、服务器等各种设备上。更棒的是，还支持工作流可视化展示，让你在团队汇报时轻松展示令人惊艳的AI效果，瞬间成为焦点！[点击查看超简单的入门教程，5分钟上手](https://nndeploy-zh.readthedocs.io/zh-cn/latest/quick_start/python.html)
- [2025/05/29] 🔥 **免费AI推理课程**：想要在AI推理部署领域找到更好的工作机会？我们基于nndeploy框架打造了业内最全面的AI推理部署课程，深度覆盖模型中间表示、模型解释、计算图构建、图优化、内存优化、并行优化和算子优化等企业核心技术需求。这门课程都将成为你职业发展的强力助推器。[昇腾平台免费学习](https://www.hiascend.com/developer/courses/detail/1923211251905150977) | [B站同步更新](https://space.bilibili.com/435543077?spm_id_from=333.788.0.0) -->

<!-- --- -->

## 快速开始

### 安装

```bash
pip install --upgrade nndeploy
```

### 启动可视化工作流界面

```bash
# 方法一：仅使用内置节点
nndeploy-app --port 8000

# 方法二：使用用户自定义节点
nndeploy-app --port 8000 --plugin plugin1.py plugin2.py 
```

#### 命令参数说明
- `--port`：指定Web服务端口号（默认为8000）
- `--plugin`：加载用户自定义插件文件（可选参数，如果没有该参数，仅使用内置节点）
  - Python插件：参考[Python插件模板写法](template/python/template.py)
  - C++插件：参考[C++插件模板写法](template/cpp/template.h)
  - 可以同时加载多个插件：`--plugin plugin1.py plugin2.so`

启动成功后，打开 http://localhost:8000 即可访问工作流界面。

### 保存工作流为JSON和执行工作流

在可视化界面中配置好工作流后，可将其保存为JSON文件（例如workflow.json）。您可以使用以下命令执行该工作流：

```bash
nndeploy-run-json --json-file workflow.json --plugin plugin.py
```

> 需要 Python 3.10 及以上版本。默认包含 PyTorch 和 ONNXRuntime 两个推理后端。如需使用更多推理后端（如 TensorRT、OpenVINO、ncnn、MNN 等），请采用开发者模式

> 使用`nndeploy-clean`可清理过期的后端资源。

### 文档
- [如何构建](docs/zh_cn/quick_start/build.md)
- [如何获取模型](docs/zh_cn/quick_start/model.md)
- [如何执行](docs/zh_cn/quick_start/example.md)
- [Python快速开始](docs/zh_cn/quick_start/python.md)
- [可视化工作流快速开始](docs/zh_cn/quick_start/workflow.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [C++插件开发手册](docs/zh_cn/quick_start/plugin.md)
- [Python++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)

## 核心特性

### **告别复杂开发，专注算法创新**
- **可视化开发**：告别复杂的代码编写！无论你是技术小白还是经验丰富的开发者，通过拖拉拽操作，你就能快速搭建专业的AI工作流
- **代码即工作流节点**：你只需使用熟悉的Python/C++编写算法逻辑，无需掌握任何前端技术，框架自动将代码转化为工作流中节点
- **一键部署**：搭建好的工作流可导出为JSON，Python/C++直接调用，从开发到生产环境无缝衔接

### **快速迭代，实时调试**
- **积木式算法组合**：像搭乐高一样组合AI模型，快速验证创新想法
- **热更新参数调试**：前端实时调参，后端立即响应，调试效率提升10倍
- **可视化性能监控**：实时查看每个节点的执行时间

### **生产级性能**
- **13种推理引擎无缝集成**：一套工作流，多端部署。通过零抽象成本接入了13种主流推理框架，覆盖云端、桌面、移动、边缘等全平台

  | 推理框架 | 适用场景 | 状态 |
  | :------- | :------ | :--- |
  | [PyTorch](https://pytorch.org/) | 研发调试、快速原型 | ✅ |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime) | 跨平台推理 | ✅ |
  | [TensorRT](https://github.com/NVIDIA/TensorRT) | NVIDIA GPU高性能推理 | ✅ |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino) | Intel CPU/GPU优化 | ✅ |
  | [MNN](https://github.com/alibaba/MNN) | 移动端轻量化部署 | ✅ |
  | [TNN](https://github.com/Tencent/TNN) | 腾讯高性能推理引擎 | ✅ |
  | [ncnn](https://github.com/Tencent/ncnn) | ARM设备高效推理 | ✅ |
  | [CoreML](https://github.com/apple/coremltools) | iOS/macOS原生加速 | ✅ |
  | [AscendCL](https://www.hiascend.com/zh/) | 华为昇腾AI芯片 | ✅ |
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) | 瑞芯微NPU加速 | ✅ |
  | [TVM](https://github.com/apache/tvm) | 深度学习编译栈 | ✅ |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | 高通骁龙NPU | ✅ |
  | [自研推理框架](docs/zh_cn/inference/README_INFERENCE.md) | 定制化推理需求 | ✅ |

- **并行加速**：支持串行、流水线并行、任务并行等执行模式，性能提升无需修改代码
- **内存优化**：零拷贝、内存池、内存复用等优化策略
- **CUDA/SIMD优化**：内置高性能节点，无需手动调优

## 开箱即用的AI算法

已经部署了以下AI算法，并制作了[工作流模板](https://github.com/nndeploy/nndeploy-workflow)，让你能够立即体验和使用各种AI功能：

| 应用场景 | 可用模型 | 
|---------|---------|
| **图像分类** | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet |
| **目标检测** | YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx | 
| **目标追踪** | FairMot | 
| **图像分割** | RBMGv1.4, PPMatting, **Segment Anything** |
| **生成模型** | Stable Diffusion 1.5 | 
| **大语言模型** | QWen-0.5B | 
| **换脸应用** | deep-live-cam | 

> 会持续部署更多AI算法，如果你有需要的算法，请通过[issue](https://github.com/nndeploy/nndeploy/issues)告诉我们

## 保持领先

在 GitHub 上给 nndeploy Star，并立即收到新版本的通知。

<img src="docs/image/star.gif">

## 下一步计划

- [工作流生态](https://github.com/nndeploy/nndeploy/issues/191)
- [端侧大模型推理](https://github.com/nndeploy/nndeploy/issues/161)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)
- [架构优化](https://github.com/nndeploy/nndeploy/issues/189)

## 联系我们
- **加入开发者社区**：与工程师一起交流技术、获取支持、抢先体验新功能！微信：Always031856（请备注：名称 + 技术方向）
  
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

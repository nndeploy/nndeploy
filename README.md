
[English](README_EN.md) | 简体中文

<h3 align="center">
基于工作流的多端AI部署工具
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

nndeploy是一款基于工作流的多端AI部署工具，可以将你创意的算法想法快速、高性能的完成部署。无论你是AI算法工程师、产品经理还是开发者，nndeploy都能帮助你快速、高效地将AI模型部署到各种设备和平台上。

---

## 最新动态
- [2025/08/01] 🔥 **可视化工作流**：告别复杂的代码编写！通过简单的拖拽操作，你就能快速搭建专业的AI应用。无论你是技术小白还是经验丰富的开发者，都能在几分钟内完成AI算法的部署。[立即体验丰富的工作流模板：换脸、LLM对话、AI绘画、目标检测、图像分割等应用](https://github.com/nndeploy/nndeploy-workflow)
- [2025/07/20] 🔥 **Python API**：只需几行代码，就能将你的AI模型部署到手机、电脑、服务器等各种设备上。更棒的是，还支持工作流可视化展示，让你在团队汇报时轻松展示令人惊艳的AI效果，瞬间成为焦点！[点击查看超简单的入门教程，5分钟上手](https://nndeploy-zh.readthedocs.io/zh-cn/latest/quick_start/python.html)
- [2025/05/29] 🔥 **免费AI推理课程**：想要在AI推理部署领域找到更好的工作机会？我们基于nndeploy框架打造了业内最全面的AI推理部署课程，深度覆盖模型中间表示、模型解释、计算图构建、图优化、内存优化、并行优化和算子优化等企业核心技术需求。这门课程都将成为你职业发展的强力助推器。[昇腾平台免费学习](https://www.hiascend.com/developer/courses/detail/1923211251905150977) | [B站同步更新](https://space.bilibili.com/435543077?spm_id_from=333.788.0.0)

---

## 开箱即用的AI算法

已经部署了以下AI算法，让你能够立即体验和使用各种AI功能：

| 应用场景 | 可用模型 | 
|---------|---------|
| **图像分类** | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet |
| **目标检测** | YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx | 
| **目标追踪** | FairMot | 
| **图像分割** | RBMGv1.4, PPMatting, **Segment Anything** |
| **生成模型** | Stable Diffusion 1.5 | 
| **大语言模型** | QWen-0.5B | 
| **换脸应用** | deep-live-cam | 

> [查看完整模型列表和使用说明](docs/zh_cn/quick_start/model_list.md) 

> 会持续部署更多AI算法，如果你有需要的算法，请通过[issue](https://github.com/nndeploy/nndeploy/issues)告诉我们

## 介绍

nndeploy是一款基于工作流的多端AI部署工具，具有以下功能：

### 1. AI部署的效率工具

- **可视化工作流**：通过拖拉拽部署AI算法，突出开发效率

- **函数调用**：工作流导出为JSON配置文件，支持Python/C++ API调用，在多端的生产环境中使用起来

- **多端推理**：一套工作流，多端部署。通过零抽象成本接入了13种主流推理框架，覆盖云端、桌面、移动、边缘等全平台

  | 框架 | 支持状态 |
  | :------- | :------ |
  | [PyTorch](https://pytorch.org/) | ✅ |
  | [TensorRT](https://github.com/NVIDIA/TensorRT) | ✅ |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino) | ✅ |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime) | ✅ |
  | [MNN](https://github.com/alibaba/MNN) | ✅ |
  | [TNN](https://github.com/Tencent/TNN) | ✅ |
  | [ncnn](https://github.com/Tencent/ncnn) | ✅ |
  | [CoreML](https://github.com/apple/coremltools) | ✅ |
  | [AscendCL](https://www.hiascend.com/zh/) | ✅ |
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) | ✅ |
  | [TVM](https://github.com/apache/tvm) | ✅ |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | ✅ |
  | [自研推理框架](docs/zh_cn/inference/README_INFERENCE.md) | ✅ |

### 2. AI部署的性能工具

- **并行优化**：支持串行、流水线并行、任务并行等执行模式

- **内存优化**：零拷贝、内存池、内存复用等优化策略
  
- **高性能优化**：内置C++/CUDA/SIMD等优化实现的节点

### 3. AI部署的创意工具

- **自定义节点**：支持Python/C++自定义节点，无需前端代码，无缝集成到可视化界面

- **算法组合**：灵活组合不同算法，快速构建创新AI应用

- **所调即所见**：前端可视化调节AI算法部署的所有节点参数，快速预览算法调参后的效果

  <img src="docs/image/workflow/face_swap_segment.png">

  <img src="docs/image/workflow/qwen_sd.png">

  <img src="docs/image/workflow/sd_yolo.png">


## 快速开始

### 启动可视化工作流界面

安装nndeploy包并启动可视化工作流工具

```bash
# 通过pip安装nndeploy
pip install nndeploy
# 在8000端口启动可视化工作流应用
nndeploy-app --port 8000
```

打开 http://localhost:8000 即可访问工作流界面。

### 通过JSON保存和执行工作流

在可视化界面中配置好工作流后，将其保存为JSON文件（例如yolo.json）。您可以使用以下命令执行该工作流：

```bash
# 执行JSON文件中定义的工作流
# -i：指定输入文件（例如input.jpg）
# -o：指定输出文件（例如output.jpg）
nndeploy-run-json --json-file yolo.json -i input.jpg -o output.jpg
```

> 需要Python 3.10或更高版本。使用`nndeploy-clean`可清理过期的后端资源。

> 由于pypi包体机限制，目前的python包包含torch和onnxruntime两个推理后端，想使用更丰富的后端请采用开发者模式

### 文档
- [如何构建](docs/zh_cn/quick_start/build.md)
- [如何获取模型](docs/zh_cn/quick_start/model.md)
- [如何执行](docs/zh_cn/quick_start/example.md)
- [Python快速开始](docs/zh_cn/quick_start/python.md)
- [可视化工作流快速开始](docs/zh_cn/quick_start/workflow.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [Python++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)


## 下一步计划

- [工作流生态](https://github.com/nndeploy/nndeploy/issues/191)
- [端侧大模型推理](https://github.com/nndeploy/nndeploy/issues/161)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)
- [架构优化](https://github.com/nndeploy/nndeploy/issues/189)

## 联系我们
- 欢迎加入交流群！微信：titian5566（请简单备注个人信息^_^）
  
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

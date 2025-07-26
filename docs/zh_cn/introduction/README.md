
[English](README_EN.md) | 简体中文

<h1 align="center">
nndeploy：一款基于工作流的多端AI部署工具
</h1>

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
| <a href="../../zh_cn/knowledge_shared/wechat.md"><b>公众号</b></a> 
| <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>知乎</b></a> 
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
| <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>哔哩哔哩</b></a> 
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
</p>


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../image/workflow.png">
    <img alt="nndeploy" src="../../image/workflow.png" width=100%>
  </picture>
</p>

---

## 最新动态
- [2025/07/20] 🔥 **可视化工作流**：通过拖拉拽部署AI算法
- [2025/07/20] 🔥 **Python API**：快速入门，便捷开发 ([文档](https://nndeploy-zh.readthedocs.io/zh-cn/latest/quick_start/python.html))
- [2025/05/29] 🔥 **与华为昇腾合作推理框架课程**：官方认证，专业指导 ([链接](https://www.hiascend.com/developer/courses/detail/1923211251905150977))

---

## 已部署模型

| 模型类别 | 支持模型 |
|---------|---------|
| **图像分类** | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet |
| **目标检测** | YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx |
| **目标追踪** | FairMot |
| **图像分割** | RBMGv1.4, PPMatting |
| **生成模型** | Stable Diffusion 1.5 |
| **大语言模型** | QWen-0.5B |
| **换脸** | deep-live-cam |

> [已部署模型详情列表](../../zh_cn/quick_start/model_list.md)

## 介绍

nndeploy是一款基于工作流的多端AI部署工具，具有以下功能：

### 1. AI部署的效率工具

- **可视化工作流**：通过拖拉拽部署AI算法

- **函数调用**：工作流导出为JSON配置文件，支持Python/C++ API调用

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
  | [自研推理框架](../../zh_cn/inference/README_INFERENCE.md) | ✅ |

### 2. AI部署的性能工具

- **并行优化**：支持串行、流水线并行、任务并行等执行模式

- **内存优化**：零拷贝、内存池、内存复用等优化策略
  
- **高性能优化**：内置C++/CUDA/SIMD等优化实现的节点

### 3. AI部署的创意工具

- **自定义节点**：支持Python/C++自定义节点，无需前端代码，无缝集成到可视化界面

- **算法组合**：灵活组合不同算法，快速构建创新AI应用

- **所调即所见**：前端可视化调节AI算法部署的所有节点参数，快速预览算法调参后的效果

  <img src="../../image/workflow/face_swap_segment.png">

  <img src="../../image/workflow/qwen_sd.png">

  <img src="../../image/workflow/sd_yolo.png">

## 快速开始

- [如何编译](../../zh_cn/quick_start/build.md)
- [如何获取模型](../../zh_cn/quick_start/model.md)
- [如何执行](../../zh_cn/quick_start/example.md)
- [nndeploy C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [nndeploy Python 快速入门](../../zh_cn/quick_start/python.md)
- [nndeploy Python++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)

## 下一步计划

- [工作流生态](https://github.com/nndeploy/nndeploy/issues/191)
- [端侧大模型推理](https://github.com/nndeploy/nndeploy/issues/161)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)
- [架构优化](https://github.com/nndeploy/nndeploy/issues/189)

## 联系我们
- 欢迎加入交流群！微信：titian5566（请简单备注个人信息^_^）
  
  <img src="../../image/wechat.jpg" width="225px">

## 致谢

- 感谢以下项目：[TNN](https://github.com/Tencent/TNN)、[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)、[opencv](https://github.com/opencv/opencv)、[CGraph](https://github.com/ChunelFeng/CGraph)、[CThreadPool](https://github.com/ChunelFeng/CThreadPool)、[tvm](https://github.com/apache/tvm)、[mmdeploy](https://github.com/open-mmlab/mmdeploy)、[FlyCV](https://github.com/PaddlePaddle/FlyCV)、[oneflow](https://github.com/Oneflow-Inc/oneflow)、[flowgram.ai](https://github.com/bytedance/flowgram.ai)。

- 感谢[HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy)推荐

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## 贡献者

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)


[English](README_EN.md) | 简体中文

## 介绍

nndeploy是一款端到端的模型推理和部署框架。它旨在为用户提供功能强大、简单易用、高性能且兼容主流框架的模型推理和部署体验。

## 架构

![Architecture](docs/image/architecture.jpg)

## 特性

### 1. 开箱即用的算法

目前已完成 [YOLOV5](https://github.com/ultralytics/yolov5)、[YOLOV6](https://github.com/meituan/YOLOv6)、[YOLOV8](https://github.com/ultralytics) 等模型的部署，可供您直接使用，后续我们持续不断去部署其它开源模型，让您开箱即用

| model                                                       | Inference                         | developer                                                                                            | remarks |
| :---------------------------------------------------------- | :-------------------------------- | :--------------------------------------------------------------------------------------------------- | :-----: |
| [YOLOV5](https://github.com/ultralytics/yolov5)             | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss)                 |         |
| [YOLOV6](https://github.com/meituan/YOLOv6)                 | TensorRt/OpenVINO/ONNXRuntime     | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss)                 |         |
| [YOLOV8](https://github.com/ultralytics)                    | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss)                 |         |
| [SAM](https://github.com/facebookresearch/segment-anything) | ONNXRuntime                       | [youxiudeshouyeren](https://github.com/youxiudeshouyeren)、[Always](https://github.com/Alwaysssssss) |         |

### 2. 支持跨平台和多推理框架

**一套代码多端部署**：通过切换推理配置，一套代码即可完成模型`跨多个平台以及多个推理框架`部署

当前支持的推理框架如下：

| Inference/OS                                                                     | Linux | Windows | Android | MacOS |  IOS  | developer                                                                          | remarks |
| :------------------------------------------------------------------------------- | :---: | :-----: | :-----: | :---: | :---: | :--------------------------------------------------------------------------------- | :-----: |
| [TensorRT](https://github.com/NVIDIA/TensorRT)                                   |   √   |    -    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          |         |
| [OpenVINO](https://github.com/openvinotoolkit/openvino)                          |   √   |    √    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          |         |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime)                          |   √   |    √    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          |         |
| [MNN](https://github.com/alibaba/MNN)                                            |   √   |    √    |    √    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          |         |
| [TNN](https://github.com/Tencent/TNN)                                            |   √   |    √    |    √    |   -   |   -   | [02200059Z](https://github.com/02200059Z)                                          |         |
| [ncnn](https://github.com/Tencent/ncnn)                                          |   -   |    -    |    √    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          |         |
| [coreML](https://github.com/apple/coremltools)                                   |   -   |    -    |    -    |   √   |   -   | [JoDio-zd](https://github.com/JoDio-zd)、[jaywlinux](https://github.com/jaywlinux) |         |
| [AscendCL](https://www.hiascend.com/zh/)                                         |   √   |    -    |    -    |   -   |   -   | [CYYAI](https://github.com/CYYAI)                                                  |         |
| [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) |   √   |    -    |    -    |   -   |   -   | [100312dog](https://github.com/100312dog)                                          |         |
| **[default](https://github.com/nndeploy/nndeploy)**                              |   √   |    -    |    -    |   -   |   -   | [nndeploy](https://github.com/nndeploy)                                            |**nndeploy内部推理子模块**         |

**Notice:** 
- TFLite, TVM, OpenPPL, sophgo, Horizon正在开发中，我们正在努力覆盖绝大部分的主流推理框架
- **default: nndeploy内部开发的推理子模块，已支持分类(resnet50)、检测(YOLOV11)、分割(RMBG1.4)等模型。[README_INFERENCE.md](README_INFERENCE.md)**

### 3. 简单易用

- **基于有向无环图部署模型**： 将 AI 算法端到端（前处理->推理->后处理）的部署抽象为有向无环图 `Graph`，前处理为一个 `Node`，推理也为一个 `Node`，后处理也为一个 `Node`
 
- **推理模板Infer**： 基于`多端推理模块Inference` + `有向无环图节点Node`再设计功能强大的`推理模板Infer`，Infer推理模板可以帮您在内部处理不同的模型带来差异，例如**单输入、多输入、单输出、多输出、静态形状输入、动态形状输入、静态形状输出、动态形状输出**一系列不同
 
- **高效解决多模型的复杂场景**：在多模型组合共同完成一个任务的复杂场景下（例如老照片修复），每个模型都可以是独立的Graph，nndeploy的有向无环图支持`图中嵌入图`灵活且强大的功能，将大问题拆分为小问题，通过组合的方式快速解决多模型的复杂场景问题

- **快速构建demo**：对于已部署好的模型，需要编写demo展示效果，而demo需要处理多种格式的输入，例如图片输入输出、文件夹中多张图片的输入输出、视频的输入输出等，通过将上述编解码节点化，可以更通用以及更高效的完成demo的编写，达到快速展示效果的目的（目前主要实现了基于OpneCV的编解码节点化）

### 4. 高性能

- **推理框架的高性能抽象**：每个推理框架也都有其各自的特性，需要足够尊重以及理解这些推理框架，才能在抽象中不丢失推理框架的特性，并做到统一的使用的体验。`nndeploy` 可配置第三方推理框架绝大部分参数，保证了推理性能。可直接操作推理框架内部分配的输入输出，实现前后处理的零拷贝，提升模型部署端到端的性能。

- **线程池**：提高模型部署的并发性能和资源利用率（thread pool）。此外，还支持CPU端算子自动并行，可提升CPU算子执行性能（parallel_for）。
  
- **内存池**：完成后可实现高效的内存分配与释放(TODO)
  
- **一组高性能的算子**：完成后将加速您模型前后处理速度(TODO)

### 5. 并行

- **串行**：按照模型部署的有向无环图的拓扑排序，依次执行每个节点。

- **流水线并行**：在处理多帧的场景下，基于有向无环图的模型部署方式，可将前处理 `Node`、推理 `Node`、后处理 `Node`绑定三个不同的线程，每个线程又可绑定不同的硬件设备下，从而三个`Node`可流水线并行处理。在多模型以及多硬件设备的的复杂场景下，更加可以发挥流水线并行的优势，从而可显著提高整体吞吐量。

- **任务并行**：在多模型以及多硬件设备的的复杂场景下，基于有向无环图的模型部署方式，可充分挖掘模型部署中的并行性，缩短单次算法全流程运行耗时

- **上述模式的组合并行**：在多模型、多硬件设备以及处理多帧的复杂场景下，nndeploy的有向无环图支持图中嵌入图的功能，每个图都可以有独立的并行模式，故用户可以任意组合模型部署任务的并行模式，可充分发挥硬件性能。

## 资源仓库

- 我们已将第三方库、模型仓库和测试数据上传至[HuggingFace](https://huggingface.co/alwaysssss/nndeploy)上，如有需要，欢迎您前往下载使用。

## 文档
- 如何编译：https://nndeploy-zh.readthedocs.io/zh/latest/quick_start/build.html
- 如何执行：https://nndeploy-zh.readthedocs.io/zh/latest/quick_start/example.html
- 如何获取模型：https://nndeploy-zh.readthedocs.io/zh/latest/quick_start/model.html
- 更多信息访问[nndeploy文档](https://nndeploy-zh.readthedocs.io/zh/latest/)

## 参考
- [TNN](https://github.com/Tencent/TNN)
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)
- [opencv](https://github.com/opencv/opencv)
- [CGraph](https://github.com/ChunelFeng/CGraph)
- [CThreadPool](https://github.com/ChunelFeng/CThreadPool)
- [tvm](https://github.com/apache/tvm)
- [mmdeploy](https://github.com/open-mmlab/mmdeploy)
- [FlyCV](https://github.com/PaddlePaddle/FlyCV)
- [oneflow](https://github.com/Oneflow-Inc/oneflow)


## 联系我们
- 当前nndeploy正处于发展阶段，如果您热爱开源、喜欢折腾，不论是出于学习目的，抑或是有更好的想法，欢迎加入我们。
- 微信：titian5566 (加微信进AI推理部署交流群，请简单备注个人信息)

  <img align="left" src="docs/image/wechat.jpg" width="225px">
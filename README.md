
[English](README_EN.md) | 简体中文

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/logo.png">
    <img alt="nndeploy" src="docs/image/logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
简单易用、高性能、支持多端的AI推理部署框架
</h3>

<p align="center">
| <a href="https://nndeploy-zh.readthedocs.io/zh/latest/"><b>文档</b></a> | <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>公众号</b></a> | <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>知乎</b></a> | <a href="https://discord.gg/xAWvmZn3"><b>discord</b></a> |
</p>

---

## 快速开始

- [如何编译](docs/zh_cn/quick_start/build.md)
- [如何获取模型](docs/zh_cn/quick_start/model.md)
- [如何执行](docs/zh_cn/quick_start/example.md)

## 已部署的模型

<table>
  <tr>
    <td><b>文生图 (Stable Diffusion 1.5)</b></td>
    <td><b>大语言模型 (QWen)</b></td>
    <td><b>图像分割 (RBMGv1.4)</b></td>
    <td><b>更多模型</b></td>
  </tr>
  <tr>
    <td><img src="docs/image/demo/stable_diffusion/apple.png" alt="Stable Diffusion" width="256"></td>
    <td><img src="demo/llama/imgs/result.png" alt="QWen" width="256"></td>
    <td><img src="docs/image/demo/segment/sample_output.jpg" alt="RBMGv1.4" width="256"></td>
    <td><a href="docs/zh_cn/quick_start/model_list.md">链接</a></td>
  </tr>
</table>


## 介绍

nndeploy是一个简单易用、高性能、支持多端的AI推理部署框架。

主要解决以下模型部署中的痛点：

1. **推理框架碎片化**：针对不同硬件平台需要使用不同推理框架的问题，nndeploy提供统一接口，降低学习、开发和维护成本
   
2. **多推理框架的学习与维护成本**：通过抽象统一的接口层，使开发者只需编写一套代码即可适配多种推理框架
   
3. **模型多样性带来的挑战**：提供灵活的接口设计，轻松处理单/多输入输出、静态/动态形状等多种模型特性，无需深厚经验也能找到最优解决方案
   
4. **前后处理代码复用**：设计可复用的前后处理组件，避免重复开发相似功能，提高开发效率
   
5. **多模型组合场景的复杂性**：通过基于图的设计，降低多模型组合场景下的代码耦合度，提高灵活性和可维护性，支持并行处理

### 架构以及特点

<img src="docs/image/architecture.jpg" alt="Architecture">

### 1. 简单易用

- **基于有向无环图部署模型**： 将AI算法部署抽象为有向无环图，前处理、推理、后处理各为一个节点
 
- **推理模板Infer**： 模板可处理各种模型差异，包括单/多输入输出和静态/动态形状等等
 
- **高效解决多模型组合场景**：支持`图中嵌入图`功能，将复杂任务拆分为多个独立子图，通过组合方式快速解决多模型场景问题

- **快速构建demo**：支持多种输入输出格式（图片、文件夹、视频等），通过编解码节点化实现高效通用的demo构建

### 2. 高性能

- **多种并行模式**：支持串行（按拓扑排序依次执行节点）、流水线并行（多帧场景下将不同节点绑定到不同线程和设备）、任务并行（多模型场景下挖掘并行性缩短运行时间）以及上述组合并行模式。

- **线程池与内存池**：通过线程池提高并发性能和资源利用率，支持CPU算子自动并行（parallel_for）提升执行效率；内存池实现高效的内存分配与释放（开发中）
  
- **一组高性能的算子**：完成后将加速您模型前后处理速度(开发中)

### 3. 支持多端

- **一套代码多端部署**：通过切换推理配置，实现一套代码即可完成模型**跨多个平台以及多个推理框架**部署，性能与原始框架一致，还可直接操作推理框架内部分配的输入输出，实现前后处理的零拷贝，提升模型部署端到端的性能

- 当前支持的推理框架如下：

  | Inference/OS                                                                     | Linux | Windows | Android | MacOS |  IOS  | developer                                                                          | 
  | :------------------------------------------------------------------------------- | :---: | :-----: | :-----: | :---: | :---: | :--------------------------------------------------------------------------------- | 
  | [TensorRT](https://github.com/NVIDIA/TensorRT)                                   |   √   |    -    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          | 
  | [OpenVINO](https://github.com/openvinotoolkit/openvino)                          |   √   |    √    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          | 
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime)                          |   √   |    √    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          | 
  | [MNN](https://github.com/alibaba/MNN)                                            |   √   |    √    |    √    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          | 
  | [TNN](https://github.com/Tencent/TNN)                                            |   √   |    √    |    √    |   -   |   -   | [02200059Z](https://github.com/02200059Z)                                          | 
  | [ncnn](https://github.com/Tencent/ncnn)                                          |   -   |    -    |    √    |   -   |   -   | [Always](https://github.com/Alwaysssssss)                                          | 
  | [coreML](https://github.com/apple/coremltools)                                   |   -   |    -    |    -    |   √   |   -   | [JoDio-zd](https://github.com/JoDio-zd)、[jaywlinux](https://github.com/jaywlinux) | 
  | [AscendCL](https://www.hiascend.com/zh/)                                         |   √   |    -    |    -    |   -   |   -   | [CYYAI](https://github.com/CYYAI)                                                  | 
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) |   √   |    -    |    -    |   -   |   -   | [100312dog](https://github.com/100312dog)                                          | 
  | **[default](https://github.com/nndeploy/nndeploy)**                              |   √   |    -    |    -    |   -   |   -   | [nndeploy team](https://github.com/nndeploy)                                            | 

- **default为nndeploy内部的推理子模块**：整体架构如图所示，目前后端算子以华为昇腾NPU和CPU为主，支持ResNet50、YOLOv11、RMBG1.4等模型，更多介绍[default_inference.md]()

  <img src="docs/image/inference/inference_framework_arch.png">


## 下一步计划

- 设备管理模块
  - 新增OpenCL的设备管理模块
  - 新增ROCM的设备管理模块
  - 新增OpenGL的设备管理模块
- 内存优化
  - `主从内存拷贝优化`：针对统一内存的架构，通过主从内存映射、主从内存地址共享等方式替代主从内存拷贝
  - `内存池`：针对nndeploy的内部的数据容器Buffer、Mat、Tensor，建立异构设备的内存池，实现高性能的内存分配与释放
  - `多节点共享内存机制`：针对多模型串联场景下，基于模型部署的有向无环图，在串行执行的模式下，支持多推理节点共享内存机制
  - `边的环形队列内存复用机制`：基于模型部署的有向无环图，在流水线并行执行的模式下，支持边的环形队列共享内存机制
- 接入算子库
  - 接入oneDNN，对于部分不支持算子，手写x86平台下的实现
  - 接入cudnn和cutlass，对于部分不支持算子，手写cuda平台下的实现
  - 接入XNNPACK和QNNPACK，对于部分不支持算子，手写ARM平台下的实现
- 推理子模块
  - 支持大语言模型
  - 支持stable diffusion 
  - 增加通信原语，支持分布式推理
  - 优化内部的基于图的内存优化策略
- 部署更多的模型
  - OCR
  - 追踪
  - ...

## 联系我们
- 当前nndeploy正处于发展阶段，如果您热爱开源、喜欢折腾，不论是出于学习目的，抑或是有更好的想法，欢迎加入我们。
- 微信：titian5566 (加微信进AI推理部署交流群，请简单备注个人信息)
  
  <img src="docs/image/wechat.jpg" width="225px">


## 致谢

我们参考了以下项目：[TNN](https://github.com/Tencent/TNN)、[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)、[opencv](https://github.com/opencv/opencv)、[CGraph](https://github.com/ChunelFeng/CGraph)、[CThreadPool](https://github.com/ChunelFeng/CThreadPool)、[tvm](https://github.com/apache/tvm)、[mmdeploy](https://github.com/open-mmlab/mmdeploy)、[FlyCV](https://github.com/PaddlePaddle/FlyCV)和[oneflow](https://github.com/Oneflow-Inc/oneflow)。


## 贡献者

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>


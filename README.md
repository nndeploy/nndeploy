
[English](README_EN.md) | 简体中文

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/kapybara_logo.png">
    <img alt="nndeploy" src="docs/image/kapybara_logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
简单易用、高性能、支持多端的AI推理部署框架
</h3>

<p align="center">
| <a href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"><b>文档</b></a> | <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>公众号</b></a> | <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>知乎</b></a> | <a href="https://discord.gg/9rUwfAaMbr"><b>discord</b></a> | <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>哔哩哔哩</b></a> | <a href="https://deepwiki.com/nndeploy/nndeploy"><b>ask deepwiki</b></a> | 
</p>

---

## 新闻
- [2025/05/29] 🔥 nndeploy开源团队和昇腾官方合作的[推理框架课程](https://www.hiascend.com/developer/courses/detail/1923211251905150977)上线啦


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

1. **推理框架的碎片化**：现在业界尚不存在各方面都远超其同类产品的推理框架，不同推理框架在不同平台、硬件下分别具有各自的优势。例如，在NVidia显卡上TensorRT性能最佳，在x86 CPU上OpenVINO最优，在苹果生态下CoreML最佳，在ARM Android有ncnn、MNN等多种选择。
   
2. **多个推理框架的学习成本、开发成本、维护成本**：不同的推理框架有不一样的推理接口、超参数配置、Tensor等等，假如一个模型需要多端部署，针对不同推理框架都需要写一套代码，这对模型部署工程师而言，将带来较大学习成本、开发成本、维护成本。
   
3. **模型的多样性**：从模型部署的角度出发，可以分为单输入、多输入、单输出、多输出、静态形状输入、动态形状输入、静态形状输出、动态形状输出一系列不同。当这些差异点与内存零拷贝优化结合时，通常只有具备丰富模型部署经验的工程师才能快速找到最优解。
   
4. **模型高性能的前后处理**：模型部署不仅仅只有模型推理，还有前处理、后处理，推理框架往往只提供模型推理的功能。通常需要部署工程师基于对原始算法的理解，通过C++开发该算法前后处理，这需要大量重复工作。
   
5. **多模型的复杂场景**：目前很多场景需要由多个模型组合解决业务问题，没有部署框架的支持，会有大量业务代码、模型耦合度高、灵活性差、代码不适合并行等问题。

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

### 3. 支持多种推理后端

- **一套代码多种推理后端部署**：通过切换推理配置，实现一套代码即可完成模型**跨多个平台以及多个推理框架**部署，性能与原始框架一致

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
  | [tvm](https://github.com/apache/tvm)                              |   √   |    -    |    -    |   -   |   -   | [youxiudeshouyeren](https://github.com/youxiudeshouyeren)                                            | 
  | [snpe](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) |   √   |    -    |    -    |   -   |   -   | [yhwang-hub](https://github.com/yhwang-hub)                                            | 


### 4. 内置推理子模块

框架内部开发的推理子模块，作为缺省推理框架，当用户环境未编译链接其他推理框架时可使用此框架。**在实际应用中，推荐使用芯片厂商提供的对应平台推理框架**。

当前支持华为昇腾NPU和纯CPU算子后端。计划扩展至X86、CUDA、ARM、OpenCL等异构计算平台。

已适配主流视觉模型：图像分类（ResNet50等）、目标检测（YOLOv11等）、图像分割（RMBG1.4等）。未来将支持大语言模型（LLM）和文本图像多模态模型（Dit等）。

> 更多技术细节请参考：
  - [B站课程-AI推理框架最全视频教程](https://www.bilibili.com/video/BV1HU7CznE39?vd_source=c5d7760172919cd367c00bf4e88d6f57&spm_id_from=333.788.videopod.sections)
  - [知乎专栏](https://www.zhihu.com/column/c_1690464325314240512)

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
  - 优化内部的基于图的内存优化策略，探索更多的内存优化策略
- 部署更多的模型
  - OCR
  - 追踪
  - ...

## 联系我们
- 当前nndeploy正处于发展阶段，如果您热爱开源、喜欢折腾，不论是出于学习目的，抑或是有更好的想法，欢迎加入我们。
- 微信：titian5566 (欢迎加好友，进活跃的AI推理部署交流群，请简单备注个人信息)
  
  <img src="docs/image/wechat.jpg" width="225px">


## 致谢

- 我们参考了以下项目：[TNN](https://github.com/Tencent/TNN)、[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)、[opencv](https://github.com/opencv/opencv)、[CGraph](https://github.com/ChunelFeng/CGraph)、[CThreadPool](https://github.com/ChunelFeng/CThreadPool)、[tvm](https://github.com/apache/tvm)、[mmdeploy](https://github.com/open-mmlab/mmdeploy)、[FlyCV](https://github.com/PaddlePaddle/FlyCV)和[oneflow](https://github.com/Oneflow-Inc/oneflow)。

- 感谢[HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy)推荐

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>


## 贡献者

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)



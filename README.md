
[English](README_EN.md) | 简体中文

## 介绍

`nndeploy`是一款跨平台、高性能、简单易用的模型端到端部署框架。我们致力于屏蔽不同推理框架的差异，提供一致且用户友好的编程体验，同时专注于部署全流程的性能。

## 架构

![Architecture](docs/image/architecture.jpg)

## 特性

### 1. 支持多平台和多推理框架

只要环境支持，通过`nndeploy`部署模型的代码无需修改即可跨多个平台以及多个推理框架使用。

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
| [paddle-lite](https://github.com/PaddlePaddle/Paddle-Lite)                       |   -   |    -    |    -    |   -   |   -   | [qixuxiang](https://github.com/qixuxiang)                                          |         |
| [AscendCL](https://www.hiascend.com/zh/)                                         |   √   |    -    |    -    |   -   |   -   | [CYYAI](https://github.com/CYYAI)                                                  |         |
| [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) |   √   |    -    |    -    |   -   |   -   | [100312dog](https://github.com/100312dog)                                          |         |


**Notice:** TFLite, TVM, OpenPPL, sophgo, Horizon正在开发中，我们正在努力覆盖绝大部分的主流推理框架

### 2. 开箱即用的算法

目前已完成 [YOLOV5](https://github.com/ultralytics/yolov5)、[YOLOV6](https://github.com/meituan/YOLOv6)、[YOLOV8](https://github.com/ultralytics) 等模型的部署，可供您直接使用，后续我们持续不断去部署其它开源模型，让您开箱即用

| model                                           | Inference                         | developer                                                                            | remarks |
| :---------------------------------------------- | :-------------------------------- | :----------------------------------------------------------------------------------- | :-----: |
| [YOLOV5](https://github.com/ultralytics/yolov5) | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |
| [YOLOV6](https://github.com/meituan/YOLOv6)     | TensorRt/OpenVINO/ONNXRuntime     | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |
| [YOLOV8](https://github.com/ultralytics)        | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |


### 3. 简单易用

- **一套代码多端部署**：通过切换推理配置，一套代码即可在多端部署，算法的使用接口简单易用
  
- **算法部署简单**：将 AI 算法端到端（前处理->推理->后处理）的部署抽象为有向无环图 `Graph`，前处理为一个 `Node`，推理也为一个 `Node`，后处理也为一个 `Node`，提供了高性能的前后处理模板和推理模板，上述模板可帮助您进一步简化端到端的部署流程。有向无环图还可以高性能且高效的解决多模型部署的痛点问题

### 4. 高性能

- **推理框架的高性能抽象**：每个推理框架也都有其各自的特性，需要足够尊重以及理解这些推理框架，才能在抽象中不丢失推理框架的特性，并做到统一的使用的体验。`nndeploy` 可配置第三方推理框架绝大部分参数，保证了推理性能。可直接操作推理框架内部分配的输入输出，实现前后处理的零拷贝，提升模型部署端到端的性能。

- **线程池**：提高模型部署的并发性能和资源利用率。此外，还支持CPU端算子自动并行，可提升CPU算子执行性能
  
- **内存池**：完成后可实现高效的内存分配与释放(TODO)
  
- **一组高性能的算子**：完成后将加速您模型前后处理速度(TODO)

### 5. 并行

- **流水线并行**：在处理多帧的场景下，基于有向无环图的模型部署方式，可将前处理 `Node`、推理 `Node`、后处理 `Node`绑定三个不同的线程，每个线程又可绑定不同的硬件设备下，从而三个`Node`可流水线并行处理。在多模型以及多硬件设备的的复杂场景下，更加可以发挥流水线并行的优势，从而可显著提高整体吞吐量。

- **任务并行**：在多模型以及多硬件设备的的复杂场景下，基于有向无环图的模型部署方式，可充分挖掘模型部署中的并行性，缩短单次算法全流程运行耗时
    

## 文档
- 更多信息，访问[nndeploy文档](https://nndeploy-zh.readthedocs.io/zh/latest/)。


## 下一步规划
- 部署更多的算法
  - [BEV](https://github.com/fundamentalvision/BEVFormer)
  - [InstantID](https://github.com/InstantID/InstantID)
  - [OCR](https://github.com/PaddlePaddle/PaddleOCR)
  - ......
- 单机下的大语言模型推理模块
- 文档
- 视频
- 代码review
- 用户友好 - 编译问题、第三方库资源、模型资源、数据资源
- 完善已接入的推理框架coreml、paddle-lite，接入新的推理框架TFLite


## 参考
- [TNN](https://github.com/Tencent/TNN)
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)
- [opencv](https://github.com/opencv/opencv)
- [CGraph](https://github.com/ChunelFeng/CGraph)
- [CThreadPool](https://github.com/ChunelFeng/CThreadPool)
- [tvm](https://github.com/apache/tvm)
- [mmdeploy](https://github.com/open-mmlab/mmdeploy)
- [FlyCV](https://github.com/PaddlePaddle/FlyCV)
- [torchpipe](https://github.com/torchpipe/torchpipe)


## 加入我们
- `nndeploy`还处于初级阶段，欢迎参与，我们一起打造最简单易用、高性能的模型端到端部署框架
- 微信：titian5566 (可加我微信进nndeploy交流群，备注：nndeploy+姓名)

  <img align="left" src="docs/image/wechat.jpg" width="225px">

[简体中文](README.md) | English

## Introduction
nndeploy is a cross-platform, high-performing, and straightforward AI model deployment framework. We strive to deliver a consistent and user-friendly experience across various inference framework in complex deployment environments and focus on performance.

## Architecture
![Architecture](docs/image/architecture.jpg)

## Fetures

### 1. Supports multiple platforms and multiple inference frameworks

As long as the environment is supported, the code for deploying models through nndeploy can be used across multiple platforms without modification, regardless of the operating system and inference framework.

The current supported inference framework is as follows:

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


**Notice:** TFLite, TVM, OpenPPL, sophgo, Horizon are also on the agenda as we work to cover mainstream inference frameworks

### 2. Out-of-the-box AI models

[YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv8](https://github.com/ultralytics) are already supported, and it is believed that the list will soon be expanded. Out-of-the-box AI models are our goal.

| model                                           | Inference                         | developer                                                                            | remarks |
| :---------------------------------------------- | :-------------------------------- | :----------------------------------------------------------------------------------- | :-----: |
| [YOLOV5](https://github.com/ultralytics/yolov5) | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |
| [YOLOV6](https://github.com/meituan/YOLOv6)     | TensorRt/OpenVINO/ONNXRuntime     | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |
| [YOLOV8](https://github.com/ultralytics)        | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |


### 3. Simple and easy to use

- **One set of code for multi-platform deployment**: By switching inference configurations, one set of code can be deployed across multiple platforms, with a user-friendly and easy-to-use interface for the algorithms.

- **Simple Algorithm Deployment**: The deployment of AI algorithms from end to end (pre-processing -> inference -> post-processing) is abstracted into a Directed Acyclic Graph (DAG) Graph, with pre-processing as a Node, inference as another Node, and post-processing as yet another Node. High-performance templates for pre-processing and inference are provided. These templates can help you further simplify the end-to-end deployment process. The Directed Acyclic Graph can also solve the pain points of deploying multiple models efficiently and with high performance.

### 4. High Performance

- **High Performance Abstraction of the Inference Framework**: Each inference framework also has its own unique features. nndeploy deeply understands and preserves as much as possible the features of the inference framework without compromising the computational efficiency of the native inference framework with a consistent code experience. In addition, we realize the efficient connection between the pre/post-processing and the model inference process through the exquisitely designed memory zero copy, which effectively guarantees the end-to-end delay of model inference.

- **Thread Pool**: Enhances the concurrency performance and resource utilization of model deployment. Additionally, it supports automatic parallelism for CPU operators, which can improve the performance of CPU operator execution.

- **Memory Pool**: More efficient memory allocation and release(TODO)

- **HPC Operators**: Optimize pre/post-processing efficiency(TODO)

### 5. Parallel

- **Pipeline Parallel**: In scenarios dealing with multiple frames, the model deployment method based on Directed Acyclic Graph (DAG) allows binding the pre-processing Node, inference Node, and post-processing Node to three different threads. Each thread can be bound to different hardware devices, enabling these three Nodes to process in a pipeline parallel manner. In complex scenarios involving multiple models and multiple hardware devices, the advantages of pipeline parallelism can be fully leveraged, significantly increasing the overall throughput.

- **Task Parallel**: In complex scenarios involving multiple models and multiple hardware devices, the model deployment method based on Directed Acyclic Graph (DAG) can fully explore the parallelism in model deployment, shortening the duration of a single algorithm's full process execution.


## Document
- For more information, please visit the [nndeploy documentation](https://nndeploy-zh.readthedocs.io/zh/latest/).


## Roadmap
- Deploy more algorithms
  - [BEV](https://github.com/fundamentalvision/BEVFormer)
  - [InstantID](https://github.com/InstantID/InstantID)
  - [OCR](https://github.com/PaddlePaddle/PaddleOCR)
  - ......
- Large language model inference module on a single machine
- Documentation
- Video
- Code review
- User-friendly - compilation issues, third-party library resources, model resources, data resources
- Enhance already integrated inference frameworks like coreml, paddle-lite, and integrate new inference framework TFLite


# Reference
- [TNN](https://github.com/Tencent/TNN)
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)
- [opencv](https://github.com/opencv/opencv)
- [CGraph](https://github.com/ChunelFeng/CGraph)
- [CThreadPool](https://github.com/ChunelFeng/CThreadPool)
- [tvm](https://github.com/apache/tvm)
- [mmdeploy](https://github.com/open-mmlab/mmdeploy)
- [FlyCV](https://github.com/PaddlePaddle/FlyCV)
- [torchpipe](https://github.com/torchpipe/torchpipe)


## Contact Us
- nndeploy is still in its infancy, welcome to join us.
- Wechat：titian5566
  
  <img align="left" src="docs/image/wechat.jpg" width="225px">

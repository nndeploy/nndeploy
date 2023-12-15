
[简体中文](README.md) | English

## Introduction
nndeploy is a cross-platform, high-performing, and straightforward AI model deployment framework. We strive to deliver a consistent and user-friendly experience across various inference framework backends in complex deployment environments and focus on performance.

## Architecture
![Architecture](docs/image/architecture.jpg)

## Fetures

### 1. cross-platform and consistent

As long as the environment is supported, the code for deploying models through nndeploy can be used across multiple platforms without modification, regardless of the operating system and inference framework.

The current supported environment is as follows, which will continue to be updated in the future:

| Inference/OS                                               | Linux | Windows | Android | MacOS |  IOS  | developer                                 | remarks |
| :--------------------------------------------------------- | :---: | :-----: | :-----: | :---: | :---: | :---------------------------------------- | :-----: |
| [TensorRT](https://github.com/NVIDIA/TensorRT)             |   √   |    -    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss) |         |
| [OpenVINO](https://github.com/openvinotoolkit/openvino)    |   √   |    √    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss) |         |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime)    |   √   |    √    |    -    |   -   |   -   | [Always](https://github.com/Alwaysssssss) |         |
| [MNN](https://github.com/alibaba/MNN)                      |   √   |    √    |    √    |   -   |   -   | [Always](https://github.com/Alwaysssssss) |         |
| [TNN](https://github.com/Tencent/TNN)                      |   √   |    √    |    √    |   -   |   -   | [02200059Z](https://github.com/02200059Z) |         |
| [ncnn](https://github.com/Tencent/ncnn)                    |   -   |    -    |    √    |   -   |   -   | [Always](https://github.com/Alwaysssssss) |         |
| [coreML](https://github.com/apple/coremltools)             |   -   |    -    |    -    |   √   |   -   | [JoDio-zd](https://github.com/JoDio-zd)   |         |
| [paddle-lite](https://github.com/PaddlePaddle/Paddle-Lite) |   -   |    -    |    -    |   -   |   -   | [qixuxiang](https://github.com/qixuxiang) |         |
| [MDC](https://github.com/PaddlePaddle/Paddle-Lite)         |   √   |    -    |    -    |   -   |   -   | [CYYAI](https://github.com/CYYAI)         |         |

**Notice:** TFLite, TVM, OpenPPL, Tengine, AITemplate, RKNN, sophgo, MindSpore-lite, Horizon are also on the agenda as we work to cover mainstream inference frameworks

### 2. High Performance

The difference of model structure, inference framework and hardware resource will lead to different inference performance. nndeploy deeply understands and preserves as much as possible the features of the back-end inference framework without compromising the computational efficiency of the native inference framework with a consistent code experience. In addition, we realize the efficient connection between the pre/post-processing and the model inference process through the exquisitely designed memory zero copy, which effectively guarantees the end-to-end delay of model inference.

What's more, we are developing and refining the following:
* **Thread Pool**: better pipelined parallel optimization
* **Memory Pool**: more efficient memory allocation and release
* **HPC Operators**: optimize pre/post-processing efficiency

### 3. Models built-in

Out-of-the-box AI models are our goal, but our are focusing on development of the system at this time. Nevertheless, [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv8](https://github.com/ultralytics) are already supported, and it is believed that the list will soon be expanded.

| model                                           | Inference                         | developer                                                                            | remarks |
| :---------------------------------------------- | :-------------------------------- | :----------------------------------------------------------------------------------- | :-----: |
| [YOLOV5](https://github.com/ultralytics/yolov5) | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |
| [YOLOV6](https://github.com/meituan/YOLOv6)     | TensorRt/OpenVINO/ONNXRuntime     | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |
| [YOLOV8](https://github.com/ultralytics)        | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |         |


### 4. user-friendly

nndeploy's primary purpose is user friendliness and high performance. We have built-in support for the major inference frameworks and provide them with a unified interface abstraction on which you can implement platform/framework independent inference code without worrying about performance loss. We now provide additional templates for the pre/post-processing for AI algorithms, which can help you simplify the end-to-end deployment process of the model, and the built-in algorithms mentioned above are also part of the ease of use.

If you have any related questions, feel free to contact us. 😁

## Document
- For more information, please visit the [nndeploy documentation](https://nndeploy-zh.readthedocs.io/zh/latest/introduction/index.html).

## Roadmap
- Parallel
- More Model
- More Inference
- OP

## Support
| OS      |                                                                      status                                                                      |
| ------- | :----------------------------------------------------------------------------------------------------------------------------------------------: |
| Linux | [![linux](https://github.com/DeployAI/nndeploy/actions/workflows/linux.yml/badge.svg)](https://github.com/DeployAI/nndeploy/actions/workflows/linux.yml) |
| Macos | [![macos](https://github.com/DeployAI/nndeploy/actions/workflows/macos.yml/badge.svg)](https://github.com/DeployAI/nndeploy/actions/workflows/macos.yml) |
| Windows | [![windows](https://github.com/DeployAI/nndeploy/actions/workflows/windows.yml/badge.svg)](https://github.com/DeployAI/nndeploy/actions/workflows/windows.yml) |

# Reference
- [TNN](https://github.com/Tencent/TNN)
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)
- [opencv](https://github.com/opencv/opencv)
- [CGraph](https://github.com/ChunelFeng/CGraph)
- [CThreadPool](https://github.com/ChunelFeng/CThreadPool)
- [tvm](https://github.com/apache/tvm)
- [mmdeploy](https://github.com/open-mmlab/mmdeploy)
- [FlyCV](https://github.com/PaddlePaddle/FlyCV)
- [ThreadPool](https://github.com/progschj/ThreadPool)
- [torchpipe](https://github.com/torchpipe/torchpipe)

## COntributors
- [02200059Z](https://github.com/02200059Z)
- [JoDio-zd](https://github.com/JoDio-zd)
- [qixuxiang](https://github.com/qixuxiang)
- [CYYAI](https://github.com/CYYAI)
- [Always](https://github.com/Alwaysssssss)
- [youxiudeshouyeren](https://github.com/youxiudeshouyeren)
- [PeterH0323](https://github.com/PeterH0323)
- [100312dog](https://github.com/100312dog)
- [wangzhaode](https://github.com/wangzhaode)
- [ChunelFeng](https://github.com/ChunelFeng)
- [acheerfulish](https://github.com/acheerfulish)

## Contact Us
> nndeploy is still in its infancy, welcome to join us.

* Wechat：titian5566
  
  <img align="left" src="docs/image/wechat.jpg" width="225px">

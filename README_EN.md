
[简体中文](README.md) | English

## Introduction

nndeploy is an end-to-end model inference and deployment framework. It aims to provide users with a powerful, easy-to-use, high-performance, and mainstream framework compatible model inference and deployment experience.

## Architecture

![Architecture](docs/image/architecture.jpg)

## Fetures

### 1. Out-of-the-box AI models

[YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [YOLOv8](https://github.com/ultralytics) are already supported, and it is believed that the list will soon be expanded. Out-of-the-box AI models are our goal.

| model                                                       | Inference                         | developer                                                                                            | remarks |
| :---------------------------------------------------------- | :-------------------------------- | :--------------------------------------------------------------------------------------------------- | :-----: |
| [YOLOV5](https://github.com/ultralytics/yolov5)             | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss)                 |         |
| [YOLOV6](https://github.com/meituan/YOLOv6)                 | TensorRt/OpenVINO/ONNXRuntime     | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss)                 |         |
| [YOLOV8](https://github.com/ultralytics)                    | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss)                 |         |
| [SAM](https://github.com/facebookresearch/segment-anything) | ONNXRuntime                       | [youxiudeshouyeren](https://github.com/youxiudeshouyeren)、[Always](https://github.com/Alwaysssssss) |         |

### 2. Supports multiple platforms and multiple inference frameworks

**One Codebase for Multi-Platform Deployment**: By switching inference configurations, a single codebase can accomplish model deployment across multiple platforms and various inference frameworks.

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

### 3. Simple and easy to use

- **Deploying Models Based on Directed Acyclic Graphs (DAG)**: The end-to-end deployment of AI algorithms (preprocessing -> inference -> postprocessing) is abstracted as a directed acyclic graph `Graph`, where preprocessing is one `Node`, inference is another `Node`, and postprocessing is also a `Node`.

- **Inference Template**: Based on the `multi-end inference module Inference` combined with the `directed acyclic graph node Node`, a powerful `Inference Template` is designed. The Infer inference template can help you handle differences brought by different models internally, such as **single input, multiple inputs, single output, multiple outputs, static shape input, dynamic shape input, static shape output, dynamic shape output**, and a series of other variations.

- **Efficiently Solving Complex Scenarios with Multiple Models**: In complex scenarios where multiple models are combined to complete a single task (e.g., old photo restoration), each model can be an independent Graph. The directed acyclic graph of nndeploy supports `embedding graphs within graphs`, a flexible and powerful feature that breaks down large problems into smaller ones, enabling the rapid resolution of complex scenarios involving multiple models through a composite approach.

- **Rapid Construction of Demos**: When a model has been deployed, it is necessary to write a demo to showcase its effects. Demos need to handle various formats of input, such as image input and output, multiple images in a folder, video input and output, etc. By node-izing the aforementioned encoding and decoding processes, one can more universally and efficiently complete the writing of demos, achieving the goal of quickly demonstrating effects (currently, mainly node-ized based on OpenCV for encoding and decoding).

### 4. High Performance

- **High Performance Abstraction of the Inference Framework**: Each inference framework also has its own unique features. nndeploy deeply understands and preserves as much as possible the features of the inference framework without compromising the computational efficiency of the native inference framework with a consistent code experience. In addition, we realize the efficient connection between the pre/post-processing and the model inference process through the exquisitely designed memory zero copy, which effectively guarantees the end-to-end delay of model inference.

- **Thread Pool**: Enhances the concurrency performance and resource utilization of model deployment. Additionally, it supports automatic parallelism for CPU operators, which can improve the performance of CPU operator execution.

- **Memory Pool**: More efficient memory allocation and release(TODO)

- **HPC Operators**: Optimize pre/post-processing efficiency(TODO)

### 5. Parallel

- **Serial Execution**: Execute each node in the order of the topological sort of the directed acyclic graph (DAG) used for model deployment.

- **Pipeline Parallelism**: In scenarios where multiple frames are processed, the model deployment method based on the directed acyclic graph allows for binding the preprocessing `Node`, inference `Node`, and postprocessing `Node` to three different threads. Each thread can be further bound to different hardware devices, enabling the three `Nodes` to process in a pipelined parallel manner. In complex scenarios with multiple models and hardware devices, pipeline parallelism can be particularly advantageous, significantly improving overall throughput.

- **Task Parallelism**: In complex scenarios with multiple models and hardware devices, the model deployment method based on the directed acyclic graph can fully exploit the parallelism within model deployment, reducing the time taken for a single algorithm end-to-end process.

- **Combination of Parallel Modes**: In complex scenarios involving multiple models, hardware devices, and the processing of multiple frames, nndeploy's directed acyclic graph supports the embedding of graphs within graphs. Each graph can have an independent parallel mode, allowing users to freely combine parallel modes for model deployment tasks, thereby fully leveraging hardware performance.


## Document
- For more information, please visit the [nndeploy documentation](https://nndeploy-zh.readthedocs.io/zh/latest/).

## resource repository

- We have uploaded third-party libraries, model repositories, and test data to [HuggingFace](https://huggingface.co/alwaysssss/nndeploy). If you need them, feel free to download them.


## Roadmap

- **Inference Backend**
  - Improve the existing inference framework CoreML
  - Enhance the existing inference framework Paddle-Lite
  - Integrate the new inference framework TFLite

- **Device Management**
  - Add a device management module for OpenCL
  - Add a device management module for ROCM
  - Add a device management module for OpenGL

- **Memory Optimization**
  - **Host-Device Memory Copy Optimization**: For unified memory architectures, replace host-device memory copying with methods such as host-device memory mapping and host-device memory address sharing.
  - **Memory Pool**: For nndeploy's internal data containers such as Buffer, Mat, and Tensor, establish a heterogeneous device memory pool to achieve high-performance memory allocation and release.
  - **Multi-Node Shared Memory Mechanism**: In scenarios where multiple models are connected in series, based on the directed acyclic graph used for model deployment, support a shared memory mechanism for multiple inference nodes under a serial execution mode.
  - **Edge's Ring Queue Memory Reuse Mechanism**: Based on the directed acyclic graph used for model deployment, support a shared memory mechanism for the edge's ring queue under a pipeline parallel execution mode.

- **Stable Diffusion Model**
  - Deploy the Stable Diffusion Model
  - Build stable_diffusion.cpp for the Stable Diffusion Model (inference submodule, manually constructing the computational graph)
  - High-Performance OP
  - **Distributed Computing**
    - In scenarios where multiple models collaboratively complete a single task, dispatch multiple models to multiple machines for distributed execution.
    - In scenarios involving large models, the method of partitioning a large model into several sub-models, and subsequently dispatching these sub-models for distributed execution across various machines, is a viable strategy.


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
- nndeploy is currently in its development stage. If you are passionate about open source and enjoy tinkering, whether for learning purposes or if you have better ideas, you are welcome to join us.
- WeChat: titian5566 (Please briefly introduce yourself when adding WeChat to join the AI Inference Deployment communication group)

  <img align="left" src="docs/image/wechat.jpg" width="225px">

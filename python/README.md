
<h3 align="center">
nndeploy: Your Local AI Workflow
</h3>

## Features

Write algorithm node logic in Python/C++ without frontend skills to quickly build your visual AI workflow.

Provides out-of-the-box algorithm nodes for non-AI programmers, including large language models, Stable Diffusion, object detection, image segmentation, etc. Build AI applications quickly through drag-and-drop.

Workflows can be exported as JSON configuration files, supporting direct loading and execution via Python/C++ APIs, deployable to cloud servers, desktop, mobile, and edge devices across multiple platforms.

The framework features built-in mainstream high-performance inference engines and deep optimization strategies to help you transform workflows into enterprise-grade production applications.

### **Efficiency**
- **Visual Workflow**: Build professional AI workflows quickly through drag-and-drop operations, supporting real-time parameter adjustment on the frontend and instant backend response, view execution time for each node
- **Custom Nodes**: You only need to write algorithm node logic using familiar Python/C++, no frontend technology required, the framework automatically converts code into workflow nodes
- **Algorithm Composition**: Flexibly combine different algorithms to rapidly build innovative AI applications
- **One-Click Deployment**: Built workflows can be exported as JSON, directly callable by Python/C++, seamless transition from development to production environment

### **Performance**
- **13 Inference Engines Seamlessly Integrated**: One workflow, multi-platform deployment. Zero-abstraction cost integration with 13 mainstream inference frameworks, covering cloud, desktop, mobile, and edge platforms

  | Inference Framework | Use Case | Status |
  | :------- | :------ | :--- |
  | [PyTorch](https://pytorch.org/) | R&D debugging, rapid prototyping | ✅ |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime) | Cross-platform inference | ✅ |
  | [TensorRT](https://github.com/NVIDIA/TensorRT) | NVIDIA GPU high-performance inference | ✅ |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino) | Intel CPU/GPU optimization | ✅ |
  | [MNN](https://github.com/alibaba/MNN) | Alibaba's mobile inference engine | ✅ |
  | [TNN](https://github.com/Tencent/TNN) | Tencent's mobile inference engine | ✅ |
  | [ncnn](https://github.com/Tencent/ncnn) | Tencent's mobile inference engine | ✅ |
  | [CoreML](https://github.com/apple/coremltools) | iOS/macOS native acceleration | ✅ |
  | [AscendCL](https://www.hiascend.com/zh/) | Huawei Ascend AI chip inference framework | ✅ |
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) | Rockchip NPU inference framework | ✅ |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | Qualcomm Snapdragon NPU inference framework | ✅ |
  | [TVM](https://github.com/apache/tvm) | Deep learning compilation stack | ✅ |
  | [Custom Inference Framework](docs/zh_cn/inference/README_INFERENCE.md) | Custom inference requirements | ✅ |

- **Parallel Optimization**: Support for serial, pipeline parallel, and task parallel execution modes
- **Memory Optimization**: Zero-copy, memory pools, memory reuse optimization strategies
- **High-Performance Optimization**: Built-in nodes optimized with C++/CUDA/Ascend C/SIMD implementations

## More infomation
You can get everything in nndeploy github main page : [nndeploy](https://github.com/nndeploy/nndeploy)

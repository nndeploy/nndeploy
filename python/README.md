
<h3 align="center">
Workflow-based Multi-platform AI Deployment Tool
</h3>

## Features

nndeploy is a workflow-based multi-platform AI deployment tool with the following capabilities:

### 1. Efficiency Tool for AI Deployment

- **Visual Workflow**: Deploy AI algorithms through drag-and-drop interface

- **Function Calls**: Export workflows as JSON configuration files, supporting Python/C++ API calls

- **Multi-platform Inference**: One workflow, multi-platform deployment. Zero-abstraction cost integration with 13 mainstream inference frameworks, covering cloud, desktop, mobile, and edge platforms

  | Framework | Support Status |
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
  | [Custom Inference Framework](docs/zh_cn/inference/README_INFERENCE.md) | ✅ |

### 2. Performance Tool for AI Deployment

- **Parallel Optimization**: Support for serial, pipeline parallel, and task parallel execution modes

- **Memory Optimization**: Zero-copy, memory pools, memory reuse and other optimization strategies
  
- **High-Performance Optimization**: Built-in nodes optimized with C++/CUDA/SIMD implementations

### 3. Creative Tool for AI Deployment

- **Custom Nodes**: Support Python/C++ custom nodes with seamless integration into visual interface without frontend code

- **Algorithm Composition**: Flexible combination of different algorithms to rapidly build innovative AI applications

- **What You Tune Is What You See**: Frontend visual adjustment of all node parameters in AI algorithm deployment with quick preview of post-tuning effects

## More infomation
You can get everything in nndeploy github main page : [nndeploy](https://github.com/nndeploy/nndeploy)

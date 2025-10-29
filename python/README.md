
<h3 align="center">
nndeploy: An Easy-to-Use, and High-Performance AI Deployment Framework
</h3>

## Introduction

nndeploy is an easy-to-use, and high-performance AI deployment framework. Based on the design concepts of visual workflows and multi-backend inference, developers can quickly develop SDKs for specified platforms and hardware from algorithm repositories, significantly saving development time. Furthermore, the framework has already deployed numerous AI models including LLM, AIGC generation, face swap, object detection, image segmentation, etc., ready to use out-of-the-box.

### **Simple and Easy to Use**

- **Visual Workflow**: Deploy AI algorithms through drag-and-drop operations. Visually adjust all node parameters of the AI algorithm in the frontend and quickly preview the effect after parameter tuning.
- **Custom Nodes**: Support Python/C++ custom nodes, seamlessly integrated into the visual interface without frontend code.
- **Algorithm Combination**: Flexibly combine different algorithms to quickly build innovative AI applications.
- **One-Click Deployment**: The completed workflow can be exported as a JSON configuration file with one click, supporting direct calls via Python/C++ API, achieving seamless transition from development to production environments, and fully supporting platforms like Linux, Windows, macOS, Android, iOS, etc.

### **High Performance**

- **Parallel Optimization**: Supports execution modes like serial, pipeline parallel, task parallel, etc.
- **Memory Optimization**: Optimization strategies like zero-copy, memory pool, memory reuse, etc.
- **High-Performance Optimization**: Built-in nodes optimized with C++/CUDA/Ascend C/SIMD, etc.
- **Multi-Backend Inference**: One workflow, multiple backend inference. Integrates 13 mainstream inference frameworks with zero abstraction cost, covering all platforms including cloud, desktop, mobile, edge, etc.

  | Inference Framework                                                                         | Application Scenario              | Status |
  | :----------------------------------------------------------------------------------------- | :-------------------------------- | :----- |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime)                                    | Cross-platform inference          | ✅     |
  | [TensorRT](https://github.com/NVIDIA/TensorRT)                                             | NVIDIA GPU high-performance inference | ✅     |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino)                                    | Intel CPU/GPU optimization        | ✅     |
  | [MNN](https://github.com/alibaba/MNN)                                                      | Mobile inference engine by Alibaba| ✅     |
  | [TNN](https://github.com/Tencent/TNN)                                                      | Mobile inference engine by Tencent| ✅     |
  | [ncnn](https://github.com/Tencent/ncnn)                                                    | Mobile inference engine by Tencent| ✅     |
  | [CoreML](https://github.com/apple/coremltools)                                             | iOS/macOS native acceleration     | ✅     |
  | [AscendCL](https://www.hiascend.com/zh/)                                                   | Huawei Ascend AI chip inference framework | ✅     |
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html)           | Rockchip NPU inference framework  | ✅     |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)             | Qualcomm Snapdragon NPU inference framework | ✅     |
  | [TVM](https://github.com/apache/tvm)                                                       | Deep learning compiler stack      | ✅     |
  | [PyTorch](https://pytorch.org/)                                                            | Rapid prototyping / Cloud deployment | ✅     |
  | [Self-developed Inference Framework](docs/zh_cn/inference/README_INFERENCE.md)             | Default inference framework       | ✅     |

### **Out-of-the-Box Algorithms**

List of deployed models, with **100+ nodes** created. We will continue to deploy more high-value AI algorithms. If you have algorithms you need deployed, please let us know via [issue](https://github.com/nndeploy/nndeploy/issues).

| Application Scenario   | Available Models                                                                              | Remarks                                             |
| ---------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Large Language Model** |**QWen-2.5**, **QWen-3**                                                                     |                                                     |
| **Image Generation**   | Stable Diffusion 1.5, Stable Diffusion XL, Stable Diffusion 3, HunyuanDiT, etc.               | Supports text-to-image, image-to-image, inpainting; based on **diffusers** |
| **Face Swap**          | **deep-live-cam**                                                                             |                                                     |
| **OCR**                | **Paddle OCR**                                                                                |                                                     |
| **Object Detection**   | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx**                                           |                                                     |
| **Object Tracking**    | FairMot                                                                                       |                                                     |
| **Image Segmentation** | RBMGv1.4, PPMatting, **Segment Anything**                                                     |                                                     |
| **Classification**     | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet                    |                                                     |
| **API Service**        | OPENAI, DeepSeek, Moonshot                                                                    | Supports LLM and AIGC services                      |

> See more details in the [Deployed Model List Details](docs/zh_cn/quick_start/model_list.md)

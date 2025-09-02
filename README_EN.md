[简体中文](README.md) | English

<h3 align="center">
nndeploy: Your Local AI Workflow
</h3>

<p align="center">
<a href="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml/badge.svg" alt="Linux">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml/badge.svg" alt="Windows">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml/badge.svg" alt="Android">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml/badge.svg" alt="macOS">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml/badge.svg" alt="iOS">
</a>
</p>

<p align="center">
<a href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"><b>Documentation</b></a> 
| <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>WeChat</b></a> 
| <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>Zhihu</b></a> 
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
| <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>Bilibili</b></a> 
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow.gif" width=100%>
  </picture>
</p>

Write algorithm node logic in Python/C++ without frontend skills to quickly build your visual AI workflow.

Provides out-of-the-box algorithm nodes for non-AI programmers, including large language models, Stable Diffusion, object detection, image segmentation, etc. Build AI applications quickly through drag-and-drop.

Workflows can be exported as JSON configuration files, supporting direct loading and execution via Python/C++ APIs, deployable to cloud servers, desktop, mobile, and edge devices across multiple platforms.

The framework features built-in mainstream high-performance inference engines and deep optimization strategies to help you transform workflows into enterprise-grade production applications.

---

## Quick Start

### Installation

```bash
pip install --upgrade nndeploy
```

### Launch Visual Workflow Interface

```bash
# Method 1: Use built-in nodes only
nndeploy-app --port 8000

# Method 2: Use custom nodes
nndeploy-app --port 8000 --plugin plugin1.py plugin2.py 
```

- Command parameter description
  - `--port`: Specify web service port (default: 8000)
  - `--plugin`: Load custom plugin files (optional parameter, if not provided, only built-in nodes are used)
    - Python plugin: Refer to [Python Plugin Template](template/python/template.py)
    - C++ plugin: Refer to [C++ Plugin Template](template/cpp/template.h)
    - Multiple plugins can be loaded simultaneously: `--plugin plugin1.py plugin2.so`

After successful startup, open http://localhost:8000 to access the workflow interface.

#### Quick Tutorial Demo

Build AI workflows through drag-and-drop operations, intuitive and easy to understand, get started in just a few minutes.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="quick_start.gif">
    <img alt="nndeploy" src="docs/image/quick_start.gif" width=100%>
  </picture>
</p>


### Save Workflow as JSON and Execute Workflow

After configuring your workflow in the visual interface, you can save it as a JSON file (e.g., workflow.json). You can execute the workflow using the following command:

```bash
nndeploy-run-json --json-file workflow.json --plugin plugin.py
```

- API for loading and running JSON configuration files
  - [Python API Example Code](python/nndeploy/dag/run_json.py)
  - [C++ API Example Code](framework/include/nndeploy/dag/graph_runner.h)

> Requires Python 3.10 or higher. Includes PyTorch and ONNXRuntime inference backends by default. For additional inference backends (such as TensorRT, OpenVINO, ncnn, MNN, etc.), please use developer mode.

> Use `nndeploy-clean` to clear expired backend resources.

### Documentation
- [How to Build](docs/zh_cn/quick_start/build.md)
- [How to Get Models](docs/zh_cn/quick_start/model.md)
- [How to Execute](docs/zh_cn/quick_start/example.md)
- [Python Quick Start](docs/zh_cn/quick_start/python.md)
- [Visual Workflow Quick Start](docs/zh_cn/quick_start/workflow.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [C++ Plugin Development Manual](docs/zh_cn/quick_start/plugin.md)
- [Python++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)
- [Python Plugin Development Manual](docs/zh_cn/quick_start/plugin_python.md)

## Core Features

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

## Out-of-the-Box Nodes

For the following algorithms, we have developed 40+ nodes and created [workflow templates](https://github.com/nndeploy/nndeploy-workflow) for you to immediately experience and use various AI functions:

| Application Scenario | Available Models | 
|---------|---------|
| **Image Classification** | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet |
| **Object Detection** | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx** | 
| **Object Tracking** | FairMot | 
| **Image Segmentation** | RBMGv1.4, PPMatting, **Segment Anything** |
| **Generative Models** | **Stable Diffusion 1.5** | 
| **Large Language Models** | **QWen-0.5B** | 
| **Face Swapping** | **deep-live-cam** | 

### YOLO Visual Parameter Tuning and One-Click Deployment

Visually adjust detection parameters in real-time, observe effect changes without modifying code, support one-click switching to inference engines like TensorRT for high-performance deployment.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="yolo_edit_param.gif">
    <img alt="nndeploy" src="docs/image/yolo_edit_deploy.gif" width=100%>
  </picture>
</p>

### Multi-Model Workflow Demo

Visually build detection + segmentation + classification workflows, support multi-inference framework switching and parallel modes, achieving build once, deploy everywhere.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="seg_detect_class.gif">
    <img alt="nndeploy" src="docs/image/seg_detect_class.gif" width=100%>
  </picture>
</p>

### Zero-Code Face Swapping + Segmentation Workflow

Combine face detection, face swapping algorithms, portrait segmentation and other AI functions through drag-and-drop operations, no coding required, parameter adjustments show effects in 1-2 seconds. Let **product managers, designers, and non-AI developers** quickly turn ideas into prototypes.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="face_swap_seg.gif">
    <img alt="nndeploy" src="docs/image/face_swap_seg.gif" width=100%>
  </picture>
</p>

> More nodes will be continuously developed. If you need specific algorithms, please let us know via [issue](https://github.com/nndeploy/nndeploy/issues)

## Stay Ahead

Star nndeploy on GitHub and get notified of new releases immediately.

<img src="docs/image/star.gif">

## Roadmap

- [Workflow Ecosystem](https://github.com/nndeploy/nndeploy/issues/191)
- [Edge-side Large Model Inference](https://github.com/nndeploy/nndeploy/issues/161)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)
- [Architecture Optimization](https://github.com/nndeploy/nndeploy/issues/189)

## Contact Us
- **Join Developer Community**: Communicate with engineers, get support, and experience new features first! WeChat: Always031856 (Please note: Name + Technical Direction)
  
  <img src="docs/image/wechat.jpg" width="225px">

## Acknowledgments

- Thanks to the following projects: [TNN](https://github.com/Tencent/TNN), [FastDeploy](https://github.com/PaddlePaddle/FastDeploy), [opencv](https://github.com/opencv/opencv), [CGraph](https://github.com/ChunelFeng/CGraph), [CThreadPool](https://github.com/ChunelFeng/CThreadPool), [tvm](https://github.com/apache/tvm), [mmdeploy](https://github.com/open-mmlab/mmdeploy), [FlyCV](https://github.com/PaddlePaddle/FlyCV), [oneflow](https://github.com/Oneflow-Inc/oneflow), [flowgram.ai](https://github.com/bytedance/flowgram.ai), [deep-live-cam](https://github.com/hacksider/Deep-Live-Cam).

- Thanks to [HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy) for the recommendation

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## Contributors

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)

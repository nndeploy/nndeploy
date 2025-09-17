[简体中文](README.md) | English

<p align="left">
<a href="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/linux.yml/badge.svg" alt="Linux" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/windows.yml/badge.svg" alt="Windows" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/android.yml/badge.svg" alt="Android" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/macos.yml/badge.svg" alt="macOS" style="height: 16px;">
</a>
 <a href="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml">
  <img src="https://github.com/nndeploy/nndeploy/actions/workflows/ios.yml/badge.svg" alt="iOS" style="height: 16px;">
</a>
 <a href="https://pepy.tech/projects/nndeploy">
  <img src="https://static.pepy.tech/personalized-badge/nndeploy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" style="height: 16px;">
</a>
</p>

<h1 align="center">
nndeploy
</h1>

nndeploy is a **workflow-based multi-platform AI deployment framework** that makes AI algorithm deployment as simple as building with blocks!  

It adopts a visual workflow design with a rich set of built-in algorithm nodes. Users can quickly construct professional AI applications through drag-and-drop operations, eliminating the need to write complex code.  

It supports custom node development in Python/C++—no front-end code is required, and nodes are automatically integrated into the visual interface. 

Built workflows can be exported as JSON configuration files with one click and loaded for execution via Python/C++ APIs. It integrates mainstream inference engines and advanced optimization strategies to ensure optimal performance, enabling "develop once, deploy across multiple devices" and covering all platforms including Linux, Windows, macOS, Android, and iOS.  


<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow.gif" width=100%>
  </picture>
</p>

---


## Quick Start

### Installation
```bash
pip install --upgrade nndeploy
```

### Launch the Visual Workflow
```bash
# Method 1: Use only built-in nodes
nndeploy-app --port 8000

# Method 2: Use user-defined nodes
nndeploy-app --port 8000 --plugin plugin1.py plugin2.py 
```

- Command Parameter Description:
  - `--port`: Specify the Web service port number (default: 8000)
  - `--plugin`: Load user-defined plugin files (optional; if omitted, only built-in nodes are used)
    - Python plugins: Refer to the [Python Plugin Template](template/python/template.py)
    - C++ plugins: Refer to the [C++ Plugin Template](template/cpp/template.h)
    - Multiple plugins can be loaded simultaneously: `--plugin plugin1.py plugin2.so`

Once launched successfully, open http://localhost:8000 to access the workflow interface.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="quick_start.gif">
    <img alt="nndeploy" src="docs/image/quick_start.gif" width=100%>
  </picture>
</p>


### Export and Execute the Workflow
After building the workflow in the visual interface, save it as a JSON file (e.g., `workflow.json`), then execute the workflow using the following command:
```bash
nndeploy-run-json --json-file workflow.json --plugin plugin.py
```

- API for Loading and Running JSON Configuration Files:
  - [Python API Example Code](python/nndeploy/dag/run_json.py)
  - [C++ API Example Code](framework/include/nndeploy/dag/graph_runner.h)

> Requires Python 3.10 or higher. By default, it includes two inference backends: PyTorch and ONNXRuntime. For additional inference backends (e.g., TensorRT, OpenVINO, ncnn, MNN), please use the developer mode.  

> Use `nndeploy-clean` to clear outdated backend resources.

### Documentation
- [How to Build](docs/en/quick_start/build.md)
- [How to Obtain Models](docs/en/quick_start/model.md)
- [How to Execute](docs/en/quick_start/example.md)
- [Python Quick Start](docs/en/quick_start/python.md)
- [Visual Workflow Quick Start](docs/en/quick_start/workflow.md)
- [C++ API](https://nndeploy.readthedocs.io/en/latest/cpp_api/doxygen.html)
- [C++ Plugin Development Guide](docs/en/quick_start/plugin.md)
- [Python++ API](https://nndeploy.readthedocs.io/en/latest/python_api/index.html)
- [Python Plugin Development Guide](docs/en/quick_start/plugin_python.md)


## Core Features

### **Efficiency Tool for AI Deployment**
- **Visual Workflow**: Deploy AI algorithms via drag-and-drop operations. Adjust all node parameters of the AI algorithm visually in the front end and preview the effect of parameter adjustments in real time.
- **Custom Nodes**: Support custom nodes in Python/C++—no front-end code required, and nodes are seamlessly integrated into the visual interface.
- **Algorithm Combination**: Flexibly combine different algorithms to quickly build innovative AI applications.
- **One-Click Deployment**: Export built workflows as JSON files, which can be directly called by Python/C++—enabling a seamless transition from development to production.

### **Performance Tool for AI Deployment**
- **Seamless Integration with 13 Inference Engines**: One workflow, deployed across multiple devices. It integrates 13 mainstream inference frameworks with zero abstraction cost, covering cloud, desktop, mobile, edge, and other full-platform scenarios.

  | Inference Framework | Application Scenario       | Status |
  | :----------------- | :------------------------- | :----- |
  | [PyTorch](https://pytorch.org/) | R&D debugging, rapid prototyping | ✅ |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime) | Cross-platform inference | ✅ |
  | [TensorRT](https://github.com/NVIDIA/TensorRT) | High-performance inference on NVIDIA GPU | ✅ |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino) | Optimization for Intel CPU/GPU | ✅ |
  | [MNN](https://github.com/alibaba/MNN) | Mobile inference engine developed by Alibaba | ✅ |
  | [TNN](https://github.com/Tencent/TNN) | Mobile inference engine developed by Tencent | ✅ |
  | [ncnn](https://github.com/Tencent/ncnn) | Mobile inference engine developed by Tencent | ✅ |
  | [CoreML](https://github.com/apple/coremltools) | Native acceleration for iOS/macOS | ✅ |
  | [AscendCL](https://www.hiascend.com/zh/) | Inference framework for Huawei Ascend AI chips | ✅ |
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) | Inference framework for Rockchip NPU | ✅ |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | Inference framework for Qualcomm Snapdragon NPU | ✅ |
  | [TVM](https://github.com/apache/tvm) | Deep learning compilation stack | ✅ |
  | [Self-developed Inference Framework](docs/en/inference/README_INFERENCE.md) | Custom inference requirements | ✅ |

- **Parallel Optimization**: Supports execution modes such as serial, pipeline parallelism, and task parallelism.
- **Memory Optimization**: Adopts optimization strategies including zero-copy, memory pooling, and memory reuse.
- **High-Performance Optimization**: Built-in nodes optimized with C++/CUDA/Ascend C/SIMD implementations.


## Ready-to-Use Nodes
For the following AI algorithms, we have developed 40+ nodes and created [workflow templates](https://github.com/nndeploy/nndeploy-workflow), allowing you to experience and use various AI functions immediately:

| Application Scenario | Available Models | 
|----------------------|------------------|
| **Image Classification** | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet |
| **Object Detection** | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx** | 
| **Object Tracking** | FairMot | 
| **Image Segmentation** | RBMGv1.4, PPMatting, **Segment Anything** |
| **Generative Models** | **Stable Diffusion 1.5** | 
| **Large Language Models (LLMs)** | **QWen-0.5B** | 
| **Face Swapping** | **deep-live-cam** | 

### YOLO: Visual Parameter Tuning & One-Click Deployment
Adjust detection parameters in real time in the visual interface—observe effect changes without modifying code. Supports one-click switching to inference engines like TensorRT for high-performance deployment.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="yolo_edit_param.gif">
    <img alt="nndeploy" src="docs/image/yolo_edit_deploy.gif" width=100%>
  </picture>
</p>

### Multi-Model Workflow Demo
Visually build a workflow combining detection + segmentation + classification. Supports switching between multiple inference frameworks and parallel modes, enabling "build once, deploy across multiple devices."

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="seg_detect_class.gif">
    <img alt="nndeploy" src="docs/image/seg_detect_class.gif" width=100%>
  </picture>
</p>

### No-Code Face Swap + Segmentation Workflow
Combine AI functions such as face detection, face swapping, and portrait segmentation via drag-and-drop—no code required. See parameter adjustment results in 1–2 seconds. Empowers **product managers, designers, and non-AI developers** to quickly turn ideas into prototypes.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="face_swap_seg.gif">
    <img alt="nndeploy" src="docs/image/face_swap_seg.gif" width=100%>
  </picture>
</p>

> We will continue to develop new nodes. If you have algorithms that need deployment, please let us know via [issues](https://github.com/nndeploy/nndeploy/issues).


## Stay Updated
Give nndeploy a Star on GitHub to receive notifications about new versions immediately.

<img src="docs/image/star.gif">


## Roadmap
- [Workflow Ecosystem](https://github.com/nndeploy/nndeploy/issues/191)
- [On-Device LLM Inference](https://github.com/nndeploy/nndeploy/issues/161)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)
- [Architecture Optimization](https://github.com/nndeploy/nndeploy/issues/189)


## Contact Us
- Welcome to join our technical communication group! WeChat: Always031856 (please briefly introduce yourself ^_^)
  
  <img src="docs/image/wechat.jpg" width="225px">


## Acknowledgements
- Thanks to the following projects: [TNN](https://github.com/Tencent/TNN), [FastDeploy](https://github.com/PaddlePaddle/FastDeploy), [opencv](https://github.com/opencv/opencv), [CGraph](https://github.com/ChunelFeng/CGraph), [CThreadPool](https://github.com/ChunelFeng/CThreadPool), [tvm](https://github.com/apache/tvm), [mmdeploy](https://github.com/open-mmlab/mmdeploy), [FlyCV](https://github.com/PaddlePaddle/FlyCV), [oneflow](https://github.com/Oneflow-Inc/oneflow), [flowgram.ai](https://github.com/bytedance/flowgram.ai), [deep-live-cam](https://github.com/hacksider/Deep-Live-Cam).

- Thanks to [HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy) for the recommendation.

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>


## Contributors
<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)
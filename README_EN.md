[简体中文](README.md) | English

<h3 align="center">
nndeploy：An Easy-to-Use and High-Performance AI Deployment Framework
</h3>

<p align="center">
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

<p align="center">
<a href="https://nndeploy.readthedocs.io/en/latest/"><b>Documentation</b></a> 
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
| <a href="docs/en/knowledge_shared/wechat.md"><b>WeChat</b></a> 
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
<!-- | <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>Zhihu</b></a>  -->
<!-- | <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>Bilibili</b></a>  -->
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow.gif" width=100%>
  </picture>
</p>

---

## Latest Updates

- [2025/05/29]🔥 A free course on the inference framework, co-developed by the nndeploy open-source team and Huawei Ascend officials, is now available. It is suitable for those who want to get started with or improve their AI inference deployment skills. [Learn on Ascend Platform](https://www.hiascend.com/developer/courses/detail/1923211251905150977) | [Learn on Bilibili](https://space.bilibili.com/435543077?spm_id_from=333.788.0.0)

---

## Introduction

nndeploy is a easy-to-use and high-performance AI deployment framework. Built on the design of visual workflows and multi-end inference, it enables developers to easily create SDKs tailored for specific platforms and hardware directly from training algorithm repositories, significantly reducing development time. Additionally, the framework comes pre-deployed with a wide range of AI models, including Large Language Models (LLMs), AIGC generation, face swapping, object detection, and image segmentation, allowing for out-of-the-box use.

In practical applications, we recommend using the visual workflow for design and debugging. After verifying the algorithm's effectiveness and performance, you can leverage the provided Python/C++ APIs to load and run the workflow in production environments. Whether accessed through the visual frontend interface or API calls, the workflow ultimately operates on the underlying high-performance C++ computing engine. This design ensures that the workflow exhibits completely consistent execution behavior and performance in both development/debugging and production deployment environments, achieving the goal of "develop once, run anywhere".
### **Simple and Easy to Use**

- **Visual Workflow**: Deploy AI algorithms through drag-and-drop operations. Adjust all node parameters of AI algorithms via the front-end visualization interface and quickly preview the effect after parameter tuning.
- **Custom Nodes**: Support custom nodes in Python/C++. No front-end code is required, and they can be seamlessly integrated into the visualization interface.
- **Algorithm Combination**: Flexibly combine different algorithms to quickly build innovative AI applications.
- **One-Click Deployment**: The built workflow can be exported as a JSON configuration file with one click. It supports direct calls via Python/C++ APIs, enabling seamless transition from the development environment to the production environment. It fully supports platforms such as Linux, Windows, macOS, Android, and iOS.

### **High Performance**

- **Seamless Integration of 13 Inference Engines**: One workflow for multi-end deployment. It has integrated 13 mainstream inference frameworks with zero abstraction cost, covering full platforms including cloud, desktop, mobile, and edge.

  | Inference Framework                                                                 | Application Scenario          | Status |
  | :----------------------------------------------------------------------------------- | :---------------------------- | :----- |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime)                              | Cross-platform inference      | ✅     |
  | [TensorRT](https://github.com/NVIDIA/TensorRT)                                       | High-performance inference on NVIDIA GPU | ✅     |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino)                              | Optimization for Intel CPU/GPU | ✅     |
  | [MNN](https://github.com/alibaba/MNN)                                                | Mobile inference engine launched by Alibaba | ✅     |
  | [TNN](https://github.com/Tencent/TNN)                                                | Mobile inference engine launched by Tencent | ✅     |
  | [ncnn](https://github.com/Tencent/ncnn)                                              | Mobile inference engine launched by Tencent | ✅     |
  | [CoreML](https://github.com/apple/coremltools)                                       | Native acceleration for iOS/macOS | ✅     |
  | [AscendCL](https://www.hiascend.com/en/)                                             | Inference framework for Huawei Ascend AI chips | ✅     |
  | [RKNN](https://www.rock-chips.com/a/en/downloadcenter/BriefDatasheet/index.html)     | Inference framework for Rockchip NPU | ✅     |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)       | Inference framework for Qualcomm Snapdragon NPU | ✅     |
  | [TVM](https://github.com/apache/tvm)                                                 | Deep learning compilation stack | ✅     |
  | [PyTorch](https://pytorch.org/)                                                      | Rapid prototyping/cloud deployment | ✅     |
  | [Self-developed Inference Framework](docs/en/inference/README_INFERENCE.md)          | Default inference framework   | ✅     |

- **Parallel Optimization**: Support execution modes such as serial, pipeline parallelism, and task parallelism.
- **Memory Optimization**: Optimization strategies including zero-copy, memory pool, and memory reuse.
- **High-performance Optimization**: Built-in nodes optimized with C++/CUDA/Ascend C/SIMD implementations.

### **Out-of-the-Box Algorithms**

A list of deployed models has been created, with **100+ nodes** available. We will continue to deploy more high-value AI algorithms. If you have algorithms that need to be deployed, please let us know via [issues](https://github.com/nndeploy/nndeploy/issues).

| Application Scenario | Available Models                                                                 | Notes                                              |
| -------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Large Language Models** | **QWen-0.5B**                                                                    |                                                   |
| **Image Generation** | Stable Diffusion 1.5, Stable Diffusion XL, Stable Diffusion 3, HunyuanDiT, etc.   | Support text-to-image, image-to-image, and image inpainting, implemented based on **diffusers** |
| **Face Swapping**    | **deep-live-cam**                                                                |                                                   |
| **OCR**              | **Paddle OCR**                                                                   |                                                   |
| **Object Detection** | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx**                               |                                                   |
| **Object Tracking**  | FairMot                                                                          |                                                   |
| **Image Segmentation** | RBMGv1.4, PPMatting, **Segment Anything**                                        |                                                   |
| **Classification**   | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet       |                                                   |
| **API Services**     | OPENAI, DeepSeek, Moonshot                                                       | Support LLM and AIGC services                     |

> For more details, see the [List of Deployed Models](docs/en/quick_start/model_list.md)

## Performance Comparison

Test Environment: Ubuntu 22.04, CPU: 12th Gen Intel(R) Core(TM) i7-12700, GPU: RTX3060

### Pipeline Parallel Acceleration

Taking the end-to-end workflow of YOLOv11s as an example, the comparison of end-to-end time consumption is as follows:

![yolov11s_performance](docs/image/workflow/yolo_performance.png)

| Running Mode \ Inference Engine | ONNXRuntime | OpenVINO  | TensorRT  |
| -------------------------------- | ----------- | --------- | --------- |
| Serial                           | 54.803 ms   | 34.139 ms | 13.213 ms |
| Pipeline Parallelism             | 47.283 ms   | 29.666 ms | 5.681 ms  |
| Performance Improvement          | 13.7%       | 13.1%     | 57%       |

### Task Parallel Acceleration

End-to-end total time consumption of combined tasks (segmentation RMBGv1.4 + detection YOLOv11s + classification ResNet50), Serial vs. Task Parallelism:

![rmbg_yolo_resnet.png](docs/image/workflow/rmbg_yolo_resnet.png)

| Running Mode \ Inference Engine | ONNXRuntime | OpenVINO   | TensorRT  |
| -------------------------------- | ----------- | ---------- | --------- |
| Serial                           | 654.315 ms  | 489.934 ms | 59.140 ms |
| Task Parallelism                 | 602.104 ms  | 435.181 ms | 51.883 ms |
| Performance Improvement          | 7.98%       | 11.2%      | 12.2%     |

## Quick Start

+ **Installation**

  ```bash
  pip install --upgrade nndeploy
  ```

+ **Launch the Visual Workflow**

  ```bash
  nndeploy-app --port 8000
  ```

  After successful launch, open http://localhost:8000 to access the workflow interface

  <p align="left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="quick_start.gif">
      <img alt="nndeploy" src="docs/image/quick_start.gif" width=100%>
    </picture>
  </p>  

### Export and Execute the Workflow

  After building the workflow in the visualization interface, save it as a JSON file (e.g., workflow.json), then execute the workflow using the following command:

  ```bash
  # Python CLI
  nndeploy-run-json --json_file path/to/workflow.json
  # C++ CLI
  nndeploy_demo_run_json --json_file path/to/workflow.json
  ```

  - API to Load and Run JSON Configuration Files
    - [Python API Example Code](python/nndeploy/dag/run_json.py)
    - [Python 检测算法示例代码](demo/detect/demo.cc)
    - [C++ API Example Code](framework/include/nndeploy/dag/graph_runner.h)
    - [C++ 检测算法示例代码](demo/detect/demo.py)

> Python 3.10 or higher is required. By default, two inference backends (PyTorch and ONNXRuntime) are included. To use more inference backends (such as TensorRT, OpenVINO, ncnn, MNN, etc.), please use the developer mode.

> Use `nndeploy-clean` to clean up outdated backend resources.

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

## Stay Ahead

Give nndeploy a Star on GitHub to get notified of new releases immediately.

<img src="docs/image/star.gif">

## Roadmap

- [Workflow Ecosystem](https://github.com/nndeploy/nndeploy/issues/191)
- [On-Device LLM Inference](https://github.com/nndeploy/nndeploy/issues/161)
- [Architecture Optimization](https://github.com/nndeploy/nndeploy/issues/189)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)

## Contact Us

- Currently, nndeploy is in the development stage. If you love open-source projects and enjoy exploring, whether for learning purposes or to share better ideas, you are welcome to join us.

- WeChat: Always031856 (Feel free to add as a friend and join the active AI inference deployment communication group. Please note: nndeploy\_Your Name)

  <img src="docs/image/wechat.jpg" width="225px">

## Acknowledgements

- Thanks to the following projects: [TNN](https://github.com/Tencent/TNN), [FastDeploy](https://github.com/PaddlePaddle/FastDeploy), [opencv](https://github.com/opencv/opencv), [CGraph](https://github.com/ChunelFeng/CGraph), [tvm](https://github.com/apache/tvm), [mmdeploy](https://github.com/open-mmlab/mmdeploy), [FlyCV](https://github.com/PaddlePaddle/FlyCV), [oneflow](https://github.com/Oneflow-Inc/oneflow), [flowgram.ai](https://github.com/bytedance/flowgram.ai), [deep-live-cam](https://github.com/hacksider/Deep-Live-Cam).

- Thanks to [HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy) for the recommendation

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## Contributors

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)
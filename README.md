
[English](README_EN.md) | 简体中文

<h3 align="center">
nndeploy：一款基于工作流的多端AI部署工具
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

<!-- <p align="center">
<a href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"><b>文档</b></a> 
| <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>公众号</b></a> 
| <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>知乎</b></a> 
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
| <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>哔哩哔哩</b></a> 
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
</p> -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow.gif" width=100%>
  </picture>
</p>

框架提供了丰富的开箱即用AI算法节点，涵盖目标检测、图像分割、大语言模型、换脸、图像生成等等，用户只需通过直观的拖拽操作即可完成复杂AI算法的部署。

支持Python/C++编写自定义算法节点，无需掌握前端技术即可将算法无缝集成到可视化工作流中。

工作流可导出为JSON配置文件，支持Python/C++ API直接加载运行，部署至云服务器、桌面端、移动端、边缘设备等多平台。

框架内置主流高性能推理引擎和深度优化策略，助你将工作流转化为企业级生产应用。

---

<!-- ## 最新动态
- [2025/08/01] 🔥 **可视化工作流**：告别复杂的代码编写！通过简单的拖拽操作，你就能快速搭建专业的AI应用。无论你是技术小白还是经验丰富的开发者，都能在几分钟内完成AI算法的部署。[立即体验丰富的工作流模板：换脸、LLM对话、AI绘画、目标检测、图像分割等应用](https://github.com/nndeploy/nndeploy-workflow)
- [2025/07/20] 🔥 **Python API**：只需几行代码，就能将你的AI模型部署到手机、电脑、服务器等各种设备上。更棒的是，还支持工作流可视化展示，让你在团队汇报时轻松展示令人惊艳的AI效果，瞬间成为焦点！[点击查看超简单的入门教程，5分钟上手](https://nndeploy-zh.readthedocs.io/zh-cn/latest/quick_start/python.html)
- [2025/05/29] 🔥 **免费AI推理课程**：想要在AI推理部署领域找到更好的工作机会？我们基于nndeploy框架打造了业内最全面的AI推理部署课程，深度覆盖模型中间表示、模型解释、计算图构建、图优化、内存优化、并行优化和算子优化等企业核心技术需求。这门课程都将成为你职业发展的强力助推器。[昇腾平台免费学习](https://www.hiascend.com/developer/courses/detail/1923211251905150977) | [B站同步更新](https://space.bilibili.com/435543077?spm_id_from=333.788.0.0) -->

<!-- --- -->

## 快速开始

### 安装

```bash
pip install --upgrade nndeploy
```

### 启动可视化工作流界面

```bash
# 方法一：仅使用内置节点
nndeploy-app --port 8000

# 方法二：使用用户自定义节点
nndeploy-app --port 8000 --plugin plugin1.py plugin2.py 
```

- 命令参数说明
  - `--port`：指定Web服务端口号（默认为8000）
  - `--plugin`：加载用户自定义插件文件（可选参数，如果没有该参数，仅使用内置节点）
    - Python插件：参考[Python插件模板写法](template/python/template.py)
    - C++插件：参考[C++插件模板写法](template/cpp/template.h)
    - 可以同时加载多个插件：`--plugin plugin1.py plugin2.so`

启动成功后，打开 http://localhost:8000 即可访问工作流界面。

#### 快速上手演示

通过拖拽操作快速搭建AI工作流，直观易懂，几分钟即可上手。

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="quick_start.gif">
    <img alt="nndeploy" src="docs/image/quick_start.gif" width=100%>
  </picture>
</p>


### 保存工作流为JSON和执行工作流

在可视化界面中配置好工作流后，可将其保存为JSON文件（例如workflow.json）。您可以使用以下命令执行该工作流：

```bash
nndeploy-run-json --json-file workflow.json --plugin plugin.py
```

- API加载运行JSON配置文件
  - [Python API示例代码](python/nndeploy/dag/run_json.py)
  - [C++ API示例代码](framework/include/nndeploy/dag/graph_runner.h)

> 需要 Python 3.10 及以上版本。默认包含 PyTorch 和 ONNXRuntime 两个推理后端。如需使用更多推理后端（如 TensorRT、OpenVINO、ncnn、MNN 等），请采用开发者模式

> 使用`nndeploy-clean`可清理过期的后端资源。

### 文档
- [如何构建](docs/zh_cn/quick_start/build.md)
- [如何获取模型](docs/zh_cn/quick_start/model.md)
- [如何执行](docs/zh_cn/quick_start/example.md)
- [Python快速开始](docs/zh_cn/quick_start/python.md)
- [可视化工作流快速开始](docs/zh_cn/quick_start/workflow.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [C++插件开发手册](docs/zh_cn/quick_start/plugin.md)
- [Python++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)
- [Python插件开发手册](docs/zh_cn/quick_start/plugin_python.md)

## 核心特性

### **AI部署的效率工具**
- **可视化工作流**：通过拖拉拽操作就能部署AI算法，前端可视化调节AI算法部署的所有节点参数，快速预览算法调参后的效果
- **自定义节点**：支持Python/C++自定义节点，无需前端代码，无缝集成到可视化界面
- **算法组合**：灵活组合不同算法，快速构建创新AI应用
- **一键部署**：搭建好的工作流可导出为JSON，Python/C++直接调用，从开发到生产环境无缝衔接

### **AI部署的性能工具**
- **13种推理引擎无缝集成**：一套工作流，多端部署。通过零抽象成本接入了13种主流推理框架，覆盖云端、桌面、移动、边缘等全平台

  | 推理框架 | 适用场景 | 状态 |
  | :------- | :------ | :--- |
  | [PyTorch](https://pytorch.org/) | 研发调试、快速原型 | ✅ |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime) | 跨平台推理 | ✅ |
  | [TensorRT](https://github.com/NVIDIA/TensorRT) | NVIDIA GPU高性能推理 | ✅ |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino) | Intel CPU/GPU优化 | ✅ |
  | [MNN](https://github.com/alibaba/MNN) | 阿里推出的移动端推理引擎 | ✅ |
  | [TNN](https://github.com/Tencent/TNN) | 腾讯推出的移动端推理引擎 | ✅ |
  | [ncnn](https://github.com/Tencent/ncnn) | 腾讯推出的移动端推理引擎 | ✅ |
  | [CoreML](https://github.com/apple/coremltools) | iOS/macOS原生加速 | ✅ |
  | [AscendCL](https://www.hiascend.com/zh/) | 华为昇腾AI芯片推理框架 | ✅ |
  | [RKNN](https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html) | 瑞芯微NPU推理框架 | ✅ |
  | [SNPE](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) | 高通骁龙NPU推理框架 | ✅ |
  | [TVM](https://github.com/apache/tvm) | 深度学习编译栈 | ✅ |
  | [自研推理框架](docs/zh_cn/inference/README_INFERENCE.md) | 定制化推理需求 | ✅ |

- **并行优化**：支持串行、流水线并行、任务并行等执行模式
- **内存优化**：零拷贝、内存池、内存复用等优化策略
- **高性能优化**：内置C++/CUDA/Ascend C/SIMD等优化实现的节点

## 开箱即用的节点

针对以下AI算法，开发了40+节点，并制作了[工作流模板](https://github.com/nndeploy/nndeploy-workflow)，让你能够立即体验和使用各种AI功能：

| 应用场景 | 可用模型 | 
|---------|---------|
| **图像分类** | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet |
| **目标检测** | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx** | 
| **目标追踪** | FairMot | 
| **图像分割** | RBMGv1.4, PPMatting, **Segment Anything** |
| **生成模型** | **Stable Diffusion 1.5** | 
| **大语言模型** | **QWen-0.5B** | 
| **换脸** | **deep-live-cam** | 

### YOLO可视化调参与一键部署

可视化界面实时调整检测参数，无需修改代码即可观察效果变化，支持一键切换到TensorRT等推理引擎实现高性能部署。

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="yolo_edit_param.gif">
    <img alt="nndeploy" src="docs/image/yolo_edit_deploy.gif" width=100%>
  </picture>
</p>

### 多模型工作流演示

可视化搭建检测+分割+分类工作流，支持多推理框架切换和并行模式，实现一次搭建、多端部署。

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="seg_detect_class.gif">
    <img alt="nndeploy" src="docs/image/seg_detect_class.gif" width=100%>
  </picture>
</p>

### 零代码搭建换脸+分割工作流

通过拖拽操作组合人脸检测、换脸算法、人像分割等AI功能，无需编写代码，参数调整1-2秒看到效果。让**产品经理、设计师、非AI开发者**快速将创意变成原型。

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="face_swap_seg.gif">
    <img alt="nndeploy" src="docs/image/face_swap_seg.gif" width=100%>
  </picture>
</p>

> 会持续开发的节点，如果你有需要部署的算法，请通过[issue](https://github.com/nndeploy/nndeploy/issues)告诉我们

## 保持领先

在 GitHub 上给 nndeploy Star，并立即收到新版本的通知。

<img src="docs/image/star.gif">

## 下一步计划

- [工作流生态](https://github.com/nndeploy/nndeploy/issues/191)
- [端侧大模型推理](https://github.com/nndeploy/nndeploy/issues/161)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)
- [架构优化](https://github.com/nndeploy/nndeploy/issues/189)

## 联系我们
- 欢迎加入技术交流群！微信：Always031856（请简单备注个人信息^_^）
  
  <img src="docs/image/wechat.jpg" width="225px">

## 致谢

- 感谢以下项目：[TNN](https://github.com/Tencent/TNN)、[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)、[opencv](https://github.com/opencv/opencv)、[CGraph](https://github.com/ChunelFeng/CGraph)、[CThreadPool](https://github.com/ChunelFeng/CThreadPool)、[tvm](https://github.com/apache/tvm)、[mmdeploy](https://github.com/open-mmlab/mmdeploy)、[FlyCV](https://github.com/PaddlePaddle/FlyCV)、[oneflow](https://github.com/Oneflow-Inc/oneflow)、[flowgram.ai](https://github.com/bytedance/flowgram.ai)、[deep-live-cam](https://github.com/hacksider/Deep-Live-Cam)。

- 感谢[HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy)推荐

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## 贡献者

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)

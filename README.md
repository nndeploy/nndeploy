[English](README_EN.md) | 简体中文

<h3 align="center">
nndeploy：一款简单易用和高性能的AI部署框架
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
 <!-- <a href="https://pepy.tech/projects/nndeploy">
  <img src="https://static.pepy.tech/personalized-badge/nndeploy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads" alt="PyPI Downloads" style="height: 16px;">
</a> -->
</p>

<p align="center">
<a href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"><b>文档</b></a> 
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
<!-- | <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>微信</b></a>  -->
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
<!-- | <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>知乎</b></a>  -->
<!-- | <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>哔哩哔哩</b></a>  -->
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow/worflow_llm.gif" width=100%>
  </picture>
</p>

---

## 最新动态

- [2025/01/29] 🔥 与昇腾联合推出 AI 推理框架免费课程！[昇腾官方](https://www.hiascend.com/developer/courses/detail/1923211251905150977) | [B 站视频](https://space.bilibili.com/435543077?spm_id_from=333.788.0.0) 从零开始，系统掌握AI推理框架的最全教程。

---

## 介绍

nndeploy 是一款简单易用和高性能的 AI 部署框架。基于可视化工作流和多端推理，开发者可以快速从算法仓库开发出指定平台和硬件所需的 SDK，大幅节省开发时间。此外，框架已部署包括 LLM、AIGC 生成、换脸、目标检测、图像分割等 AI 模型，开箱即用。

### **简单易用**

- **可视化工作流**：拖拽节点即可部署 AI 算法，参数实时可调，效果一目了然。
- **自定义节点**：支持 Python/C++自定义节点，无论是用 Python 实现预处理，还是用 C++/CUDA 编写高性能节点，均可无缝集成到与可视化工作流。
- **一键部署**：工作流支持导出为 JSON，可通过 C++/Python API 调用，适用于 Linux、Windows、macOS、Android 等平台

  <table cellpadding="5" cellspacing="0" border="1">
  <tr>
    <td>桌面端搭建AI工作流</td>
    <td><a href="https://github.com/nndeploy/nndeploy/blob/main/app/android/README.md">移动端部署</a></td>
  </tr>
  <tr>
    <td><img src="docs/image/workflow/worflow_segment_rmbg.gif" width="500px"></td>
    <td><img src="docs/image/android_app/app-seg-result.jpg" width="100px"></td>
  </tr>
  </table>

### **高性能**

- **并行优化**：支持串行、流水线并行、任务并行等执行模式
- **内存优化**：零拷贝、内存池、内存复用等优化策略
- **高性能优化**：内置 C++/CUDA/Ascend C/SIMD 等优化实现的节点
- **多端推理**：一套工作流，多端推理，接入 13 种主流推理框架，覆盖云端、桌面、移动、边缘等全平台部署场景。

  <table cellpadding="5" cellspacing="0" border="1">
  <tr>
    <td><a href="https://github.com/microsoft/onnxruntime">ONNXRuntime</a></td>
    <td><a href="https://github.com/NVIDIA/TensorRT">TensorRT</a></td>
    <td><a href="https://github.com/openvinotoolkit/openvino">OpenVINO</a></td>
    <td><a href="https://github.com/alibaba/MNN">MNN</a></td>
    <td><a href="https://github.com/Tencent/TNN">TNN</a></td>
    <td><a href="https://github.com/Tencent/ncnn">ncnn</a></td>
    <td><a href="https://github.com/apple/coremltools">CoreML</a></td>
    <td><a href="https://www.hiascend.com/zh/">AscendCL</a></td>
    <td><a href="https://www.rock-chips.com/a/cn/downloadcenter/BriefDatasheet/index.html">RKNN</a></td>
    <td><a href="https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk">SNPE</a></td>
    <td><a href="https://github.com/apache/tvm">TVM</a></td>
    <td><a href="https://pytorch.org/">PyTorch</a></td>
    <td><a href="docs/zh_cn/inference/README_INFERENCE.md">nndeploy_inner</a></td>
  </tr>
  <tr>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  </table>

  > 支持灵活的框架选择，可根据需求选择性编译。如果有自定义推理框架，可完全不依赖其他推理框架独立运行

### **开箱即用的算法**

已部署模型列表，并制作了 100+可视化节点，让你开箱即用。

| 应用场景       | 可用模型                                                                         | 备注                                                |
| -------------- | -------------------------------------------------------------------------------- | --------------------------------------------------- |
| **大语言模型** | **QWen-2.5**, **QWen-3**                                                         | 支持小 B 模型                                       |
| **图片生成**   | Stable Diffusion 1.5, Stable Diffusion XL, Stable Diffusion 3, HunyuanDiT 等模型 | 支持文生图、图生图、图像修复，基于**diffusers**实现 |
| **换脸**       | **deep-live-cam**                                                                |                                                     |
| **OCR**        | **Paddle OCR**                                                                   |                                                     |
| **目标检测**   | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx**                               |                                                     |
| **目标追踪**   | FairMot                                                                          |                                                     |
| **图像分割**   | RBMGv1.4, PPMatting, **Segment Anything**                                        |                                                     |
| **分类**       | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet       |                                                     |
| **API 服务**   | OPENAI, DeepSeek, Moonshot                                                       | 支持 LLM 和 AIGC 服务                               |

> 更多查看[已部署模型列表详解](docs/zh_cn/quick_start/model_list.md)

## 快速开始

- **步骤一：安装**

  ```bash
  pip install --upgrade nndeploy
  ```

- **步骤二：启动可视化界面**

  ```bash
  # 方式一：命令行
  nndeploy-app --port 8000
  # 方式二：代码启动
  cd path/to/nndeploy
  python app.py --port 8000
  ```

  启动成功后，打开 http://localhost:8000 即可访问工作流编辑器。在这里，你可以拖拽节点、调整参数、实时预览效果，所见即所得。

  <p align="left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="quick_start.gif">
      <img alt="nndeploy" src="docs/image/quick_start.gif" width=100%>
    </picture>
  </p>

- **步骤三：保存并加载运行**

  在可视化界面中搭建、调试完成后，点击保存，工作流会导出 JSON 文件，文件中封装了所有的处理流程。你可以用以下两种方式在**生产环境**中运行：

  - 方式一：命令行运行

    用于调试

    ```bash
    # Python CLI
    nndeploy-run-json --json_file path/to/workflow.json
    # C++ CLI
    nndeploy_demo_run_json --json_file path/to/workflow.json
    ```

  - 方式 2：在 Python/C++ 代码中加载运行

    可以将 JSON 文件集成到你现有的 Python 或 C++ 项目中，以下是一个加载和运行 LLM 工作流的示例代码：

    - Python API 加载运行 LLM 工作流
      ```Python
      graph = nndeploy.dag.Graph("")
      graph.remove_in_out_node()
      graph.load_file("path/to/llm_workflow.json")
      graph.init()
      input = graph.get_input(0)
      text = nndeploy.tokenizer.TokenizerText()
      text.texts_ = [ "<|im_start|>user\nPlease introduce NBA superstar Michael Jordan<|im_end|>\n<|im_start|>assistant\n" ]
      input.set(text)
      status = graph.run()
      output = graph.get_output(0)
      result = output.get_graph_output()
      graph.deinit()
      ```
    - C++ API 加载运行 LLM 工作流
      ```C++
      std::shared_ptr<dag::Graph> graph = std::make_shared<dag::Graph>("");
      base::Status status = graph->loadFile("path/to/llm_workflow.json");
      graph->removeInOutNode();
      status = graph->init();
      dag::Edge* input = graph->getInput(0);
      tokenizer::TokenizerText* text = new tokenizer::TokenizerText();
      text->texts_ = {
          "<|im_start|>user\nPlease introduce NBA superstar Michael Jordan<|im_end|>\n<|im_start|>assistant\n"};
      input->set(text, false);
      status = graph->run();
      dag::Edge* output = graph->getOutput(0);
      tokenizer::TokenizerText* result =
          output->getGraphOutput<tokenizer::TokenizerText>();
      status = graph->deinit();
      ```

> 要求 Python 3.10+，默认包含 ONNXRuntime、MNN，更多推理后端请采用开发者模式。

## 文档

- [如何构建](docs/zh_cn/quick_start/build.md)
- [如何获取模型](docs/zh_cn/quick_start/model.md)
- [可视化工作流](docs/zh_cn/quick_start/workflow.md)
- [生产环境部署](docs/zh_cn/quick_start/deploy.md)
- [Python API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/python_api/index.html)
- [Python 自定义节点开发手册](docs/zh_cn/quick_start/plugin_python.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/zh-cn/latest/cpp_api/doxygen.html)
- [C++自定义节点开发手册](docs/zh_cn/quick_start/plugin.md)
- [部署新算法](docs/zh_cn/quick_start/ai_deploy.md)
- [接入新推理框架](docs/zh_cn/developer_guide/how_to_support_new_inference.md)

## 性能测试

测试环境：Ubuntu 22.04，i7-12700，RTX3060

- **流水线并行加速**。以 YOLOv11s 端到端工作流总耗时，串行 vs 流水线并行

  <img src="docs/image/workflow/yolo_performance.png" width="60%">

  | 运行方式\推理引擎 | ONNXRuntime | OpenVINO  | TensorRT  |
  | ----------------- | ----------- | --------- | --------- |
  | 串行              | 54.803 ms   | 34.139 ms | 13.213 ms |
  | 流水线并行        | 47.283 ms   | 29.666 ms | 5.681 ms  |
  | 性能提升          | 13.7%       | 13.1%     | 57%       |

- **任务并行加速**。组合任务(分割 RMBGv1.4+检测 YOLOv11s+分类 ResNet50)的端到端总耗时，串行 vs 任务并行

  <img src="docs/image/workflow/rmbg_yolo_resnet.png" width="60%">

  | 运行方式\推理引擎 | ONNXRuntime | OpenVINO   | TensorRT  |
  | ----------------- | ----------- | ---------- | --------- |
  | 串行              | 654.315 ms  | 489.934 ms | 59.140 ms |
  | 任务并行          | 602.104 ms  | 435.181 ms | 51.883 ms |
  | 性能提升          | 7.98%       | 11.2%      | 12.2%     |

## 下一步计划

- [工作流生态](https://github.com/nndeploy/nndeploy/issues/191)
- [端侧大模型推理](https://github.com/nndeploy/nndeploy/issues/161)
- [架构优化](https://github.com/nndeploy/nndeploy/issues/189)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)

## 联系我们

- 如果你热爱开源、喜欢折腾，不论是出于学习目的，亦或是有更好的想法，欢迎加入我们

- 微信：Always031856（欢迎加好友，进群交流，备注：nndeploy\_姓名）

## 致谢

- 感谢以下项目：[TNN](https://github.com/Tencent/TNN)、[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)、[opencv](https://github.com/opencv/opencv)、[CGraph](https://github.com/ChunelFeng/CGraph)、[tvm](https://github.com/apache/tvm)、[mmdeploy](https://github.com/open-mmlab/mmdeploy)、[FlyCV](https://github.com/PaddlePaddle/FlyCV)、[oneflow](https://github.com/Oneflow-Inc/oneflow)、[flowgram.ai](https://github.com/bytedance/flowgram.ai)、[deep-live-cam](https://github.com/hacksider/Deep-Live-Cam)。

- 感谢[HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy)推荐

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## 贡献者

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)

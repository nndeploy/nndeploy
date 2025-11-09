[简体](README.md) | English

<h3 align="center">
nndeploy: An Easy-to-Use and High-Performance AI deployment framework
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
<a href="https://nndeploy-zh.readthedocs.io/en/latest/"><b>Documentation</b></a> 
| <a href="https://deepwiki.com/nndeploy/nndeploy"><b>Ask DeepWiki</b></a>
<!-- | <a href="docs/zh_cn/knowledge_shared/wechat.md"><b>WeChat</b></a>  -->
| <a href="https://discord.gg/9rUwfAaMbr"><b>Discord</b></a> 
<!-- | <a href="https://www.zhihu.com/column/c_1690464325314240512"><b>Zhihu</b></a>  -->
<!-- | <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"><b>Bilibili</b></a>  -->
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/image/workflow.png">
    <img alt="nndeploy" src="docs/image/workflow/worflow_llm.gif" width=100%>
  </picture>
</p>

---

## Latest Updates

- [2025/05/29] 🔥 Jointly launched a free inference framework course with Huawei Ascend official [Ascend Official](https://www.hiascend.com/developer/courses/detail/1923211251905150977) | [Bilibili Video](https://space.bilibili.com/435543077?spm_id_from=333.788.0.0)! Based on nndeploy's internal inference sub-module, helping you quickly master core AI inference deployment technologies.

---

## Introduction

nndeploy is an easy-to-use and high-performance AI deployment framework. Based on visual workflows and multi-end inference, developers can quickly develop SDKs for specified platforms and hardware from algorithm repositories, significantly saving development time. In addition, the framework has deployed numerous AI models including LLM, AIGC generation, face swapping, object detection, image segmentation, etc., which are ready to use out of the box.

### **Easy to Use**

- **Visual Workflow**: Deploy AI algorithms by dragging nodes, with real-time adjustable parameters and intuitive effects.
- **Custom Nodes**: Support Python/C++ custom nodes. Whether implementing preprocessing in Python or writing high-performance nodes in C++/CUDA, they can be seamlessly integrated into the visual workflow.
- **One-Click Deployment**: Workflows can be exported as JSON and called through C++/Python APIs, applicable to platforms such as Linux, Windows, macOS, Android, and iOS.

  <table cellpadding="5" cellspacing="0" border="1">
  <tr>
    <td>Building AI Workflow on Desktop</td>
    <td><a href="https://github.com/nndeploy/nndeploy/blob/main/app/android/README.md">Deployment on Mobile</a></td>
  </tr>
  <tr>
    <td><img src="docs/image/workflow/worflow_segment_rmbg.gif" width="500px"></td>
    <td><img src="docs/image/android_app/app-seg-result.jpg" width="100px"></td>
  </tr>
  </table>

### **High Performance**

- **Parallel Optimization**: Support execution modes such as serial, pipeline parallelism, and task parallelism.
- **Memory Optimization**: Zero-copy, memory pool, memory reuse and other optimization strategies.
- **High-Performance Optimization**: Built-in nodes optimized with C++/CUDA/Ascend C/SIMD implementations.
- **Multi-End Inference**: One workflow for multi-end inference, integrating 13 mainstream inference frameworks, covering full-platform deployment scenarios such as cloud, desktop, mobile, and edge. 

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

  > If there is a custom inference framework, it can be used completely independently without relying on any third-party frameworks.

### **Out-of-the-Box Algorithms**

A list of deployed models with over 100+ visual nodes.

| Application Scenario | Available Models                                                                    | Remarks                                                  |
| -------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Large Language Models** | **QWen-2.5**, **QWen-3**                                                           | Support small B models                                   |
| **Image Generation** | Stable Diffusion 1.5, Stable Diffusion XL, Stable Diffusion 3, HunyuanDiT, etc.    | Support text-to-image, image-to-image, image inpainting, based on **diffusers** |
| **Face Swapping**    | **deep-live-cam**                                                                  |                                                          |
| **OCR**              | **Paddle OCR**                                                                      |                                                          |
| **Object Detection** | **YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv11, YOLOx**                                 |                                                          |
| **Object Tracking**  | FairMot                                                                            |                                                          |
| **Image Segmentation** | RBMGv1.4, PPMatting, **Segment Anything**                                          |                                                          |
| **Classification**   | ResNet, MobileNet, EfficientNet, PPLcNet, GhostNet, ShuffleNet, SqueezeNet          |                                                          |
| **API Services**     | OPENAI, DeepSeek, Moonshot                                                         | Support LLM and AIGC services                            |

> For more, see [Detailed List of Deployed Models](docs/zh_cn/quick_start/model_list.md)

## Quick Start

- **Step 1: Installation**

  ```bash
  pip install --upgrade nndeploy
  ```

- **Step 2: Launch the Visual Interface**

  ```bash
  # Method 1: Command line
  nndeploy-app --port 8000
  # Method 2: Code startup
  cd path/to/nndeploy
  python app.py --port 8000
  ```

  After successful launch, open http://localhost:8000 to access the workflow editor. Here, you can drag nodes, adjust parameters, and preview effects in real-time, with a what-you-see-is-what-you-get experience.

  <p align="left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="quick_start.gif">
      <img alt="nndeploy" src="docs/image/quick_start.gif" width=100%>
    </picture>
  </p>


- **Step 3: Save and Load for Execution**

  After building and debugging in the visual interface, click save, and the workflow will be exported as a JSON file, which encapsulates all processing procedures. You can run it in the **production environment** in the following two ways:

  - Method 1: Command-line execution

    For debugging

    ```bash
    # Python CLI
    nndeploy-run-json --json_file path/to/workflow.json
    # C++ CLI
    nndeploy_demo_run_json --json_file path/to/workflow.json
    ```

  - Method 2: Load and run in Python/C++ code

    You can integrate the JSON file into your existing Python or C++ project. Here is an example code for loading and running an LLM workflow:

    - Python API to load and run LLM workflow
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
    - C++ API to load and run LLM workflow
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

> Requires Python 3.10+. By default, it includes ONNXRuntime, and MNN. For more inference backends, please use developer mode.

## Documentation

- [How to Build](docs/zh_cn/quick_start/build.md)
- [How to Obtain Models](docs/zh_cn/quick_start/model.md)
- [Visual Workflow](docs/zh_cn/quick_start/workflow.md)
- [Production Environment Deployment](docs/zh_cn/quick_start/deploy.md)
- [Python API](https://nndeploy-zh.readthedocs.io/en/latest/python_api/index.html)
- [Python Custom Node Development Guide](docs/zh_cn/quick_start/plugin_python.md)
- [C++ API](https://nndeploy-zh.readthedocs.io/en/latest/cpp_api/doxygen.html)
- [C++ Custom Node Development Guide](docs/zh_cn/quick_start/plugin.md)
- [Deploy New Algorithms](docs/zh_cn/quick_start/ai_deploy.md)
- [Integrate New Inference Frameworks](docs/zh_cn/developer_guide/how_to_support_new_inference.md)

## Performance Testing

Test environment: Ubuntu 22.04, i7-12700, RTX3060

- **Pipeline parallel acceleration**. End-to-end workflow total time for YOLOv11s, serial vs pipeline parallel

  <img src="docs/image/workflow/yolo_performance.png" width="60%">

  | Execution Mode \ Inference Engine | ONNXRuntime | OpenVINO  | TensorRT  |
  | --------------------------------- | ----------- | --------- | --------- |
  | Serial                            | 54.803 ms   | 34.139 ms | 13.213 ms |
  | Pipeline Parallel                 | 47.283 ms   | 29.666 ms | 5.681 ms  |
  | Performance Improvement           | 13.7%       | 13.1%     | 57%       |

- **Task parallel acceleration**. End-to-end total time for combined tasks (segmentation RMBGv1.4 + detection YOLOv11s + classification ResNet50), serial vs task parallel

  <img src="docs/image/workflow/rmbg_yolo_resnet.png" width="60%">

  | Execution Mode \ Inference Engine | ONNXRuntime | OpenVINO   | TensorRT  |
  | --------------------------------- | ----------- | ---------- | --------- |
  | Serial                            | 654.315 ms  | 489.934 ms | 59.140 ms |
  | Task Parallel                     | 602.104 ms  | 435.181 ms | 51.883 ms |
  | Performance Improvement           | 7.98%       | 11.2%      | 12.2%     |



## Roadmap

- [Workflow Ecosystem](https://github.com/nndeploy/nndeploy/issues/191)
- [Edge Large Model Inference](https://github.com/nndeploy/nndeploy/issues/161)
- [Architecture Optimization](https://github.com/nndeploy/nndeploy/issues/189)
- [AI Box](https://github.com/nndeploy/nndeploy/issues/190)

## Contact Us

- If you love open source and enjoy tinkering, whether for learning purposes or to share better ideas, you are welcome to join us.

- WeChat: Always031856 (Feel free to add as a friend to join the group discussion. Please note: nndeploy_name)

## Acknowledgements

- Thanks to the following projects: [TNN](https://github.com/Tencent/TNN), [FastDeploy](https://github.com/PaddlePaddle/FastDeploy), [opencv](https://github.com/opencv/opencv), [CGraph](https://github.com/ChunelFeng/CGraph), [tvm](https://github.com/apache/tvm), [mmdeploy](https://github.com/open-mmlab/mmdeploy), [FlyCV](https://github.com/PaddlePaddle/FlyCV), [oneflow](https://github.com/Oneflow-Inc/oneflow), [flowgram.ai](https://github.com/bytedance/flowgram.ai), [deep-live-cam](https://github.com/hacksider/Deep-Live-Cam).

- Thanks to [HelloGithub](https://hellogithub.com/repository/nndeploy/nndeploy) for recommendation

  <a href="https://hellogithub.com/repository/314bf8e426314dde86a8c62ea5869cb7" target="_blank"><img src="https://abroad.hellogithub.com/v1/widgets/recommend.svg?rid=314bf8e426314dde86a8c62ea5869cb7&claim_uid=mu47rJbh15yQlAs" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## Contributors

<a href="https://github.com/nndeploy/nndeploy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nndeploy/nndeploy" />
</a>

[![Star History Chart](https://api.star-history.com/svg?repos=nndeploy/nndeploy&type=Date)](https://star-history.com/#nndeploy/nndeploy)
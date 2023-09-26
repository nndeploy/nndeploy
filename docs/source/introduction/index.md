
> 项目地址：<https://github.com/Alwaysssssss/nndeploy>

# 介绍

`nndeploy` 是一款最新上线的支持多平台、高性能、简单易用的机器学习部署框架。做到一个框架就可完成多端(云、边、端)模型的高性能部署。

作为一个多平台模型部署工具，我们的框架最大的宗旨就是高性能以及使用简单贴心😚，目前 `nndeploy` 已完成 [TensorRT](https://github.com/NVIDIA/TensorRT)、[OpenVINO](https://github.com/openvinotoolkit/openvino) 、[ONNXRuntime](https://github.com/microsoft/onnxruntime)、[MNN](https://github.com/alibaba/MNN)、[TNN](https://github.com/Tencent/TNN)、[ncnn](https://github.com/Tencent/ncnn/) 六个业界知名的推理框架的集成，后续会继续接入 `TFLite`、`paddle-lite`、`coreML`、`TVM`、`AITemplate`，在我们的框架下可使用一套代码轻松切换不同的推理后端进行推理，且不用担心部署框架对推理框架的抽象而带来的性能损失。

如果您需要部署自己的模型，目前 `nndeploy` 只需大概只要 `200` 行代码就可以完成模型在多端的部署。 同时还提供了高性能的前后处理模板和推理模板，该模板可帮助您简化模型端到端的部署流程。

目前 `nndeploy` 已完成 `YOLO` 系列等多个开源模型的部署，可供直接使用，目前我们还在积极部署其它开源模型。（如果您或团队有需要部署的开源模型或者其他部署相关的问题，非常欢迎随时来和我们探讨 😁）

# 模型部署的痛点

- 现在业界尚不存在各方面都远超其同类产品的推理框架，不同推理框架在不同平台、硬件下分别具有各自的优势。例如，在 `Linux` + `NVidia` 显卡机器推理，`TensorRT` 是性能最好的推理框架；在 `Windows` + `x86 CPU` 机器推理，`OpenVINO` 是性能最好的推理框架；在 `ARM Android` 下，有 `ncnn`、`MNN`、`TFLite`、TNN等一系列选择。

- 不同的推理框架有不一样的推理接口、推理配置等 API，针对不同推理框架都需要写一套代码，这对模型部署工程师而言，将带来较大学习成本、开发成本、维护成本
  
- 模型部署不仅仅只有模型推理，还有前处理、后处理，推理框架往往只提供模型推理的功能
  
- 目前很多场景是需要由多个模型组合解决该业务问题（例如stable diffusion、老照片修复、人脸识别等等），直接采用推理框架的原生接口，会有大量且低效的业务代码编写

# 架构简介

![架构简介](../../image/arch.png)

# `nndeploy` 的优势

## 支持多平台和多推理框架

- 支持多种推理框架：对多个业界知名推理框架的全面支持，包括 `TensorRT`、`OpenVINO`、`ONNXRuntime`、`MNN`、`TNN`、`ncnn` 等。未来，我们将继续扩展支持，包括 `TFLite`、`paddle-lite`、`coreML`、`TVM`、`AITemplate`、`RKNN`等
- 支持多种不同操作系统，包括 `Android`、`Linux`、`Windows`，正在适配 `macOS`、`IOS`。致力于在各种操作系统上无缝运行您的深度学习模型

|                      OS/Inference                       | Linux | Windows | Android | MacOS |  IOS  |                 开发人员                  | 备注  |
| :-----------------------------------------------------: | :---: | :-----: | :-----: | :---: | :---: | :---------------------------------------: | :---: |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      |  yes  |   no    |   no    |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) |  yes  |   yes   |   no    |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) |  yes  |   yes   |   no    |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
|          [MNN](https://github.com/alibaba/MNN)          |  yes  |   yes   |   yes   |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
|          [TNN](https://github.com/Tencent/TNN)          |  yes  |   yes   |   yes   |  no   |  no   | [02200059Z](https://github.com/02200059Z) |       |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  no   |   no    |   yes   |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |

## 直接可用的算法

- 目前已完成 [YOLOV5](https://github.com/ultralytics/yolov5)、[YOLOV6](https://github.com/meituan/YOLOv6)、[YOLOV8](https://github.com/ultralytics) 等模型的部署，可供您直接使用，后续我们持续不断去部署其它开源模型，让您开箱即用

|                      算法                       |             Inference             |                                       开发人员                                       | 备注  |
| :---------------------------------------------: | :-------------------------------: | :----------------------------------------------------------------------------------: | :---: |
| [YOLOV5](https://github.com/ultralytics/yolov5) | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |       |
|   [YOLOV6](https://github.com/meituan/YOLOv6)   |   TensorRt/OpenVINO/ONNXRuntime   | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |       |
|    [YOLOV8](https://github.com/ultralytics)     | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |       |

## 高性能

- **推理框架的高性能抽象**：每个推理框架也都有其各自的特性，需要足够尊重以及理解这些推理框架，才能在抽象中不丢失推理框架的特性，并做到统一的使用的体验。`nndeploy` 可配置第三方推理框架绝大部分参数，保证了推理性能。可直接操作理框架内部分配的输入输出，实现前后处理的零拷贝，提升模型部署端到端的性能。
  
- 线程池正在开发完善中，可实现有向无环图的流水线并行
  
- 内存池正在开发完善中，可实现高效的内存分配与释放
  
- 一组高性能的算子正在开发中，完成后将加速您模型前后处理速度

![阿Q正传.gif](../../image/meme_aq.gif)

## 简单易用

- **一套代码多端部署**：通过切换推理配置，一套代码即可在多端部署，算法的使用接口简单易用。示例代码如下：

  ```c++
  int main(int argc, char *argv[]) {
     // 有向无环图pipeline名称，例如:
    //  NNDEPLOY_YOLOV5/NNDEPLOY_YOLOV6/NNDEPLOY_YOLOV8
    std::string name = demo::getName();
    // 推理后端类型，例如:
    // kInferenceTypeOpenVino / kInferenceTypeTensorRt / kInferenceTypeOnnxRuntime
    base::InferenceType inference_type = demo::getInferenceType();
    // 推理设备类型，例如:
    // kDeviceTypeCodeX86:0/kDeviceTypeCodeCuda:0/...
    base::DeviceType device_type = demo::getDeviceType();
    // 模型类型，例如:
    // kModelTypeOnnx/kModelTypeMnn/...
    base::ModelType model_type = demo::getModelType();
    // 模型是否是路径
    bool is_path = demo::isPath();
    // 模型路径或者模型字符串
    std::vector<std::string> model_value = demo::getModelValue();
    // 有向无环图pipeline的输入边packert
    model::Packet input("detect_in");
    // 有向无环图pipeline的输出边packert
    model::Packet output("detect_out");
    // 创建模型有向无环图pipeline
    model::Pipeline *pipeline =
        model::createPipeline(name, inference_type, device_type, &input, &output,
                            model_type, is_path, model_value);

    // 初始化有向无环图pipeline
    base::Status status = pipeline->init();

    // 输入图片
    cv::Mat input_mat = cv::imread(input_path);
    // 将图片写入有向无环图pipeline输入边
    input.set(input_mat);
    // 定义有向无环图pipeline的输出结果
    model::DetectResult result;
    // 将输出结果写入有向无环图pipeline输出边
    output.set(result);

    // 有向无环图Pipeline运行
    status = pipeline->run();

    // 有向无环图pipelinez反初始化
    status = pipeline->deinit();

    // 有向无环图pipeline销毁
    delete pipeline;

    return 0;
  }
  ```
  
- **算法部署简单**：将 AI 算法端到端（前处理->推理->后处理）的部署抽象为有向无环图 `Pipeline`，前处理为一个 `Task`，推理也为一个 `Task`，后处理也为一个 `Task`，提供了高性能的前后处理模板和推理模板，上述模板可帮助您进一步简化端到端的部署流程。有向无环图还可以高性能且高效的解决多模型部署的痛点问题。示例代码如下:

  ```c++
  model::Pipeline* createYoloV5Pipeline(const std::string& name,
                                      base::InferenceType inference_type,
                                      base::DeviceType device_type,
                                      Packet* input, Packet* output,
                                      base::ModelType model_type, bool is_path,
                                      std::vector<std::string>& model_value) {
    model::Pipeline* pipeline = new model::Pipeline(name, input, output); // 有向无环图

    model::Packet* infer_input = pipeline->createPacket("infer_input"); // 推理模板的输入边
    model::Packet* infer_output = pipeline->createPacket("infer_output"); // 推理模板的输出

    // 搭建有向无图（preprocess->infer->postprocess）
    // 模型前处理模板model::CvtColrResize，输入边为input，输出边为infer_input
    model::Task* pre = pipeline->createTask<model::CvtColrResize>(
        "preprocess", input, infer_input);
    // 模型推理模板model::Infer(通用模板)，输入边为infer_input，输出边为infer_output
    model::Task* infer = pipeline->createInfer<model::Infer>(
        "infer", inference_type, infer_input, infer_output);
    // 模型后处理模板YoloPostProcess，输入边为infer_output，输出边为output
    model::Task* post = pipeline->createTask<YoloPostProcess>(
        "postprocess", infer_output, output);

    // 模型前处理任务pre的参数配置
    model::CvtclorResizeParam* pre_param =
        dynamic_cast<model::CvtclorResizeParam*>(pre->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 640;
    pre_param->w_ = 640;

    // 模型推理任务infer的参数配置
    inference::InferenceParam* inference_param =
        (inference::InferenceParam*)(infer->getParam());
    inference_param->is_path_ = is_path;
    inference_param->model_value_ = model_value;
    inference_param->device_type_ = device_type;

    // 模型后处理任务post的参数配置
    YoloPostParam* post_param = dynamic_cast<YoloPostParam*>(post->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.45;
    post_param->num_classes_ = 80;
    post_param->model_h_ = 640;
    post_param->model_w_ = 640;
    post_param->version_ = 5;

    return pipeline;
  }
  ```

# 架构详解

- **Directed Acyclic Graph**：有向无环图子模块。模型端到端的部署流程可抽象成 `3` 个子块：**模型前处理->模型推理->模型推理**，这是一个非常典型的有向无环图，对于多模型组合的算法而言，是更加复杂的的有向无环图，直接写业务代码去串联整个过程不仅容易出错，而且还效率低下，采用有向无环图的方式可以极大的缩减业务代码的编写。

- **Process Template**：前后处理模板以及推理子模板。我们希望还再可以简化您的部署流程，因此在模型端到端的部署的**模型前处理->模型推理->模型推理**的三个过程中，我们进一步设计模板。尤其是在推理模板上面花了足够多的心思，针对不同的模型，又有很多差异性，例如**单输入、多输出、静态形状输入、动态形状输入、静态形状输出、动态形状输出、是否可操作推理框架内部分配输入输出**等等一系列不同，只有具备丰富模型部署经验的工程师才能快速解决上述问题，故我们基于多端推理模块 `Inference` + 有向无环图节点 `Task` 再设计功能强大的**推理模板Infer**，这个推理模板可以帮您在内部处理上述针对模型的不同带来的差异。
  
- **Resouce Pool**：资源管理子模块。正在开发线程池以及内存池（这块是 `nndeploy` 正在火热开发的模块，期待大佬一起来搞事情）。线程池可实现有向无环图的流水线并行，内存池可实现高效的内存分配与释放。

- **Inference**：多端推理子模块（ `nndeploy` 还需要集成更多的推理框架，期待大佬一起来搞事情）。提供统一的推理接口去操作不同的推理后端，在封装每个推理框架时，我们都花了大量时间去理解并研究各个推理框架的特性，例如 `TensorRT` 可以使用外存推理，`OpenVINO` 有高吞吐率模式、`TNN` 可以操作内部分配输入输出等等。我们在抽象的过程中不会丢失推理框架的特性，并做到统一的使用的体验，还保证了性能。

- **OP**：高性能算子模块。我们打算去开发一套高性能的前后处理算子（期待有大佬一起来搞事情），提升模型端到端的性能，也打算开发一套 `nn` 算子库或者去封装 `oneDNN`、`QNN` 等算子库（说不定在 `nndeploy` 里面还会做一个推理框架呀）

- **Data Container**：数据容器子模块。推理框架的封装不仅推理接口的 API 的封装，还需要设计一个 Tensor，用于去与第三方推理框架的 Tensor 进行数据交互。 `nndeploy` 还设计图像处理的数据容器 Mat，并设计多设备的统一内存 Buffer。

- **Device**：设备管理子模块。为不同的设备提供统一的内存分配、内存拷贝、执行流管理等操作。

# TODO

- 接入更多的推理框架，包括`TFLite`、`paddle-lite`、`coreML`、`TVM`、`AITemplate`、`RKNN`、算能等等推理软件栈
- 部署更多的算法，包括 `Stable Diffusion`、`DETR`、`SAM`等等热门开源模型

# 加入我们

- 欢迎大家参与，一起打造最简单易用、高性能的机器学习部署框架
- 微信：titian5566 (可加我微信进 `nndeploy` 交流群，备注：`nndeploy`)

# 本文作者

- [02200059Z](https://github.com/02200059Z)
- [qixuxiang](https://github.com/qixuxiang)
- [PeterH0323](https://github.com/PeterH0323)
- [youxiudeshouyeren](https://github.com/youxiudeshouyeren)
- [Always](https://github.com/Alwaysssssss)
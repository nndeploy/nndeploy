## 介绍
nndeploy是一款最新上线的支持多平台、高性能、简单易用的机器学习部署框架，一套实现可在多端(云、边、端)完成模型的高性能部署。

作为一个多平台模型部署工具，我们的框架最大的宗旨就是高性能以及简单贴心(^‹^)，目前nndeploy已完成TensorRT、OpenVINO、ONNXRuntime、MNN、TNN、NCNN六个业界知名的推理框架的继承，后续会继续接入tf-lite、paddle-lite、coreML、TVM、AITemplate，在我们的框架下可使用一套代码轻松切换不同的推理后端进行推理，且不用担心部署框架对推理框架的抽象而带来的性能损失。

如果你需要部署自己的模型，目前nndeploy可帮助你在一个文件（大概只要200行代码）完成模型在多端的部署。nndeploy提供了高性能的前后处理模板和推理模板，上述模板可帮助你简化端到端的部署流程。如果只需使用已有主流模型进行自己的推理，目前nndeploy已完成YOLO系列等多个开源模型的部署，可供直接使用，目前我们还在积极部署其它开源模型。（如果你或团队有需要部署的开源模型或者其他部署相关的问题，非常欢迎随时来和我们探讨(^-^)）


## 架构简介
![架构简介](doc/image/架构.png)


## 优势特性
nndeploy具有如下优势特性：
### 支持多平台
支持的平台和推理框架如下表所示
|                      OS/Inference                       | Linux | Windows | Android | MacOS |  iOS  |                 开发人员                  | 备注  |
| :-----------------------------------------------------: | :---: | :-----: | :-----: | :---: | :---: | :---------------------------------------: | :---: |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      |  yes  |   no    |   no    |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) |  yes  |   yes   |   no    |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) |  yes  |   yes   |   no    |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
|          [MNN](https://github.com/alibaba/MNN)          |  yes  |   yes   |   yes   |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |
|          [TNN](https://github.com/Tencent/TNN)          |  yes  |   yes   |   yes   |  no   |  no   | [02200059Z](https://github.com/02200059Z) |       |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  no   |   no    |   yes   |  no   |  no   | [Always](https://github.com/Alwaysssssss) |       |

``注: yes：完成在该平台的验证，no：目前正在验证中``

## 直接可用的算法
|                      算法                       |             Inference             |                                       开发人员                                       | 备注  |
| :---------------------------------------------: | :-------------------------------: | :----------------------------------------------------------------------------------: | :---: |
| [YOLOV5](https://github.com/ultralytics/yolov5) | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |       |
|   [YOLOV6](https://github.com/meituan/YOLOv6)   |   TensorRt/OpenVINO/ONNXRuntime   | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |       |
|    [YOLOV8](https://github.com/ultralytics)     | TensorRt/OpenVINO/ONNXRuntime/MNN | [02200059Z](https://github.com/02200059Z)、[Always](https://github.com/Alwaysssssss) |       |

### 简单易用
- **一套代码多端部署**：通过切换推理配置，一套代码即可在多端部署，算法的使用接口简单易用。示例代码如下
  ```c++
  int main(int argc, char *argv[]) {
     // 有向无环图pipeline名称，例如:
    // NNDEPLOY_YOLOV5/NNDEPLOY_YOLOV6/NNDEPLOY_YOLOV8
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

- **算法部署简单**：将AI算法端到端（前处理->推理->后处理）的部署抽象为有向无环图Pipeline，前处理为一个任务Task，推理也为一个任务Task，后处理也为一个任务Task，提供了高性能的前后处理模板和推理模板，上述模板可帮助你进一步简化端到端的部署流程。有向无环图还可以高性能且高效的解决多模型部署的痛点问题。示例代码如下
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

### 高性能
- **推理框架的高性能抽象**：每个推理框架也都有其各自的特性，需要足够尊重以及理解这些推理框架，才能在抽象中不丢失推理框架的特性，并做到统一的使用的体验。nndeploy可配置第三方推理框架绝大部分参数，保证了推理性能。可直接操作理框架内部分配的输入输出，实现前后处理的零拷贝，提升模型部署端到端的性能。
  
- 线程池正在开发完善中，可实现有向无环图的流水线并行
  
- 内存池正在开发完善重，可实现高效的内存分配与释放
  
- 一组高性能的算子正在开发中，完成后将加速你模型前后处理速度

## 快速开始
### 编译
+ 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
  ```
  mkdir build
  cp cmake/config.cmake build
  cd build
  ```
+ 编辑`build/config.cmake`来定制编译选项
  + 将`set(ENABLE_NNDEPLOY_OPENCV OFF)`改为`set(ENABLE_NNDEPLOY_OPENCV PATH/linux/OpenCV)`，`nndeploy`会启用并链接`OpenCV`，如果你想启用并链接的其他第三方库，也是做同样的处理
  + 将`set(ENABLE_NNDEPLOY_DEVICE_CPU OFF)`改为`set(ENABLE_NNDEPLOY_DEVICE_CPU ON)`，`nndeploy`会启用`CPU`设备。如果你想启用其他设备（ARM、X86、CUDA …），也是做同样的处理
  + 将`set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME OFF)`改为`set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "PATH/linux/onnxruntime-linux-x64-1.15.1")`，`nndeploy`会启用并链接推理后端`ONNXRuntime`。如果你想启用并链接其他推理后端（OpenVINO、TensorRT、TNN …），也是做同样的处理
  + `启用并链接第三方库有两种选择`
    + 开关`ON` - 当你安装了该库，并且可以通过find_package找到该库，可以采用该方式，例如CUDA、CUDNN、OpenCV、TenosrRT
    + 路径`PATH` - 头文件以及库的根路径，其形式必须为
      + 头文件：`PATH/include`
      + 库：`PATH/lib `
      + windows dll: `PATH/bin`
+ 开始`make nndeploy`库
  ```
  cmake ..
  make -j4
  ```
+ 安装，将nndeploy相关库可执行文件、第三方库安装至`build/install/lib`
  ```
  make install
  ```

### nndeploy资源仓库
已验证模型、第三方库、测试数据放在HuggingFace上，如果您有需要可以去下载，[下载链接](https://huggingface.co/alwaysssss/nndeploy/tree/main)。`但强烈建议您自己去管理自己的模型仓库、第三方库、测试数据`。

+ 第三方库编译文档以及官方下载链接

|                        第三方库                         |  主版本  |                                          编译文档                                           |                                                                               官方库下载链接                                                                               |                 备注                 |
| :-----------------------------------------------------: | :------: | :-----------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   |                           [链接](https://opencv.org/get-started/)                           |                                                                  [链接](https://opencv.org/get-started/)                                                                   |                                      |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 |  [链接](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing)  |                                                            [链接](https://developer.nvidia.com/zh-cn/tensorrt)                                                             | 支持TensorRT 7、支持jetson-orin-nano |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 |      [链接](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md)      | [链接](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=RUNTIME&OP_SYSTEM=MACOS&VERSION=v_2023_0_1&DISTRIBUTION=ARCHIVE) |                                      |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [链接](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_useful_api.zh.md) |                                                   [链接](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1)                                                    |                                      |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   |            [链接](https://mnn-docs.readthedocs.io/en/latest/compile/engine.html)            |                                                         [链接](https://github.com/alibaba/MNN/releases/tag/2.6.0)                                                          |                                      |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  |          [链接](https://github.com/Tencent/TNN/blob/master/doc/cn/user/compile.md)          |                                                         [链接](https://github.com/Tencent/TNN/releases/tag/v0.3.0)                                                         |                                      |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  |            [链接](https://github.com/Tencent/ncnn/tree/master/docs/how-to-build)            |                                                       [链接](https://github.com/Tencent/ncnn/releases/tag/20230816)                                                        |                                      |
- 补充说明    
  - 我使用第三方库的上述版本，通常使用其他版本的也没有问题
  - TensorRT
    - [Windows链接](https://zhuanlan.zhihu.com/p/476679322)
    - 安装前请确保 显卡驱动、cuda、cudnn均已安装且版本一致

### 跑通检测模型YOLOV5s demo
#### 准备工作
+ Linux下需安装opencv
  + sudo apt install libopencv-dev
  + [参考链接](https://cloud.tencent.com/developer/article/1657529)
+ [下载模型](https://huggingface.co/alwaysssss/nndeploy/resolve/main/model_zoo/detect/yolo/yolov5s.onnx)，解压
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/model_zoo/detect/yolo/yolov5s.onnx
  ```
+ 下载第三方库，[ubuntu22.04](https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/ubuntu22.04_x64.tar)，[windows](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z)， [android](https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/android.tar)。 解压
  ```shell
  # ubuntu22.04_x64
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/ubuntu22.04_x64.tar
  # windows
  wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z
  # android
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/third_party/android.tar
  ```
+ [下载测试数据](https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg)
  ```shell
  wget https://huggingface.co/alwaysssss/nndeploy/resolve/main/test_data/detect/sample.jpg
  ```
#### 编译
+ 在根目录创建`build`目录，将`cmake/config_os.cmake（config_linux.cmake/config_windows.cmake/config_android.cmake）`复制到该目录，修改名称为`config.cmake`
  ```
  mkdir build
  cp cmake/config_xx.cmake build
  mv config_yolov5s.cmake config.cmake
  cd build
  ```
+ 编辑`build/config.cmake`来定制编译选项
+ 将所有第三方库的路径改为您的路径，例如set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "PATH/third_party/ubuntu22.04_x64/onnxruntime-linux-x64-1.15.1")改为set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "PATH/third_party/ubuntu22.04_x64/onnxruntime-linux-x64-1.15.1")。`PATH为您下载第三方库后的解压路径`
+ 开始`make nndeploy`库
  ```
  cmake ..
  make -j4
  ```
+ 安装，将nndeploy相关库可执行文件、第三方库安装至`build/install/lib`
  ```
  make install
  ```
#### Linux下运行YOLOV5s
```shell
cd PATH/nndeploy/build/install/lib
export LD_LIBRARY_PATH=PATH/nndeploy/build/install/lib:$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openvino 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// tensorrt 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// tensorrt 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx.mnn --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```
`注：请将上述PATH更换为自己对应的目录`

#### Windows下运行YOLOV5s
```shell
cd PATH/nndeploy/build/install/bin
export LD_LIBRARY_PATH=PATH/nndeploy/build/install/bin:$LD_LIBRARY_PATH
// onnxruntime 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOnnxRuntime --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// openvino 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeOpenVino --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// tensorrt 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeTensorRt --device_type kDeviceTypeCodeCuda:0 --model_type kModelTypeOnnx --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg

// MNN 推理
./demo_nndeploy_detect --name NNDEPLOY_YOLOV5 --inference_type kInferenceTypeMnn --device_type kDeviceTypeCodeX86:0 --model_type kModelTypeMnn --is_path --model_value PATH/model_zoo/detect/yolo/yolov5s.onnx.mnn --input_type kInputTypeImage  --input_path PATH/test_data/detect/sample.jpg --output_path PATH/temp/sample_output.jpg
```
`注：请将上述PATH更换为自己对应的目录`

## 社区文档
- [Always](https://github.com/Alwaysssssss)，[02200059Z](https://github.com/02200059Z):《[nndeploy综述](https://zhuanlan.zhihu.com/p/656359928)》
- [02200059Z](https://github.com/02200059Z):《[如何新增一个推理框架](https://blog.csdn.net/echoesssss/article/details/132674100?spm=1001.2014.3001.5502)》


## 参考
- [TNN](https://github.com/Tencent/TNN)
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy)
- [opencv](https://github.com/opencv/opencv)
- [CGraph](https://github.com/ChunelFeng/CGraph)
- [tvm](https://github.com/apache/tvm)
- [mmdeploy](https://github.com/open-mmlab/mmdeploy)


## 加入我们
* 欢迎大家参与，一起打造最简单易用、高性能的机器学习部署框架
* 微信：titian5566，备注：nndeploy

<img align="left" src="doc/image/Always.jpg" width="512px">



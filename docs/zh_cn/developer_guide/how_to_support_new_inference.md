# how_to_support_new_inference


## 介绍

inference是nndeploy的多端推理子模块，通过对第三方推理框架的抽象，屏蔽不同推理框架的差异性，并做到统一的接口调用的体验，nndeploy当前已经支持TensorRT、OpenVINO、ONNXRuntime、MNN、TNN、ncnn、coreML、paddle-lite、AscendCL、RKNN等多个推理框架。


## 步骤

新增一个推理框架主要分为以下五个步骤：

+ 1. 新增推理框架相关枚举类型
+ 2. 继承基类InferenceParam
+ 3. 继承基类Inference
+ 4. 编写Converter
+ 5. 修改cmake


### 步骤一：新增设备类型枚举

#### 1.1 新增ModelType枚举
+ （1）修改文件 `<path>\include\nndeploy\base\common.h`，在`ModelType`中添加新模型格式类枚举，格式为`kModelTypeXxx` 

+ （2）修改文件 `<path>\source\nndeploy\base\common.cc`，在`ModelType stringToModelType(const std::string &src)`函数中添加字符串转换为新模型格式类枚举实现

#### 2.1 新增InferenceType枚举
+ （1）修改文件 `<path>\include\nndeploy\base\common.h`，在`InferenceType`中添加新推理框架格式的枚举，格式为`kInferenceTypeXxx` 

+ （2）修改文件 `<path>\source\nndeploy\base\common.cc`，在`InferenceType stringToInferenceType(const std::string &src)`函数中添加字符串转换为新推理框架格式的枚举实现


### 步骤二： 继承基类InferenceParam

+ （1）在`<path>\include\nndeploy\inference`下新增`xxx\xxx_inference_param.h`文件，可参考MNN(`<path>\include/nndeploy/inference/mnn/mnn_inference_param.h`)或TensorRT(`<path>\include/nndeploy/inference/tensorrt/tensorrt_inference_param.h`)

+ （2）在`<path>\source\nndeploy\inference`下新增`xxx\xxx_inference_param.cc`文件，可参考MNN(`<path>\source/nndeploy/inference/mnn/mnn_inference_param.c`)或TensorRT(`<path>\include/nndeploy/inference/tensorrt/tensorrt_inference_param.cc`)


### 步骤三： 继承基类Inference

+ （1）在`<path>\include\nndeploy\inference`下新增`xxx\xxx_inference.h`文件，可参考MNN(`<path>\include/nndeploy/inference/mnn/mnn_inference.h`)或TensorRT(`<path>\include/nndeploy/inference/tensorrt/tensorrt_inference.h`)

+ （2）在`<path>\source\nndeploy\inference`下新增`xxx\xxx_inference.cc`文件，可参考MNN(`<path>\source/nndeploy/inference/mnn/mnn_inference.c`)或TensorRT(`<path>\include/nndeploy/inference/tensorrt/tensorrt_inference.cc`)

### 步骤四： 编写Converter

nndeploy提供了统一的Tensor以及推理所需的超参数数据结构，每个推理框架都有自定义Tensor以及超参数数据结构，为了保证统一的接口调用的体验，需编写转化器模块。

+ （1）在`<path>\include\nndeploy\inference`下新增`xxx\xxx_converter.h`文件，可参考MNN(`<path>\include/nndeploy/inference/mnn/mnn_converter.h`)或TensorRT(`<path>\include/nndeploy/inference/tensorrt/tensorrt_converter.h`)

+ （2）在`<path>\source\nndeploy\inference`下新增`xxx\xxx_inference.cc`文件，可参考MNN(`<path>\source/nndeploy/inference/mnn/mnn_converter.c`)或TensorRT(`<path>\include/nndeploy/inference/tensorrt/tensorrt_converter.cc`)


### 步骤五：修改cmake 

+ （1）修改主cmakelist `<path>\CMakeLists.txt`，
  + 新增推理框架编译选项`nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_XXX "ENABLE_NNDEPLOY_INFERENCE_XXX" OFF)`
  + 由于新设备的增加，增加了源文件和头文件，需将源文件和头文件加入到编译文件中，需在`if(ENABLE_NNDEPLOY_INFERENCE) endif()`的代码块中增加如下cmake源码
    ```shell
    if (ENABLE_NNDEPLOY_INFERENCE_XXX)
      file(GLOB_RECURSE INFERENCE_XXX_SOURCE
        "${ROOT_PATH}/include/nndeploy/inference/xxx/*.h"
        "${ROOT_PATH}/source/nndeploy/inference/xxx/*.cc"
      )
      set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_XXX_SOURCE})
    endif()
    ```

+ （2）链接推理框架的三方库
  + 需要在`<path>\cmake`目录下新增`xxx.cmake`，类似`<path>\cmake\mnn.cmake`或`<path>\cmake\xxx.cmake`
  + 修改`<path>\cmake\nndeploy.cmake`，新增`include("${ROOT_PATH}/cmake/xxx.cmake")`

+ （3）修改`<path>\build\config.cmake`,新增设备编译选项`set(ENABLE_NNDEPLOY_INFERENCE_XXX ON)`
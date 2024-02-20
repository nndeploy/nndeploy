# 编译


## 拉取源代码

```shell
git clone --recursive https://github.com/DeployAI/nndeploy.git
```


## 编译宏介绍

参考`path/cmake/config.cmake`


## 整体编译方案

+ 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
  ```
  mkdir build
  cp cmake/config.cmake build
  cd build
  ```

+ 编辑`build/config.cmake`来定制编译选项
  
  + 将`set(ENABLE_NNDEPLOY_OPENCV OFF)`改为`set(ENABLE_NNDEPLOY_OPENCV path/linux/OpenCV)`，`nndeploy`会启用并链接`OpenCV`，如果你想启用并链接的其他第三方库，也是做同样的处理
  
  + 将`set(ENABLE_NNDEPLOY_DEVICE_CPU OFF)`改为`set(ENABLE_NNDEPLOY_DEVICE_CPU ON)`，`nndeploy`会启用`CPU`设备。如果你想启用其他设备（ARM、X86、CUDA …），也是做同样的处理
  
  + 将`set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME OFF)`改为`set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "path/linux/onnxruntime-linux-x64-1.15.1")`，`nndeploy`会启用并链接推理后端`ONNXRuntime`。如果你想启用并链接其他推理后端（OpenVINO、TensorRT、TNN …），也是做同样的处理
  
  + `启用并链接第三方库有两种选择`
    + 开关`ON` - 当你安装了该库，并且可以通过find_package找到该库，可以采用该方式，例如CUDA、CUDNN、OpenCV、TenosrRT
    + 路径`path` - 头文件以及库的根路径，其形式必须为
      + 头文件：`path/include`
      + 库：`path/lib `
      + windows dll: `path/bin`
  
+ 开始`make nndeploy`库
  ```
  cmake ..
  make -j4
  ```

+ 安装，将nndeploy相关库可执行文件、第三方库安装至`build/install/lib`
  ```
  make install
  ```


## 环境要求

### Windows环境要求

+ cmake >= 3.12

+ Microsoft Visual Studio >= 2017

### Linux环境要求

+ cmake >= 3.12
  
+ gcc >= 4.9

### Android环境要求

+ cmake >= 3.12

+ ndk

### MacM1环境要求

+ cmake >= 3.12

+ xcode

### iOS环境要求

+ cmake >= 3.12

+ xcode


## nndeploy提供的第三方库资源


### Windows下载资源

|                        第三方库                         |  主版本  |               Windows下载资源               | 备注  |
| :-----------------------------------------------------: | :------: | :-----------------------------------------: | :---: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   | [下载链接](https://opencv.org/get-started/) |       |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 | [下载链接](https://opencv.org/get-started/) |       |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 | [下载链接](https://opencv.org/get-started/) |       |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [下载链接](https://opencv.org/get-started/) |       |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   | [下载链接](https://opencv.org/get-started/) |       |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |

### Linux下载资源

|                        第三方库                         |  主版本  |                Linux下载资源                | 备注  |
| :-----------------------------------------------------: | :------: | :-----------------------------------------: | :---: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   | [下载链接](https://opencv.org/get-started/) |       |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 | [下载链接](https://opencv.org/get-started/) |       |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 | [下载链接](https://opencv.org/get-started/) |       |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [下载链接](https://opencv.org/get-started/) |       |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   | [下载链接](https://opencv.org/get-started/) |       |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |

### Android下载资源

|                        第三方库                         |  主版本  |               Android下载资源               | 备注  |
| :-----------------------------------------------------: | :------: | :-----------------------------------------: | :---: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   | [下载链接](https://opencv.org/get-started/) |       |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 | [下载链接](https://opencv.org/get-started/) |       |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 | [下载链接](https://opencv.org/get-started/) |       |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [下载链接](https://opencv.org/get-started/) |       |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   | [下载链接](https://opencv.org/get-started/) |       |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |

### MacM1下载资源

|                        第三方库                         |  主版本  |                MacM1下载资源                | 备注  |
| :-----------------------------------------------------: | :------: | :-----------------------------------------: | :---: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   | [下载链接](https://opencv.org/get-started/) |       |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 | [下载链接](https://opencv.org/get-started/) |       |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 | [下载链接](https://opencv.org/get-started/) |       |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [下载链接](https://opencv.org/get-started/) |       |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   | [下载链接](https://opencv.org/get-started/) |       |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |

### iOS下载资源

|                        第三方库                         |  主版本  |                 iOS下载资源                 | 备注  |
| :-----------------------------------------------------: | :------: | :-----------------------------------------: | :---: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   | [下载链接](https://opencv.org/get-started/) |       |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 | [下载链接](https://opencv.org/get-started/) |       |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 | [下载链接](https://opencv.org/get-started/) |       |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [下载链接](https://opencv.org/get-started/) |       |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   | [下载链接](https://opencv.org/get-started/) |       |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  | [下载链接](https://opencv.org/get-started/) |       |



## 第三方库编译文档以及官方下载链接

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
  - 我们使用第三方库的上述版本，通常使用其他版本的也没有问题
  - TensorRT
    - [Windows链接](https://zhuanlan.zhihu.com/p/476679322)
    - 安装前请确保 显卡驱动、cuda、cudnn均已安装且版本一致



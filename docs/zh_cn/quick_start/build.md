# 编译


## 拉取源代码

```shell
git clone https://github.com/nndeploy/nndeploy.git
git submodule update --init --recursive
```


## 编译宏介绍

参考[config.cmake](../../../cmake/config.cmake) 详细介绍，该文件详细介绍了所有编译宏含义以及用法。


## config.cmake的编辑规则

+ `X86`设备。`set(ENABLE_NNDEPLOY_DEVICE_X86 ON)`，如果你想使能其他设备（ARM、X86、CUDA …），也可做同样的处理
+ `nndeploy`通过路径的方式链接推理后端`ONNXRuntime`。`set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "path/onnxruntime")`，如果你想启用并链接其他推理后端（OpenVINO、MNN、TNN …），也可做同样的处理
+ `nndeploy`通过find_package的方式链接推理后端`TensorRT`。`set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT ON)`，对于其他可以通过find_package找到的库，也可做同样的处理
+ 模型部署实例。首先将模型类别`set(NABLE_NNDEPLOY_MODEL_XXX ON)`，再将具体的模型`set(NABLE_NNDEPLOY_MODEL_XXX_YYY ON)`
+ 可执行程序，`set(ENABLE_NNDEPLOY_DEMO ON)`

### `使能并链接第三方库的两种方法`
+ `方法一`：路径`path`，头文件以及库的根路径，其形式必须修改为
  + 头文件：`path/include`
  + 库：`path/lib `
  + windows dll: `path/bin`
+ `方法二`：开关`ON`，如果你安装了该库，并且可以通过find_package找到该库，可以采用该方式，例如CUDA、CUDNN、OpenCV、TenosrRT


## 主库编译

+ 默认编译产物为：libnndeploy.so、demo_nndeploy_dag
  

## Windows

+ 环境要求
  + cmake >= 3.12
  + Microsoft Visual Studio >= 2017
  
+ nndeploy提供的第三方库

  |                        第三方库                         |  主版本  |                                       Windows下载链接                                       | 备注  |
  | :-----------------------------------------------------: | :------: | :-----------------------------------------------------------------------------------------: | :---: |
  |       [opencv](https://github.com/opencv/opencv)        |  4.8.0   | [下载链接](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z) |       |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 | [下载链接](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z) |       |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [下载链接](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z) |       |
  |          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   | [下载链接](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z) |       |
  |          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  | [下载链接](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z) |       |
  |        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  | [下载链接](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/windows_x64.7z) |       |

  注：将上述所有库打包为一个压缩包windows_x64.7z，存放在huggingface上，使用前请将压缩包windows_x64.7z解压

+ 具体步骤
  + 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
    ```
    mkdir build
    cp cmake/config.cmake build
    cd build
    ```

  + 编辑`build/config.cmake`自定义编译选项（笔者的自定义编译选项：[path/cmake/config_windows.cmake](../../../cmake/config_windows.cmake)）
     
  + 开始`cmake`
    ```
    cmake ..
    ```

  + 通过visual studio打开`build/nndeploy.sln`，开始编译、安装、执行

## Linux

+ 环境要求
  + cmake >= 3.12
  + gcc >= 5.1

+ nndeploy提供的第三方库

  |                        第三方库                         |  主版本  |                                       Linux下载链接                                       | 备注  |
  | :-----------------------------------------------------: | :------: | :---------------------------------------------------------------------------------------: | :---: |
  | [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/ubuntu22.04_x64.tar |       |
  | [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/ubuntu22.04_x64.tar |       |
  |          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/ubuntu22.04_x64.tar |       |
  |          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/ubuntu22.04_x64.tar |       |
  |        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/ubuntu22.04_x64.tar |       |

  注：将上述所有库打包为一个压缩包ubuntu22.04_x64.tar，存放在huggingface上，使用前请将压缩包ubuntu22.04_x64.tar解压

  + 安装opencv
    + `sudo apt install libopencv-dev` [参考链接](https://cloud.tencent.com/developer/article/1657529)
  + 安装TensorRT、cudnn、cuda、显卡驱动
  
+ 具体步骤
  + 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
    ```
    mkdir build
    cp cmake/config.cmake build
    cd build
    ```

  + 编辑`build/config.cmake`自定义编译选项（笔者的自定义编译选项：[path/cmake/config_linux.cmake](../../../cmake/config_linux.cmake)）
      
  + `cmake`
    ```
    cmake ..
    ```

  + 编译
     ```
    make -j
    ```

  + 安装, 将nndeploy的库、可执行文件、第三方库安装至build/install/lib
     ```
    make install
    ```

## Android

+ 环境要求
  + cmake >= 3.12
  + ndk

+ nndeploy提供的第三方库

  |                  第三方库                  | 主版本 |                                  Android下载链接                                  | 备注  |
  | :----------------------------------------: | :----: | :-------------------------------------------------------------------------------: | :---: |
  | [opencv](https://github.com/opencv/opencv) | 4.8.0  | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/android.tar |       |
  |   [MNN](https://github.com/alibaba/MNN)    | 2.6.2  | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/android.tar |       |
  |   [TNN](https://github.com/Tencent/TNN)    | v0.3.0 | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/android.tar |       |
  |  [ncnn](https://github.com/Tencent/ncnn/)  | v0.3.0 | wget https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party/android.tar |       |

  注：将上述所有库打包为一个压缩包android.tar，存放在huggingface上，使用前请将压缩包android.tar解压

+ 具体步骤
  + 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
    ```
    mkdir build
    cp cmake/config.cmake build
    cd build
    ```

  + 编辑`build/config.cmake`自定义编译选项（笔者的自定义编译选项：[path/cmake/config_android.cmake](../../../cmake/config_android.cmake)）
      
  + 开始`cmake`，需要指定ndk
    ```
    cmake .. -DCMAKE_TOOLCHAIN_FILE=/snap/android-ndk-r25c/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_STL=c++_static -DANDROID_NATIVE_API_LEVEL=android-14 -DANDROID_TOOLCHAIN=clang -DBUILD_FOR_ANDROID_COMMAND=true
    ```

  + 开始编译
     ```
    make -j8
    ```

  + 开始安装, 将nndeploy相关库可执行文件、第三方库安装至build/install/lib
     ```
    make install
    ```

  注：

## Mac（TODO）

+ 环境要求
  + cmake >= 3.12
  + xcode


## iOS（TODO）

+ 环境要求
  + cmake >= 3.12
  + xcode


## 第三方库官方编译文档以及下载链接

|                        第三方库                         |  主版本  |                                          编译文档                                           |                                                                               官方库下载链接                                                                               |         备注         |
| :-----------------------------------------------------: | :------: | :-----------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   |                           [链接](https://opencv.org/get-started/)                           |                                                                  [链接](https://opencv.org/get-started/)                                                                   |                      |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 |  [链接](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing)  |                                                            [链接](https://developer.nvidia.com/zh-cn/tensorrt)                                                             | 支持jetson-orin-nano |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 |      [链接](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md)      | [链接](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=RUNTIME&OP_SYSTEM=MACOS&VERSION=v_2023_0_1&DISTRIBUTION=ARCHIVE) |                      |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [链接](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_useful_api.zh.md) |                                                   [链接](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1)                                                    |                      |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   |            [链接](https://mnn-docs.readthedocs.io/en/latest/compile/engine.html)            |                                                         [链接](https://github.com/alibaba/MNN/releases/tag/2.6.0)                                                          |                      |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  |          [链接](https://github.com/Tencent/TNN/blob/master/doc/cn/user/compile.md)          |                                                         [链接](https://github.com/Tencent/TNN/releases/tag/v0.3.0)                                                         |                      |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  |            [链接](https://github.com/Tencent/ncnn/tree/master/docs/how-to-build)            |                                                       [链接](https://github.com/Tencent/ncnn/releases/tag/20230816)                                                        |                      |


## 补充说明    

- 我们使用第三方库的上述版本，通常使用其他版本的也没有问题

- TensorRT
  - [Windows链接](https://zhuanlan.zhihu.com/p/476679322)
  - 安装前请确保 显卡驱动、cuda、cudnn均已安装且版本一致

- 在windows平台下，系统目录自带onnxruntime，故你在运行时或许可能会链接到系统目录下自带的onnxruntime，从而导致运行时出错。解决办法
  - 将你自己的onnxruntime库拷贝至build目录下

- 使能ENABLE_NNDEPLOY_NET，需要链接onnx和protobuf，会出现如下cmake error，实际已经完成了cmake，故可以继续make，make不会报错
  ```shell
  CMake Error in third_party/onnx/CMakeLists.txt:
  export called with target "onnx_proto" which requires target "libprotobuf"
  that is not in any export set.
  ```

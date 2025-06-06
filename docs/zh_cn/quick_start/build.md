# 编译


## 1. 拉取源代码

```shell
git clone https://github.com/nndeploy/nndeploy.git
cd nndeploy
# 拉取子模块
git submodule update --init --recursive
# 如果拉取子模块失败，调用克隆子模块脚本
./clone_submodule.sh
```


## 2. 编译宏介绍

+ 参考[编译宏文档](./build_macro.md) 的详细介绍

包含了以下几类配置：

1. **基础构建选项（建议采用默认配置）**：如是否构建为共享库、使用的C++标准版本等等
2. **核心模块选项（建议采用默认配置）**：更细粒度控制需要编译的文件
3. **设备后端选项（按需打开，默认全部关闭，不依赖任何设备后端）**：如CUDA、OpenCL、各种NPU等硬件加速支持
4. **算子后端选项（按需打开，默认全部关闭，不依赖任何算子后端）**：如cudnn、onednn、xnnpack、qnnpack
5. **推理后端选项（按需打开，默认全部关闭，不依赖任何推理后端）**：如TensorRT、OpenVINO、ONNX Runtime等推理框架支持
6. **算法插件选项（建议采用默认配置，传统CV类算法打开，语言类和文生图类算法默认关闭）**：如检测、分割、llm、文生图等算法插件

    + 其中传统CV类算法依赖`OpenCV`，例如检测、分割、分类等，需要打开`ENABLE_NNDEPLOY_OPENCV`

    + **注意**：其中`语言类和文生图类模型`依赖C++分词器[tokenizer-cpp](https://github.com/mlc-ai/tokenizers-cpp)，所以需要打开`ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP`，打开前参考[precompile_tokenizer_cpp.md](./precompile_tokenizer_cpp.md)
  
## 3. 编译方法

[config.cmake](../../../cmake/config.cmake)是nndeploy的编译配置文件，用于控制项目的编译选项。

> 相比于原生cmake -D选项，用户配置好的编译选项文件，可保留下来多次使用，在文件上还可以增加注释，方便后续维护。

> 相比编译脚本，无需为每个平台编写多种类型脚本，也不会遇到脚本环境问题，只需在根目录创建build目录，将[config.cmake](../../../cmake/config.cmake)复制到该目录，然后修改config.cmake文件，即可开始编译。

假设你在根目录下，具体命令行如下：

```shell
mkdir build                 # 创建build目录
cp cmake/config.cmake build # 将编译配置模板复制到build目录
cd build                    # 进入build目录
vim config.cmake            # 使用编辑器vscode等工具直接修改config.cmake文件
cmake ..                    # 生成构建文件
make -j                     # 使用8个线程并行编译
```


## 4. 主库编译

+ 默认编译产物为：libnndeploy_framework.so
+ 算法插件编译产物为：libnndeploy_plugin_xxx.so
+ 可执行程序编译产物为：nndeploy_demo_xxx

> 注：xxx代表特定算法插件和特定的可执行程序，例如：nndeploy_plugin_detect.so、nndeploy_demo_detect、nndeploy_demo_dag
  

## 5. Windows

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
     
  + 开始`cmake`
    ```
    cmake ..
    ```

  + 通过visual studio打开`build/nndeploy.sln`，开始编译、安装、执行

## 6. Linux

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
  + 安装TensorRT cpp sdk [参考链接](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian)、cudnn、cuda、GPU driver

  
+ 具体步骤
  + 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
    ```
    mkdir build
    cp cmake/config.cmake build
    cd build
    ```
      
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

## 7. Android

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

## 8. Mac（TODO）

+ 环境要求
  + cmake >= 3.12
  + xcode


## 9. iOS（TODO）

+ 环境要求
  + cmake >= 3.12
  + xcode


## 10. Linux + 华为昇腾


+ 环境要求
  + cmake >= 3.12
  + gcc >= 5.1

+ 三方库

  + 安装opencv
    + `sudo apt install libopencv-dev` [参考链接](https://cloud.tencent.com/developer/article/1657529)
  + 安装AscendCL sdk [ascend_env.md](./ascend_env.md)

  
+ 具体步骤
  + 在根目录创建`build`目录，将`cmake/config.cmake`复制到该目录
    ```
    mkdir build
    cp cmake/config.cmake build
    vim config.cmake # 使用编辑器vscode等工具直接修改config.cmake文件，需要打开昇腾的编译宏：setENABLE_NNDEPLOY_DEVICE_ASCEND_CL ON）
    cd build
    ```
      
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


## 11. 第三方库官方编译文档以及下载链接

|                        第三方库                         |  主版本  |                                          编译文档                                           |                                                                               官方库下载链接                                                                               |         备注         |
| :-----------------------------------------------------: | :------: | :-----------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
|       [opencv](https://github.com/opencv/opencv)        |  4.8.0   |                           [链接](https://opencv.org/get-started/)                           |                                                                  [链接](https://opencv.org/get-started/)                                                                   |                      |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 |  [链接](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing)  |                                                            [链接](https://developer.nvidia.com/zh-cn/tensorrt)                                                             | 支持jetson-orin-nano |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 |      [链接](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md)      | [链接](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=RUNTIME&OP_SYSTEM=MACOS&VERSION=v_2023_0_1&DISTRIBUTION=ARCHIVE) |                      |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [链接](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_useful_api.zh.md) |                                                   [链接](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1)                                                    |                      |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   |            [链接](https://mnn-docs.readthedocs.io/en/latest/compile/engine.html)            |                                                         [链接](https://github.com/alibaba/MNN/releases/tag/2.6.0)                                                          |                      |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  |          [链接](https://github.com/Tencent/TNN/blob/master/doc/cn/user/compile.md)          |                                                         [链接](https://github.com/Tencent/TNN/releases/tag/v0.3.0)                                                         |                      |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  |            [链接](https://github.com/Tencent/ncnn/tree/master/docs/how-to-build)            |                                                       [链接](https://github.com/Tencent/ncnn/releases/tag/20230816)                                                        |                      |


## 12. 补充说明    

- 我们使用第三方库的上述版本，通常使用其他版本的也没有问题

- TensorRT
  - [Windows链接](https://zhuanlan.zhihu.com/p/476679322)
  - 安装前请确保 显卡驱动、cuda、cudnn均已安装且版本一致

- 在windows平台下，系统目录自带onnxruntime，故你在运行时或许可能会链接到系统目录下自带的onnxruntime，从而导致运行时出错。解决办法
  - 将你自己的onnxruntime库拷贝至build目录下
      
      
      

  

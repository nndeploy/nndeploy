# 编译


## 1. 拉取源代码

```shell
git clone https://github.com/nndeploy/nndeploy.git
cd nndeploy
# 拉取子模块
git submodule update --init --recursive
# 如果拉取子模块失败，调用克隆子模块脚本
python3 clone_submodule.py
```


## 2. 编译宏介绍

+ 参考[编译宏文档](./build_macro.md) 的详细介绍

包含了以下几类配置：

1. **基础构建选项（建议采用默认配置）**：如是否构建为共享库、使用的C++标准版本等等
2. **核心模块选项（建议采用默认配置）**：更细粒度控制需要编译的文件
3. **设备后端选项（按需打开，默认全部关闭，不依赖任何设备后端）**：如CUDA、OpenCL、各种NPU等硬件加速支持
4. **算子后端选项（按需打开，默认全部关闭，不依赖任何算子后端）**：如cudnn、onednn、xnnpack、qnnpack
5. **推理后端选项（按需打开，默认全部关闭，不依赖任何推理后端）**：如TensorRT、OpenVINO、ONNX Runtime、MNN等推理框架支持
6. **算法插件选项（建议采用默认配置，传统CV类算法打开，语言类和文生图类算法默认关闭）**：如检测、分割、llm、文生图等算法插件

    + 其中传统CV类算法依赖`OpenCV`，例如检测、分割、分类等，需要打开`ENABLE_NNDEPLOY_OPENCV`

    + **注意：其中`语言类和文生图类模型`依赖C++分词器[tokenizer-cpp](https://github.com/mlc-ai/tokenizers-cpp)，所以需要打开`ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP`，由于该库依赖rust，打开前**务必**参考[precompile_tokenizer_cpp.md](./precompile_tokenizer_cpp.md)**

> 注：所有后端均可选。三方库可使用自己的，也可使用nndeploy预编译版本：
> + [huggingface](https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party)：https://huggingface.co/alwaysssss/nndeploy/blob/main/third_party
> + [modelscope](https://www.modelscope.cn/models/nndeploy/third_party)：https://www.modelscope.cn/models/nndeploy/third_party


### 编译宏编辑规则

对于绝大部分编译选项，只用ON/OFF即可。

但对于外部依赖的三方库，有如下三种`使能并链接外部的第三方库的方法`

+ `方法一`：路径`path`，头文件以及库的根路径，其形式必须修改为
  + 头文件：`path/include`
  + 库：`path/lib `
  + windows dll: `path/bin`
  + 相应的库：ONNXRuntime、OpenVINO、TNN、MNN、Window已经编译好的OpenCV的库
+ `方法二`：开关`ON`，如果你安装了该库，并且可以通过find_package找到该库，可以采用该方式
  + 相应的库：Linux平台下的CUDA、CUDNN、TenosrRT、OpenCV
+ `方法三`：源码`ON`，使用源码编译该库，对应third_party目录下的库，可以采用该方式
  + 相应的库：tokenizer-cpp、rapidjson、gflags、ONNX

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
make install                # 在build目录下生成安装目录
```

> 注：不同平台编译方式
> - **Linux**：使用上述 make 命令编译
> - **Windows**：使用 Visual Studio 编译
> - **Android**：使用 android-ndk 交叉编译
>   ```shell
>   cmake .. -DCMAKE_TOOLCHAIN_FILE=<NDK_PATH>/build/cmake/android.toolchain.cmake \
>            -DANDROID_ABI=arm64-v8a \
>            -DANDROID_STL=c++_static \
>            -DANDROID_NATIVE_API_LEVEL=android-14
>   ```
> - **macOS/iOS**：使用 Xcode 编译

### 提供的编译脚本

为了简化编译过程，nndeploy 提供了针对不同平台的编译脚本：

- `build_linux.py`: Linux 平台编译脚本
- `build_mac_arm64.py`: macOS ARM64 平台编译脚本  
- `build_win.py`: Windows x86_64 平台编译脚本

> **注意**：使用编译脚本需要稳定的网络连接以访问 GitHub 下载依赖库。如果网络环境受限或无法正常访问 GitHub，建议采用上述手动编译方式进行构建。采用脚本编译时间很长，请耐心等待。

> **编译配置说明**：
> - 对应的配置文件：[cmake/config_opencv_ort_mnn_tokenizer.cmake](../../../cmake/config_opencv_ort_mnn_tokenizer.cmake)
> - 这些脚本使用与 Python 包相同的编译选项配置
> - 默认启用的推理后端：ONNXRuntime、MNN
> - 默认启用的依赖库：OpenCV、tokenizer-cpp

## 4. 主库编译

+ 默认编译产物为：libnndeploy_framework.so(Windows下为nndeploy_framework.dll)
+ 算法插件编译产物为：libnndeploy_plugin_xxx.so(Windows下为nndeploy_plugin_xxx.dll)
+ 可执行程序编译产物为：nndeploy_demo_xxx(Windows下为nndeploy_demo_xxx.exe)

> 注：xxx代表特定算法插件和特定的可执行程序，例如：nndeploy_plugin_detect.so、nndeploy_demo_detect、nndeploy_demo_dag  

## 5. 第三方库官方编译文档以及下载链接

|                        第三方库                         |  主版本  |                                          编译文档                                           |                                                                               官方库下载链接                                                                               |         备注         |
| :-----------------------------------------------------: | :------: | :-----------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------: |
|       [opencv](https://github.com/opencv/opencv)        |  4.10.0   |                           [链接](https://opencv.org/get-started/)                           |                                                                  [链接](https://opencv.org/get-started/)                                                                   |                      |
|     [TensorRT](https://github.com/NVIDIA/TensorRT)      | 8.6.0.12 |  [链接](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing)  |                                                            [链接](https://developer.nvidia.com/zh-cn/tensorrt)                                                             | 支持jetson-orin-nano |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | 2023.0.1 |      [链接](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md)      | [链接](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?ENVIRONMENT=RUNTIME&OP_SYSTEM=MACOS&VERSION=v_2023_0_1&DISTRIBUTION=ARCHIVE) |                      |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | v1.15.1  | [链接](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_useful_api.zh.md) |                                                   [链接](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1)                                                    |                      |
|          [MNN](https://github.com/alibaba/MNN)          |  2.6.2   |            [链接](https://mnn-docs.readthedocs.io/en/latest/compile/engine.html)            |                                                         [链接](https://github.com/alibaba/MNN/releases/tag/2.6.0)                                                          |                      |
|          [TNN](https://github.com/Tencent/TNN)          |  v0.3.0  |          [链接](https://github.com/Tencent/TNN/blob/master/doc/cn/user/compile.md)          |                                                         [链接](https://github.com/Tencent/TNN/releases/tag/v0.3.0)                                                         |                      |
|        [ncnn](https://github.com/Tencent/ncnn/)         |  v0.3.0  |            [链接](https://github.com/Tencent/ncnn/tree/master/docs/how-to-build)            |                                                       [链接](https://github.com/Tencent/ncnn/releases/tag/20230816)                                                        |                      |


> 注: 我们使用第三方库的上述版本，通常使用其他版本的也没有问题

## 6. Python安装

nndeploy 提供了完整的 Python API，支持快速部署和推理各种深度学习模型。

- 环境要求

  - Python 3.10+
  - 支持的操作系统：Linux(< Python3.13 && x86)、Windows、macOS(OS >=14 && ARM)，其他平台建议采用开发者模式

+ 安装方式

  + 方式一：PyPI 安装（推荐）

  适用于大多数用户的快速安装：

  ```bash
  pip install nndeploy
  ```

  + 方式二：源码编译安装

  适用于开发者用户：

  ```bash
  cd ../python
  pip install -e .
  ```

- 安装验证

  运行以下命令确认安装成功：

  ```bash
  python -c "import nndeploy; print(nndeploy.__version__)"
  ```

- 可能的问题描述

  - **conda环境冲突问题**
  
    当与conda环境产生冲突，无法正常运行时，可参考[解决方案脚本](https://github.com/nndeploy/nndeploy/blob/main/tool/script/fixed_sys_conda.sh)进行修复。

  - **系统环境保护问题**
  
    在某些系统中，为了保护系统Python环境的完整性，不允许直接在全局Python环境中安装第三方包。此时建议使用虚拟环境进行安装：
    
    ```bash
    # 创建虚拟环境
    python3 -m venv nndeploy_env
    
    # 激活虚拟环境
    source nndeploy_env/bin/activate
    
    # 在虚拟环境中安装
    pip install -e .
    ```

  - **动态库路径问题**
  
    如果运行时提示找不到相关动态库，需要将nndeploy的库路径添加到系统环境变量中：
    
    ```bash
    export LD_LIBRARY_PATH=path/to/nndeploy/python/nndeploy:$LD_LIBRARY_PATH
    ```
    
    其中`path/to/nndeploy`需要替换为实际的nndeploy安装路径。


## 7. 补充说明    

- TensorRT
  - 安装TensorRT cpp sdk [参考链接](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian)、cudnn、cuda、GPU driver
  - [Windows链接](https://zhuanlan.zhihu.com/p/476679322)
  - 安装前请确保 显卡驱动、cuda、cudnn均已安装且版本一致


- 在windows平台下，系统目录自带onnxruntime，故你在运行时或许可能会链接到系统目录下自带的onnxruntime，从而导致运行时出错。解决办法
  - 将你自己的onnxruntime库拷贝至build目录下


- 安装opencv
  - `sudo apt install libopencv-dev` [参考链接](https://cloud.tencent.com/developer/article/1657529)



      
      
      

  

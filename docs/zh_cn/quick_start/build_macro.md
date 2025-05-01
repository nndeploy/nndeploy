

# 编译宏文档

本文档列出了nndeploy项目中可用的编译宏选项，帮助开发者根据需求配置构建过程。

## 1. 编译宏编辑规则

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

## 2.基础构建选项（建议采用默认值）

| 宏 | 默认值 | 描述 |
|---|---|---|
| `ENABLE_NNDEPLOY_BUILD_SHARED` | ON | 构建为共享库而非静态库 |
| `ENABLE_NNDEPLOY_SYMBOL_HIDE` | OFF | 隐藏库中的符号以减小二进制大小 |
| `ENABLE_NNDEPLOY_COVERAGE` | OFF | 启用代码覆盖率分析 |
| `ENABLE_NNDEPLOY_CXX11_ABI` | OFF | 使用C++11 ABI |
| `ENABLE_NNDEPLOY_CXX14_ABI` | OFF | 使用C++14 ABI |
| `ENABLE_NNDEPLOY_CXX17_ABI` | ON | 使用C++17 ABI（推荐） |
| `ENABLE_NNDEPLOY_CXX20_ABI` | OFF | 使用C++20 ABI |
| `ENABLE_NNDEPLOY_OPENMP` | OFF | 启用OpenMP进行并行计算 |
| `ENABLE_NNDEPLOY_ADDRESS_SANTIZER` | OFF | 启用地址消毒器进行内存错误检测 |
| `ENABLE_NNDEPLOY_DOCS` | OFF | 构建文档 |
| `ENABLE_NNDEPLOY_TIME_PROFILER` | ON | 启用时间分析器进行性能分析 |
| `ENABLE_NNDEPLOY_RAPIDJSON` | ON | 启用RapidJSON进行JSON解析，基于path/to/nndeploy/third_party/rapidjson源码编译 |

## 3. 核心模块选项（建议采用默认值）

| 宏 | 默认值 | 描述 |
|---|---|---|
| `ENABLE_NNDEPLOY_BASE` | ON | 启用包含基础工具的基础模块 |
| `ENABLE_NNDEPLOY_THREAD_POOL` | ON | 启用线程池进行并行任务执行 |
| `ENABLE_NNDEPLOY_CRYPTION` | OFF | 启用加密模块进行模型加密/解密（尚未实现） |
| `ENABLE_NNDEPLOY_DEVICE` | ON | 启用设备模块 |
| `ENABLE_NNDEPLOY_IR` | ON | 启用中间表示（IR）模块 |
| `ENABLE_NNDEPLOY_IR_ONNX` | ON | 在IR模块中启用ONNX格式支持 |
| `ENABLE_NNDEPLOY_OP` | ON | 启用算子模块 |
| `ENABLE_NNDEPLOY_OP_ASCEND_C` | OFF | 启用Ascend C算子 |
| `ENABLE_NNDEPLOY_NET` | ON | 启用神经网络图表示 |
| `ENABLE_NNDEPLOY_INFERENCE` | ON | 启用推理模块 |
| `ENABLE_NNDEPLOY_INFERENCE_DEFAULT` | ON | 启用默认推理后端 |
| `ENABLE_NNDEPLOY_DAG` | ON | 启用有向无环图用于模型流水线 |
| `ENABLE_NNDEPLOY_PLUGIN` | ON | 启用算法插件 |
| `ENABLE_NNDEPLOY_TEST` | ON | 构建测试用例 |
| `ENABLE_NNDEPLOY_DEMO` | ON | 构建执行程序 |
| `ENABLE_NNDEPLOY_PYTHON` | ON | 构建Python绑定 |

## 4. 设备后端选项（可选项，默认全部关闭，可以不依赖任何设备后端）

| 宏 | 默认值 | 描述 |
|---|---|---|
| `ENABLE_NNDEPLOY_DEVICE_CUDA` | OFF | 启用NVIDIA CUDA GPU支持，启用后，也必须打开ENABLE_NNDEPLOY_DEVICE_CUDNN |
| `ENABLE_NNDEPLOY_DEVICE_ROCM` | OFF | 启用AMD ROCm GPU支持 |
| `ENABLE_NNDEPLOY_DEVICE_SYCL` | OFF | 启用SYCL跨平台加速 |
| `ENABLE_NNDEPLOY_DEVICE_OPENCL` | OFF | 启用OpenCL加速 |
| `ENABLE_NNDEPLOY_DEVICE_OPENGL` | OFF | 启用OpenGL计算支持 |
| `ENABLE_NNDEPLOY_DEVICE_METAL` | OFF | 启用Apple Metal GPU加速 |
| `ENABLE_NNDEPLOY_DEVICE_VULKAN` | OFF | 启用Vulkan计算支持 |
| `ENABLE_NNDEPLOY_DEVICE_HEXAGON` | OFF | 启用高通Hexagon DSP支持 |
| `ENABLE_NNDEPLOY_DEVICE_MTK_VPU` | OFF | 启用联发科VPU支持 |
| `ENABLE_NNDEPLOY_DEVICE_ASCEND_CL` | OFF | 启用华为Ascend计算语言支持 |
| `ENABLE_NNDEPLOY_DEVICE_APPLE_NPU` | OFF | 启用Apple神经引擎支持 |
| `ENABLE_NNDEPLOY_DEVICE_QUALCOMM_NPU` | OFF | 启用高通NPU支持 |
| `ENABLE_NNDEPLOY_DEVICE_MTK_NPU` | OFF | 启用联发科NPU支持 |
| `ENABLE_NNDEPLOY_DEVICE_SOPHON_NPU` | OFF | 启用算丰NPU支持 |

## 5. 算子后端选项（可选项，默认关闭，可以不依赖任何算子后端）

| 宏 | 默认值 | 描述 |
|---|---|---|
| `ENABLE_NNDEPLOY_DEVICE_CUDNN` | OFF | 启用NVIDIA cuDNN加速库 |

## 6. 推理后端选项（可选项，默认关闭，可以不依赖任何推理后端）

| 宏 | 默认值 | 描述 |
|---|---|---|
| `ENABLE_NNDEPLOY_INFERENCE_TENSORRT` | OFF | 启用NVIDIA TensorRT后端，必须打开ENABLE_NNDEPLOY_DEVICE_CUDNN，ENABLE_NNDEPLOY_DEVICE_CUDA |
| `ENABLE_NNDEPLOY_INFERENCE_OPENVINO` | OFF | 启用Intel OpenVINO后端 |
| `ENABLE_NNDEPLOY_INFERENCE_COREML` | OFF | 启用Apple Core ML后端 |
| `ENABLE_NNDEPLOY_INFERENCE_TFLITE` | OFF | 启用TensorFlow Lite后端 |
| `ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME` | OFF | 启用ONNX Runtime后端 |
| `ENABLE_NNDEPLOY_INFERENCE_NCNN` | OFF | 启用腾讯NCNN后端 |
| `ENABLE_NNDEPLOY_INFERENCE_TNN` | OFF | 启用腾讯TNN后端 |
| `ENABLE_NNDEPLOY_INFERENCE_MNN` | OFF | 启用阿里巴巴MNN后端 |
| `ENABLE_NNDEPLOY_INFERENCE_TVM` | OFF | 启用Apache TVM后端 |
| `ENABLE_NNDEPLOY_INFERENCE_PADDLELITE` | OFF | 启用百度PaddleLite后端 |
| `ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1` | OFF | 启用瑞芯微RKNN Toolkit 1.x后端 |
| `ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2` | OFF | 启用瑞芯微RKNN Toolkit 2.x后端 |
| `ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL` | OFF | 启用华为Ascend CL后端，必须打开ENABLE_NNDEPLOY_DEVICE_ASCEND_CL |
| `ENABLE_NNDEPLOY_INFERENCE_SNPE` | OFF | 启用高通SNPE后端 |
| `ENABLE_NNDEPLOY_INFERENCE_QNN` | OFF | 启用高通Neural Network后端 |
| `ENABLE_NNDEPLOY_INFERENCE_SOPHON` | OFF | 启用算丰后端 |
| `ENABLE_NNDEPLOY_INFERENCE_TORCH` | OFF | 启用PyTorch/LibTorch后端 |
| `ENABLE_NNDEPLOY_INFERENCE_TENSORFLOW` | OFF | 启用TensorFlow后端 |
| `ENABLE_NNDEPLOY_INFERENCE_NEUROPILOT` | OFF | 启用联发科NeuroPilot后端 |

## 7. 算法插件选项

| 宏 | 默认值 | 描述 |
|---|---|---|
| `ENABLE_NNDEPLOY_OPENCV` | ON or `path` | 启用OpenCV支持（多个算法部署插件需要，强依赖OpenCV） |
| `ENABLE_NNDEPLOY_PLUGIN_PREPROCESS` | ON | 启用预处理插件 |
| `ENABLE_NNDEPLOY_PLUGIN_INFER` | ON | 启用推理插件 |
| `ENABLE_NNDEPLOY_PLUGIN_CODEC` | ON | 启用编解码插件 |
| `ENABLE_NNDEPLOY_PLUGIN_TOKENIZER` | ON | 启用分词器插件 |
| `ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP` | OFF | 启用C++分词器插件, 依赖`tokenizer-cpp`，基于path/to/nndeploy/third_party/tokenizer-cpp源码编译 |
| `ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION` | ON | 启用分类插件 |
| `ENABLE_NNDEPLOY_PLUGIN_LLM` | OFF | 启用大型语言模型插件 |
| `ENABLE_NNDEPLOY_PLUGIN_DETECT` | ON | 启用检测插件 |
| `ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR` | ON | 启用DETR检测插件 |
| `ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO` | ON | 启用YOLO检测插件 |
| `ENABLE_NNDEPLOY_PLUGIN_SEGMENT` | ON | 启用分割插件 |
| `ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING` | ON | 启用Segment Anything分割插件 |
| `ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RMBG` | ON | 启用背景移除分割插件 |
| `ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION` | OFF | 启用Stable Diffusion插件 |

## 8. 编译选项举例说明
+ `nndeploy`通过路径的方式链接推理后端`ONNXRuntime`。`set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "path/to/onnxruntime")`，如果你想启用并链接其他推理后端（OpenVINO、MNN、TNN …），也可做同样的处理
+ `nndeploy`通过find_package的方式链接推理后端`TensorRT`。`set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT ON)`，对于其他可以通过find_package找到的库，也可做同样的处理
+ `nndeploy`通过源码的方式链接三方库[tokenizer-cpp](https://github.com/mlc-ai/tokenizers-cpp)。`set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP ON)`，对于其他可以通过源码编译的库，也做同样的处理（目前所有third_party中的库都是通过源码编译的）
+ 编译算法的插件。首先将模型类别`set(NABLE_NNDEPLOY_PLUGIN_XXX ON)`，再将具体的模型`set(NABLE_NNDEPLOY_PLUGIN_XXX_YYY ON)`







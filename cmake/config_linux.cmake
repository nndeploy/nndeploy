# --------------------------------------------------------------------
# Template custom cmake config for compiling
#
# This file is used to override the build sets in build.
# If you want to change the config, please use the following
# steps. Assume you are off the root directory. First copy the this
# file so that any local changes will be ignored by git
#
# $ mkdir build
# $ cp cmake/config.cmake build
# $ cd build
# $ cmake ..
# $ make -j8
# --------------------------------------------------------------------
# common
set(ENABLE_NNDEPLOY_BUILD_SHARED ON) # 是否编译为动态库，默认ON
set(ENABLE_NNDEPLOY_SYMBOL_HIDE ON) # 符号表是否隐藏，默认为ON
set(ENABLE_NNDEPLOY_COVERAGE OFF) # 是否使能代码覆盖率分析，默认为OFF
set(ENABLE_NNDEPLOY_CXX11_ABI OFF) # C++的版本，选择为C++11，默认为ON
set(ENABLE_NNDEPLOY_CXX14_ABI OFF) # C++的版本，选择为C++14，默认为OFF
set(ENABLE_NNDEPLOY_CXX17_ABI ON) # C++的版本，选择为C++17，默认为OFF
set(ENABLE_NNDEPLOY_CXX20_ABI OFF) # C++的版本，选择为C++20，默认为OFF
set(ENABLE_NNDEPLOY_OPENMP ON) # 否使用OpenMP，该选项在Mac/iOS平台无效，默认为ON
set(ENABLE_NNDEPLOY_ADDRESS_SANTIZER OFF) # 内存泄露检测，默认为OFF
set(ENABLE_NNDEPLOY_TIME_PROFILER ON) # 时间性能Profile，默认为ON
set(ENABLE_NNDEPLOY_OPENCV ON) # 是否链接第三方库opencv，默认为OFF
set(NNDEPLOY_OPENCV_LIBS) # 链接的具体的opencv库名称，例如opencv_world480，opencv_java4等
set(ENABLE_NNDEPLOY_RAPIDJSON ON)

# # base
set(ENABLE_NNDEPLOY_BASE ON) # 是否编译base目录中文件，默认为ON

# # thread
set(ENABLE_NNDEPLOY_THREAD_POOL ON) # 是否编译thread_pool目录中文件，默认为ON

# # cryption
set(ENABLE_NNDEPLOY_CRYPTION OFF) # 是否编译crytion目录中文件，默认为ON

# # device
set(ENABLE_NNDEPLOY_DEVICE ON) # 是否编译device目录中文件，默认为ON
set(ENABLE_NNDEPLOY_DEVICE_CPU ON) # 是否使能device cpu，默认为ON
set(ENABLE_NNDEPLOY_DEVICE_ARM OFF) # 是否使能device arm，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_X86 ON) # 是否使能device x86，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_CUDA ON) # 是否使能device cuda，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_CUDNN ON) # 是否使能device cudnn，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_OPENCL OFF) # 是否使能device opencl，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_OPENGL OFF) # 是否使能device opengl，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_METAL OFF) # 是否使能device metal，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_APPLE_NPU OFF) # 是否使能device apple npu，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_HVX OFF) # 是否使能device apple hvx，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_MTK_VPU OFF) # 是否使能device apple hvx，默认为OFF
set(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL OFF) # 是否使能device apple ascend cl，默认为OFF

# # ir
set(ENABLE_NNDEPLOY_IR ON) # 是否编译ir目录中文件，默认为OFF
set(ENABLE_NNDEPLOY_IR_ONNX ON) # 是否编译ir目录中文件，默认为OFF

# # op
set(ENABLE_NNDEPLOY_OP ON) # 是否编译op目录中文件，默认为OFF

# # net
set(ENABLE_NNDEPLOY_NET ON) # 是否编译forward目录中文件，默认为OFF

# # inference
set(ENABLE_NNDEPLOY_INFERENCE ON) # 是否编译inference目录中文件，默认为ON
set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT ON) # 是否使能INFERENCE TENSORRT，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/libopenvino.so.2023.1") # 是否使能INFERENCE OPENVINO，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_COREML OFF) # 是否使能INFERENCE COREML，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_TFLITE OFF) # 是否使能INFERENCE TFLITE，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/onnxruntime-linux-x64-1.15.1") # 是否使能INFERENCE ONNXRUNTIME，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_NCNN "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/ncnn-20230816-ubuntu-2204-shared") # 是否使能INFERENCE NCNN，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_TNN "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/tnn-v0.3.0") # 是否使能INFERENCE TNN，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_MNN "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/mnn_2.6.0_linux_x64_cpu_opencl") # 是否使能INFERENCE MNN，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_TVM OFF) # 是否使能INFERENCE TVM，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE OFF) # 是否使能INFERENCE PADDLELITE，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1 OFF) # 是否使能INFERENCE RKNN_TOOLKIT_1，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2 OFF) # 是否使能INFERENCE RKNN_TOOLKIT_2，默认为OFF
set(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL OFF) # 是否使能INFERENCE ASCEND_CL，默认为OFF

# # dag
set(ENABLE_NNDEPLOY_DAG ON) # 是否编译dag目录中文件，默认为ON

# plugin
set(ENABLE_NNDEPLOY_PLUGIN ON) # 是否编译plugin目录中文件，默认为ON

# test
set(ENABLE_NNDEPLOY_TEST OFF) # 是否使能单元测试，默认为OFF

# demo
set(ENABLE_NNDEPLOY_DEMO ON) # 是否使能可执行程序demo，默认为OFF

# enable python api
set(ENABLE_NNDEPLOY_PYTHON ON) # ON 表示构建nndeploy的python接口

# plugin
# # preprocess
set(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS ON) # 是否编译plugin目录中文件，默认为ON

# # infer
set(ENABLE_NNDEPLOY_PLUGIN_INFER ON) # 是否编译plugin目录中文件，默认为ON

# # codec
set(ENABLE_NNDEPLOY_PLUGIN_CODEC ON) # 是否编译plugin目录中文件，默认为ON

# # classification
set(ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION ON)

# # detect
set(ENABLE_NNDEPLOY_PLUGIN_DETECT ON)
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR OFF)
set(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO ON)

# # segment
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT ON)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING OFF)
set(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RMBG ON)

# # tokenizer
set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER OFF)
set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP OFF)

# # stable_diffusion
set(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION OFF)

# demo
set(ENABLE_NNDEPLOY_DEMO_LLAMA OFF)

set(ENABLE_NNDEPLOY_DEMO_TENSOR_POOL OFF)

set(ENABLE_NNDEPLOY_DEMO_RESNET OFF)
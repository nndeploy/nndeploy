# --------------------------------------------------------------------
# Template custom cmake config for compiling
#
# This file is used to override the build sets in build.
# If you want to change the config, please use the following
# steps. Assume you are off the root directory. First copy the this
# file so that any local changes will be ignored by git
#
# $ mkdir build
# $ cp cmake/config_nndeploy.cmake build
# $ cd build
# $ mv config_nndeploy.cmake config.cmake
# $ cmake ..
# $ make -j8
# --------------------------------------------------------------------
# common
set(ENABLE_NNDEPLOY_BUILD_SHARED ON)
set(ENABLE_NNDEPLOY_SYMBOL_HIDE ON)
set(ENABLE_NNDEPLOY_COVERAGE OFF)
set(ENABLE_NNDEPLOY_CXX11_ABI ON)
set(ENABLE_NNDEPLOY_CXX14_ABI OFF)
set(ENABLE_NNDEPLOY_CXX17_ABI OFF)
set(ENABLE_NNDEPLOY_CXX20_ABI OFF)
set(ENABLE_NNDEPLOY_OPENMP ON)
set(ENABLE_NNDEPLOY_ADDRESS_SANTIZER OFF)
set(ENABLE_NNDEPLOY_DOCS OFF)
set(ENABLE_NNDEPLOY_TIME_PROFILER ON)
set(ENABLE_NNDEPLOY_OPENCV ON)
set(NNDEPLOY_OPENCV_LIBS)
## base
set(ENABLE_NNDEPLOY_BASE ON)
## thread
set(ENABLE_NNDEPLOY_THREAD_POOL ON)
## cryption
set(ENABLE_NNDEPLOY_CRYPTION OFF)
## device
set(ENABLE_NNDEPLOY_DEVICE ON)
set(ENABLE_NNDEPLOY_DEVICE_CPU ON)
set(ENABLE_NNDEPLOY_DEVICE_ARM OFF)
set(ENABLE_NNDEPLOY_DEVICE_X86 ON)
set(ENABLE_NNDEPLOY_DEVICE_CUDA ON)
set(ENABLE_NNDEPLOY_DEVICE_CUDNN ON)
set(ENABLE_NNDEPLOY_DEVICE_OPENCL OFF)
set(ENABLE_NNDEPLOY_DEVICE_OPENGL OFF)
set(ENABLE_NNDEPLOY_DEVICE_METAL OFF)
set(ENABLE_NNDEPLOY_DEVICE_APPLE_NPU OFF)
set(ENABLE_NNDEPLOY_DEVICE_HVX OFF)
set(ENABLE_NNDEPLOY_DEVICE_MTK_VPU OFF)
set(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL OFF)
## op
set(ENABLE_NNDEPLOY_OP OFF)
set(ENABLE_NNDEPLOY_OP_NN OFF)
set(ENABLE_NNDEPLOY_OP_CV OFF)
set(ENABLE_NNDEPLOY_OP_AUDIO OFF)
## forward
set(ENABLE_NNDEPLOY_FORWARD OFF)
## inference
set(ENABLE_NNDEPLOY_INFERENCE ON)
set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT ON)
set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/libopenvino.so.2023.1")
set(ENABLE_NNDEPLOY_INFERENCE_COREML OFF)
set(ENABLE_NNDEPLOY_INFERENCE_TFLITE OFF)
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/onnxruntime-linux-x64-1.15.1")
set(ENABLE_NNDEPLOY_INFERENCE_NCNN "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/ncnn-20230816-ubuntu-2204-shared")
set(ENABLE_NNDEPLOY_INFERENCE_TNN "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/tnn-v0.3.0")
set(ENABLE_NNDEPLOY_INFERENCE_MNN "/home/always/huggingface/nndeploy/third_party/ubuntu22.04_x64/mnn_2.6.0_linux_x64_cpu_opencl")
set(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE "/data/qixuxiang/Paddle-Lite")
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1 "/home/toybrick/Documents/nndeploy/third_party/librknn_api_3399_aarch64")
set(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2 OFF)
set(ENABLE_NNDEPLOY_AICOMPILER_TVM OFF)
set(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE OFF)
set(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL OFF)
## dag
set(ENABLE_NNDEPLOY_DAG ON)
## model
set(ENABLE_NNDEPLOY_MODEL ON)
## test
set(ENABLE_NNDEPLOY_TEST OFF)
## demo
set(ENABLE_NNDEPLOY_DEMO OFF)

## model detect
set(ENABLE_NNDEPLOY_MODEL_DETECT ON)
set(ENABLE_NNDEPLOY_MODEL_DETECT_DETR OFF)
set(ENABLE_NNDEPLOY_MODEL_DETECT_YOLO ON)

## model segment
set(ENABLE_NNDEPLOY_MODEL_SEGMENT OFF)
set(ENABLE_NNDEPLOY_MODEL_SEGMENT_SEGMENT_ANYTHING OFF)
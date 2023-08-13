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
set(ENABLE_NNDEPLOY_COVERAGE ON)
set(ENABLE_NNDEPLOY_CXX11_ABI ON)
set(ENABLE_NNDEPLOY_CXX14_ABI OFF)
set(ENABLE_NNDEPLOY_CXX17_ABI OFF)
set(ENABLE_NNDEPLOY_OPENMP OFF)
set(ENABLE_NNDEPLOY_ADDRESS_SANTIZER ON)
set(ENABLE_NNDEPLOY_DOCS OFF)
set(ENABLE_NNDEPLOY_TIME_PROFILER ON)
set(ENABLE_NNDEPLOY_OPENCV ON)
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
set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO "/home/always/github/openvino/install/runtime")
set(ENABLE_NNDEPLOY_INFERENCE_COREML OFF)
set(ENABLE_NNDEPLOY_INFERENCE_TFLITE OFF)
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "/home/for_all_users/third_party/user/linux/onnxruntime-linux-x64-1.15.1")
set(ENABLE_NNDEPLOY_INFERENCE_NCNN OFF)
set(ENABLE_NNDEPLOY_INFERENCE_TNN "/home/for_all_users/third_party/user/linux/tnn-v0.3.0")
set(ENABLE_NNDEPLOY_INFERENCE_MNN "/home/for_all_users/third_party/user/linux/mnn_2.4.0_linux_x64_cpu_opencl")
set(ENABLE_NNDEPLOY_AICOMPILER_TVM OFF)
## model
set(ENABLE_NNDEPLOY_MODEL ON)
## test
set(ENABLE_NNDEPLOY_TEST OFF)
## demo
set(ENABLE_NNDEPLOY_DEMO OFF)

## model detect
set(ENABLE_NNDEPLOY_MODEL_DETECT ON)
set(ENABLE_NNDEPLOY_MODEL_DETECT_DETR OFF)
set(ENABLE_NNDEPLOY_MODEL_DETECT_YOLOV5 OFF)
set(ENABLE_NNDEPLOY_MODEL_DETECT_MEITUAN_YOLOV6 ON)

## model segment
set(ENABLE_NNDEPLOY_MODEL_SEGMENT ON)
set(ENABLE_NNDEPLOY_MODEL_SEGMENT_SEGMENT_ANYTHING OFF)
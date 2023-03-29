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
#
# Next modify the according entries, and then compile by
#
# $ cd build
# $ cmake ..
#
# Then build in parallel with 8 threads
#
# $ make -j8
# --------------------------------------------------------------------
# common
set(NNDEPLOY_ENABLE_BUILD_SHARED ON)
set(NNDEPLOY_ENABLE_SYMBOL_HIDE OFF)
set(NNDEPLOY_ENABLE_COVERAGE OFF)
set(NNDEPLOY_ENABLE_CXX11_ABI ON)
set(NNDEPLOY_ENABLE_CXX14_ABI OFF)
set(NNDEPLOY_ENABLE_CXX17_ABI OFF)
set(NNDEPLOY_ENABLE_OPENMP OFF)
set(NNDEPLOY_ENABLE_VALGRIND OFF)
set(NNDEPLOY_ENABLE_DOCS OFF)
# nndeploy
set(NNDEPLOY_ENABLE ON)
## base
set(NNDEPLOY_ENABLE_BASE ON)
## cryption
set(NNDEPLOY_ENABLE_CRYPTION OFF)
## device
set(NNDEPLOY_ENABLE_DEVICE ON)
set(NNDEPLOY_ENABLE_DEVICE_CPU OFF)
set(NNDEPLOY_ENABLE_DEVICE_ARM OFF)
set(NNDEPLOY_ENABLE_DEVICE_X86 ON)
set(NNDEPLOY_ENABLE_DEVICE_CUDA OFF)
set(NNDEPLOY_ENABLE_DEVICE_OPENCL OFF)
set(NNDEPLOY_ENABLE_DEVICE_OPENGL OFF)
set(NNDEPLOY_ENABLE_DEVICE_METAL OFF)
set(NNDEPLOY_ENABLE_DEVICE_APPLE_NPU OFF)
set(NNDEPLOY_ENABLE_DEVICE_APPLE_NPU OFF)
set(NNDEPLOY_ENABLE_DEVICE_APPLE_NPU OFF)
## taskflow
set(NNDEPLOY_ENABLE_TASKFLOW OFF)
## audio
set(NNDEPLOY_ENABLE_AUDIO OFF)
set(NNDEPLOY_ENABLE_AUDIO_CORE OFF)
## cv
set(NNDEPLOY_ENABLE_CV OFF)
set(NNDEPLOY_ENABLE_CV_CORE OFF)
## inference
set(NNDEPLOY_ENABLE_INFERENCE ON)
set(NNDEPLOY_ENABLE_INFERENCE_DEFAULT OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TENSORRT OFF)
set(NNDEPLOY_ENABLE_INFERENCE_OPENVINO OFF)
set(NNDEPLOY_ENABLE_INFERENCE_COREML OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TFLITE OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TNN OFF)
set(NNDEPLOY_ENABLE_INFERENCE_MNN "/home/always/github/nndeploy/third_party/user/linux/mnn_2.4.0_linux_x64_cpu_opencl")
## aicompiler
set(NNDEPLOY_ENABLE_AICOMPILER OFF)
set(NNDEPLOY_ENABLE_AICOMPILER_DEFAULT OFF)
set(NNDEPLOY_ENABLE_AICOMPILER_TVM OFF)
## test
set(NNDEPLOY_ENABLE_TEST OFF)
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
set(ENABLE_NNDEPLOY_BUILD_SHARED OFF)
set(ENABLE_NNDEPLOY_SYMBOL_HIDE OFF)
set(ENABLE_NNDEPLOY_COVERAGE OFF)
set(ENABLE_NNDEPLOY_CXX11_ABI ON)
set(ENABLE_NNDEPLOY_CXX14_ABI OFF)
set(ENABLE_NNDEPLOY_CXX17_ABI OFF)
set(ENABLE_NNDEPLOY_OPENMP OFF)
set(ENABLE_NNDEPLOY_VALGRIND OFF)
set(ENABLE_NNDEPLOY_DOCS OFF)
set(ENABLE_NNDEPLOY_OPENCV OFF)
## base
set(ENABLE_NNDEPLOY_BASE ON)
## thread
set(ENABLE_NNDEPLOY_THREAD ON)
## cryption
set(ENABLE_NNDEPLOY_CRYPTION OFF)
## device
set(ENABLE_NNDEPLOY_DEVICE ON)
set(ENABLE_NNDEPLOY_DEVICE_CPU ON)
set(ENABLE_NNDEPLOY_DEVICE_ARM OFF)
set(ENABLE_NNDEPLOY_DEVICE_X86 ON)
set(ENABLE_NNDEPLOY_DEVICE_CUDA OFF)
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
set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT OFF)
set(ENABLE_NNDEPLOY_INFERENCE_OPENVINO OFF)
set(ENABLE_NNDEPLOY_INFERENCE_COREML OFF)
set(ENABLE_NNDEPLOY_INFERENCE_TFLITE OFF)
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME OFF)
set(ENABLE_NNDEPLOY_INFERENCE_NCNN OFF)
set(ENABLE_NNDEPLOY_INFERENCE_TNN OFF)
set(ENABLE_NNDEPLOY_INFERENCE_MNN OFF)
set(ENABLE_NNDEPLOY_AICOMPILER_TVM OFF)
## task
set(ENABLE_NNDEPLOY_TASK OFF)
## test
set(ENABLE_NNDEPLOY_TEST OFF)
## demo
set(ENABLE_NNDEPLOY_DEMO OFF)
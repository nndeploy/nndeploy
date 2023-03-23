# --------------------------------------------------------------------
# Template custom cmake cofffiguratioff for compiling
#
# This file is used to override the build sets in build.
# If you want to change the cofffiguratioff, please use the following
# steps. Assume you are off the root directory. First copy the this
# file so that any local changes will be ignored by git
#
# $ mkdir build
# $ cp cmake/cofffig.cmake build
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
set(NNDEPLOY_ENABLE_BUILD_SHARED "" OFF)
set(NNDEPLOY_ENABLE_SYMBOL_HIDE "" OFF)
set(NNDEPLOY_ENABLE_COVERAGE "" OFF)
set(NNDEPLOY_ENABLE_CXX11_ABI "" ON)
set(NNDEPLOY_ENABLE_CXX14_ABI "" OFF)
set(NNDEPLOY_ENABLE_OPENMP "" OFF)
set(NNDEPLOY_ENABLE_VALGRIND "" OFF)
set(NNDEPLOY_ENABLE_DOCS "" OFF)
# nndeploy
set(NNDEPLOY_ENABLE "" ON)
# # base
set(NNDEPLOY_ENABLE_BASE "" ON)
# # cryptioff
set(NNDEPLOY_ENABLE_CRYPTIOFF "" OFF)
# # device
set(NNDEPLOY_ENABLE_DEVICE "" ON)
set(NNDEPLOY_ENABLE_DEVICE_CPU "" OFF)
set(NNDEPLOY_ENABLE_DEVICE_ARM "" OFF)
set(NNDEPLOY_ENABLE_DEVICE_X86 "" OFF)
set(NNDEPLOY_ENABLE_DEVICE_CUDA "" OFF)
set(NNDEPLOY_ENABLE_DEVICE_OPENCL "" OFF)
set(NNDEPLOY_ENABLE_DEVICE_OPENGL "" OFF)
set(NNDEPLOY_ENABLE_DEVICE_METAL "" OFF)
set(NNDEPLOY_ENABLE_DEVICE_APPLE_NPU "" OFF)
# # audio
set(NNDEPLOY_ENABLE_AUDIO "" OFF)
set(NNDEPLOY_ENABLE_AUDIO_CORE "" OFF)
# # cv
set(NNDEPLOY_ENABLE_CV "" OFF)
set(NNDEPLOY_ENABLE_CV_CORE "" OFF)
# # inference
set(NNDEPLOY_ENABLE_INFERENCE "" OFF)
set(NNDEPLOY_ENABLE_INFERENCE_DEFAULT "" OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TENSORRT "" OFF)
set(NNDEPLOY_ENABLE_INFERENCE_OPENVINO "" OFF)
set(NNDEPLOY_ENABLE_INFERENCE_COREML "" OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TFLITE OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TNN "" OFF)
# # aicompiler
set(NNDEPLOY_ENABLE_AICOMPILER "" OFF)
set(NNDEPLOY_ENABLE_AICOMPILER_DEFAULT "" OFF)
# # graph
set(NNDEPLOY_ENABLE_TASK "" OFF)
set(NNDEPLOY_ENABLE_TEST "" OFF)
# nntask
set(NNTASK_ENABLE "" ON)
# #
set(NNTASK_ENABLE_TEST "" OFF)
set(NNTASK_ENABLE_DEMO "" ON)

set(NNTASK_ENABLE_ALWAYS "" ON)
#--------------------------------------------------------------------
#  Template custom cmake configuration for compiling
#
#  This file is used to override the build options in build.
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ mkdir build
#  $ cp cmake/config.cmake build
#
#  Next modify the according entries, and then compile by
#
#  $ cd build
#  $ cmake ..
#
#  Then build in parallel with 8 threads
#
#  $ make -j8
#--------------------------------------------------------------------

# common
set(NNDEPLOY_ENABLE_BUILD_SHARED ON)
set(NNDEPLOY_ENABLE_SYMBOL_HIDE ON)
set(NNDEPLOY_ENABLE_COVERAGE ON)
set(NNDEPLOY_ENABLE_CXX11_ABI ON)
set(NNDEPLOY_ENABLE_CXX14_ABI OFF)
set(NNDEPLOY_ENABLE_OPENMP ON)
set(NNDEPLOY_ENABLE_DEBUG OFF)
# device
set(NNDEPLOY_ENABLE_DEVICE_CPU ON)
set(NNDEPLOY_ENABLE_DEVICE_ARM OFF)
set(NNDEPLOY_ENABLE_DEVICE_X86 OFF)
set(NNDEPLOY_ENABLE_DEVICE_CUDA OFF)
set(NNDEPLOY_ENABLE_DEVICE_OPENCL OFF)
set(NNDEPLOY_ENABLE_DEVICE_OPENGL OFF)
set(NNDEPLOY_ENABLE_DEVICE_METAL OFF)
set(NNDEPLOY_ENABLE_DEVICE_APPLE_NPU OFF)
# inference
set(NNDEPLOY_ENABLE_INFERENCE_TENSORRT OFF)
set(NNDEPLOY_ENABLE_INFERENCE_OPENVINO OFF)
set(NNDEPLOY_ENABLE_INFERENCE_COREML OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TFLITE  OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TVM  OFF)
set(NNDEPLOY_ENABLE_INFERENCE_TNN OFF)
# audio
set(NNDEPLOY_ENABLE_AUDIO ON)
set(NNDEPLOY_ENABLE_AUDIO_CORE ON)
# vision
set(NNDEPLOY_ENABLE_VISION ON)
set(NNDEPLOY_ENABLE_VISION_CORE ON)
# nncore
set(NNDEPLOY_ENABLE_nncore ON)
# nntask
set(NNDEPLOY_ENABLE_NNTASK ON)
# nndemo
set(NNDEPLOY_ENABLE_NNDEMO ON)
# test
set(NNDEPLOY_ENABLE_UNITTEST OFF)


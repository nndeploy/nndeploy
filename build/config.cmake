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
set(NNKIT_ENABLE_BUILD_SHARED ON)
set(NNKIT_ENABLE_SYMBOL_HIDE ON)
set(NNKIT_ENABLE_COVERAGE ON)
set(NNKIT_ENABLE_CXX11_ABI ON)
set(NNKIT_ENABLE_CXX14_ABI OFF)
set(NNKIT_ENABLE_OPENMP OFF)
set(NNKIT_ENABLE_DEBUG OFF)
# device
set(NNKIT_ENABLE_DEVICE_CPU ON)
set(NNKIT_ENABLE_DEVICE_ARM OFF)
set(NNKIT_ENABLE_DEVICE_X86 OFF)
set(NNKIT_ENABLE_DEVICE_CUDA OFF)
set(NNKIT_ENABLE_DEVICE_OPENCL OFF)
set(NNKIT_ENABLE_DEVICE_METAL OFF)
set(NNKIT_ENABLE_DEVICE_APPLE_NPU OFF)
# inference
set(NNKIT_ENABLE_INFERENCE_TENSORRT OFF)
set(NNKIT_ENABLE_INFERENCE_OPENVINO OFF)
set(NNKIT_ENABLE_INFERENCE_COREML OFF)
set(NNKIT_ENABLE_INFERENCE_TFLITE  OFF)
set(NNKIT_ENABLE_INFERENCE_TVM  OFF)
set(NNKIT_ENABLE_INFERENCE_TNN OFF)
# audio
set(NNKIT_ENABLE_AUDIO ON)
# vision
set(NNKIT_ENABLE_VISION ON)
# nntask
set(NNKIT_ENABLE_NNTASK ON)
# test
set(NNKIT_ENABLE_UNITTEST OFF)

# leetcode
set(NNKIT_TASK_ENABLE_LEETCODE ON)

# xuanxuan
set(NNKIT_TASK_ENABLE_XUANXUAN ON)


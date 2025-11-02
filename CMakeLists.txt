cmake_minimum_required(VERSION 3.12)

project(nndeploy)

ENABLE_LANGUAGE(ASM C CXX)

set(NNDEPLOY_MAJOR_VERSION 3)
set(NNDEPLOY_MINOR_VERSION 0)
set(NNDEPLOY_PATCH_VERSION 4)
set(NNDEPLOY_VERSION "${NNDEPLOY_MAJOR_VERSION}.${NNDEPLOY_MINOR_VERSION}.${NNDEPLOY_PATCH_VERSION}")
add_definitions(-DNNDEPLOY_VERSION="${NNDEPLOY_VERSION}")

message(STATUS "CMAKE BEIGIN. NNDEPLOY_VERSION: ${NNDEPLOY_VERSION}")

# 1. Include utility and summary CMake files
include(cmake/util.cmake)
include(cmake/summary.cmake)

# If config.cmake file exists, compile according to this configuration file, otherwise use default configuration
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
endif()

# set
set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# 1. **Basic Build Options (Default Configuration Recommended)**: Such as whether to build as a shared library, C++ standard used, etc.
nndeploy_option(ENABLE_NNDEPLOY_BUILD_SHARED "Build as shared library instead of static library" ON)
nndeploy_option(ENABLE_NNDEPLOY_SYMBOL_HIDE "Hide symbols in the library to reduce binary size" OFF)
nndeploy_option(ENABLE_NNDEPLOY_COVERAGE "Enable code coverage analysis" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX11_ABI "Use C++11 ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX14_ABI "Use C++14 ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX17_ABI "Use C++17 ABI" ON) ## C++17 ABI is recommended
nndeploy_option(ENABLE_NNDEPLOY_CXX20_ABI "Use C++20 ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_OPENMP "Enable OpenMP for parallel computing" OFF)
nndeploy_option(ENABLE_NNDEPLOY_ADDRESS_SANTIZER "Enable Address Sanitizer for memory error detection" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DOCS "Build documentation" OFF)
nndeploy_option(ENABLE_NNDEPLOY_TIME_PROFILER "Enable time profiling for performance analysis" ON)
nndeploy_option(ENABLE_NNDEPLOY_GPU_MEM_TRACKER "Enable gpu max memory usage analysis" OFF)
nndeploy_option(ENABLE_NNDEPLOY_RAPIDJSON "Enable RapidJSON for JSON parsing" ON)

# 2. **Core Module Options (Default Configuration Recommended)**: Controls enabling of basic modules, thread pool, device modules and other core functionalities
# # Base module
nndeploy_option(ENABLE_NNDEPLOY_BASE "Enable base module with fundamental utilities" ON)
# # Thread pool
nndeploy_option(ENABLE_NNDEPLOY_THREAD_POOL "Enable thread pool for parallel task execution" ON)
# # Cryption module - Not yet implemented
nndeploy_option(ENABLE_NNDEPLOY_CRYPTION "Enable cryption module for model encryption/decryption" OFF)
# # Device module
nndeploy_option(ENABLE_NNDEPLOY_DEVICE "Enable device module" ON)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_CPU "Enable CPU device" ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ARM "Enable ARM-specific device" ON)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_X86 "Enable x86-specific device" OFF)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_RISC_V "Enable RISC-V device" OFF)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86|x86_64|amd64|AMD64)")
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ARM "Enable ARM-specific device" OFF)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_X86 "Enable x86-specific device" ON)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_RISC_V "Enable RISC-V device" OFF)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv)")
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ARM "Enable ARM-specific device" OFF)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_X86 "Enable x86-specific device" OFF)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_RISC_V "Enable RISC-V device" ON)
else()
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ARM "Enable ARM-specific device" OFF)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_X86 "Enable x86-specific device" OFF)
  nndeploy_option(ENABLE_NNDEPLOY_DEVICE_RISC_V "Enable RISC-V device" OFF)
endif()
# # Intermediate Representation (IR) module
nndeploy_option(ENABLE_NNDEPLOY_IR "Enable Intermediate Representation module" ON)
nndeploy_option(ENABLE_NNDEPLOY_IR_ONNX "Enable ONNX format support in IR module" OFF)
# # Operators module
nndeploy_option(ENABLE_NNDEPLOY_OP "Enable operators module" ON)
nndeploy_option(ENABLE_NNDEPLOY_OP_ASCEND_C "Enable Ascend C operators" OFF)
# # Net module
nndeploy_option(ENABLE_NNDEPLOY_NET "Enable neural network graph representation" ON)
# # Inference module
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE "Enable inference module" ON)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_DEFAULT "Enable default inference backend" ON)
# # Directed Acyclic Graph (DAG) module
nndeploy_option(ENABLE_NNDEPLOY_DAG "Enable Directed Acyclic Graph for model pipeline" ON)
# # Plugin system for Algorithm
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN "Enable plugin for model deployment" ON)
# # Test cases
nndeploy_option(ENABLE_NNDEPLOY_TEST "Build test cases" OFF)
# # Demo applications
nndeploy_option(ENABLE_NNDEPLOY_DEMO "Build demo applications" ON)
# # Python bindings
nndeploy_option(ENABLE_NNDEPLOY_PYTHON "Build Python bindings" ON)
# # JAVA bindings
nndeploy_option(ENABLE_NNDEPLOY_FFI_JAVA "Build Java bindings" OFF)
# # Swift bindings
nndeploy_option(ENABLE_NNDEPLOY_FFI_SWIFT "Build Swift bindings" OFF)
# # Go bindings
nndeploy_option(ENABLE_NNDEPLOY_FFI_GO "Build Go bindings" OFF)
# # C# bindings
nndeploy_option(ENABLE_NNDEPLOY_FFI_CSHARP "Build C# bindings" OFF)

# 3. **Device Backend Options (Enable as Needed, All Disabled by Default, No Device Backend Dependencies)**: Such as CUDA, OpenCL, various NPU hardware acceleration support
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_CUDA "Enable NVIDIA CUDA GPU support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ROCM "Enable AMD ROCm GPU support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_SYCL "Enable SYCL cross-platform acceleration" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_OPENCL "Enable OpenCL acceleration" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_OPENGL "Enable OpenGL compute support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_METAL "Enable Apple Metal GPU acceleration" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_VULKAN "Enable Vulkan compute support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_HEXAGON "Enable Qualcomm Hexagon DSP support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_MTK_VPU "Enable MediaTek VPU support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL "Enable Huawei Ascend Computing Language support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_APPLE_NPU "Enable Apple Neural Engine support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_QUALCOMM_NPU "Enable Qualcomm NPU support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_MTK_NPU "Enable MediaTek NPU support" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_SOPHON_NPU "Enable Sophon NPU support" OFF)

# 4. **Operator Backend Options (Enable as Needed, All Disabled by Default, No Operator Backend Dependencies)**: Such as cudnn
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_CUDNN "Enable NVIDIA cuDNN acceleration library" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_X86_ONEDNN "Enable Intel oneDNN acceleration library" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ARM_XNNPACK "Enable XNNPACK acceleration library" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ARM_QNNPACK "Enable QNNPACK acceleration library" OFF)

# 5. **Inference Backend Options (Enable as Needed, All Disabled by Default, No Inference Backend Dependencies)**: Such as TensorRT, OpenVINO, ONNX Runtime and other inference frameworks
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TENSORRT "Enable NVIDIA TensorRT backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_OPENVINO "Enable Intel OpenVINO backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_COREML "Enable Apple Core ML backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TFLITE "Enable TensorFlow Lite backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "Enable ONNX Runtime backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_NCNN "Enable Tencent NCNN backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TNN "Enable Tencent TNN backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_MNN "Enable Alibaba MNN backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TVM "Enable Apache TVM backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE "Enable Baidu PaddleLite backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1 "Enable Rockchip RKNN Toolkit 1.x backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2 "Enable Rockchip RKNN Toolkit 2.x backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL "Enable Huawei Ascend CL backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_SNPE "Enable Qualcomm SNPE backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_QNN "Enable Qualcomm Neural Network backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_SOPHON "Enable Sophon backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TORCH "Enable PyTorch/LibTorch backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TENSORFLOW "Enable TensorFlow backend" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_NEUROPILOT "Enable MediaTek NeuroPilot backend" OFF)

# 6. **Algorithm Plugin Options (Recommended to use default configuration, traditional CV algorithms enabled, language and text-to-image algorithms disabled by default)**
## OpenCV[Required options]
### method 1:
# nndeploy_option(ENABLE_NNDEPLOY_OPENCV "Enable OpenCV library" "path/to/opencv") 
# nndeploy_option(NNDEPLOY_OPENCV_LIBS "Link specific OpenCV library" "opencv_world480") 
### method 2:
nndeploy_option(ENABLE_NNDEPLOY_OPENCV "Enable OpenCV library" ON)
nndeploy_option(NNDEPLOY_OPENCV_LIBS "Link specific OpenCV library" "") 
## preprocessing
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS "ENABLE_NNDEPLOY_PLUGIN_PREPROCESS" ON)
## infer
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_INFER "ENABLE_NNDEPLOY_PLUGIN_INFER" ON)
## codec
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_CODEC "ENABLE_NNDEPLOY_PLUGIN_CODEC" ON)
## tokenizer
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER "ENABLE_NNDEPLOY_PLUGIN_TOKENIZER" ON)
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP "ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP" OFF)
## classification
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION "ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION" ON)
## LLM
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_LLM "ENABLE_NNDEPLOY_PLUGIN_LLM" OFF)
## detection
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_DETECT "ENABLE_NNDEPLOY_PLUGIN_DETECT" ON)
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR "ENABLE_NNDEPLOY_PLUGIN_DETECT_DETR" OFF)
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO "ENABLE_NNDEPLOY_PLUGIN_DETECT_YOLO" ON)
## track
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_TRACK "ENABLE_NNDEPLOY_PLUGIN_TRACK" ON)
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_TRACK_FAIRMOT "ENABLE_NNDEPLOY_PLUGIN_TRACK_FAIRMOT" ON)
## matting
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_MATTING "ENABLE_NNDEPLOY_PLUGIN_MATTING" ON)
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_MATTING_PPMATTING "ENABLE_NNDEPLOY_PLUGIN_MATTING_PPMATTING" ON)
## segmentation
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_SEGMENT "ENABLE_NNDEPLOY_PLUGIN_SEGMENT" ON)
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING "ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING" ON)
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RMBG "ENABLE_NNDEPLOY_PLUGIN_SEGMENT_RMBG" ON)
## text-to-image
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION "ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION" OFF)
## super_resolution
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_SUPER_RESOLUTION "ENABLE_NNDEPLOY_PLUGIN_SUPER_RESOLUTION" ON)
## ocr
nndeploy_option(ENABLE_NNDEPLOY_PLUGIN_OCR "ENABLE_NNDEPLOY_PLUGIN_OCR" ON)

# Handle dependencies between build options
## plugin
if(ENABLE_NNDEPLOY_PLUGIN)
  if(ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION)
    set(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS ON)
    set(ENABLE_NNDEPLOY_PLUGIN_INFER ON)
    set(ENABLE_NNDEPLOY_PLUGIN_CODEC ON)
  endif()

  if(ENABLE_NNDEPLOY_PLUGIN_DETECT)
    set(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS ON)
    set(ENABLE_NNDEPLOY_PLUGIN_INFER ON)
    set(ENABLE_NNDEPLOY_PLUGIN_CODEC ON)
  endif()

  if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT)
    set(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS ON)
    set(ENABLE_NNDEPLOY_PLUGIN_INFER ON)
    set(ENABLE_NNDEPLOY_PLUGIN_CODEC ON)
  endif()

  if(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION)
    set(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS ON)
    set(ENABLE_NNDEPLOY_PLUGIN_INFER ON)
    set(ENABLE_NNDEPLOY_PLUGIN_CODEC ON)
    set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER ON)
    set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP ON)
  endif()

  if(ENABLE_NNDEPLOY_PLUGIN_LLM)
    set(ENABLE_NNDEPLOY_PLUGIN_INFER ON)
    set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER ON)
    set(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP ON)
  endif()
endif()

## framework
if(ENABLE_NNDEPLOY_PYTHON)
  set(ENABLE_NNDEPLOY_CXX17_ABI ON)
  set(ENABLE_NNDEPLOY_DAG ON)
  # # Add pybind11 subdirectory
  # add_subdirectory(${ROOT_PATH}/third_party/pybind11)
  # # include
  # include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
  # add_definitions(-DENABLE_NNDEPLOY_PYTHON)
endif()

if(ENABLE_NNDEPLOY_DEMO)
  set(ENABLE_NNDEPLOY_DAG ON)
endif()

if(ENABLE_NNDEPLOY_PLUGIN)
  set(ENABLE_NNDEPLOY_DAG ON)
endif()

if(ENABLE_NNDEPLOY_TEST)
  set(ENABLE_NNDEPLOY_DAG ON)
endif()

if(ENABLE_NNDEPLOY_DAG)
  set(ENABLE_NNDEPLOY_INFERENCE ON)
  # set(ENABLE_NNDEPLOY_OP ON)
  # set(ENABLE_NNDEPLOY_IR ON)
  set(ENABLE_NNDEPLOY_DEVICE ON)
  set(ENABLE_NNDEPLOY_DEVICE_CPU ON)
  set(ENABLE_NNDEPLOY_THREAD_POOL ON)
  set(ENABLE_NNDEPLOY_BASE ON) 
endif()

if (ENABLE_NNDEPLOY_INFERENCE)
  if (ENABLE_NNDEPLOY_INFERENCE_TENSORRT)
    if(ENABLE_NNDEPLOY_DEVICE_CUDNN STREQUAL "OFF")
      set(ENABLE_NNDEPLOY_DEVICE_CUDNN ON)
    endif()
  endif()
endif()

if(ENABLE_NNDEPLOY_DEVICE_CUDNN)
  if(ENABLE_NNDEPLOY_DEVICE_CUDA STREQUAL "OFF")
    set(ENABLE_NNDEPLOY_DEVICE_CUDA ON)
  endif()
endif() 

if(ENABLE_NNDEPLOY_INFERENCE_DEFAULT)
  set(ENABLE_NNDEPLOY_NET ON)
  set(ENABLE_NNDEPLOY_OP ON)
  set(ENABLE_NNDEPLOY_IR ON)
endif()

if(ENABLE_NNDEPLOY_NET)
  set(ENABLE_NNDEPLOY_OP ON)
  set(ENABLE_NNDEPLOY_IR ON)
  set(ENABLE_NNDEPLOY_THREAD_POOL ON)
endif()

if(ENABLE_NNDEPLOY_OP)
  set(ENABLE_NNDEPLOY_IR ON)
endif()

if(ENABLE_NNDEPLOY_IR)
  set(ENABLE_NNDEPLOY_DEVICE ON)
  set(ENABLE_NNDEPLOY_DEVICE_CPU ON)
endif()

if(ENABLE_NNDEPLOY_DEVICE)
  set(ENABLE_NNDEPLOY_BASE ON)
endif()

set(ENABLE_NNDEPLOY_BASE ON)

# print option
print_summary()

# general
set(NNDEPLOY_LIB_PREFIX "lib")
set(NNDEPLOY_LIB_SUFFIX ".so")

if(CMAKE_SYSTEM_NAME MATCHES "^Android")
  set(SYSTEM_Android 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Linux")
  set(SYSTEM_Linux 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Darwin")
  set(SYSTEM_Darwin 1)
  set(NNDEPLOY_LIB_SUFFIX ".dylib")
elseif(CMAKE_SYSTEM_NAME MATCHES "^iOS")
  set(SYSTEM_iOS 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Windows")
  set(SYSTEM_Windows 1)
  set(NNDEPLOY_LIB_PREFIX "")
  set(NNDEPLOY_LIB_SUFFIX ".lib")
endif()

# compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# install path
set(NNDEPLOY_INSTALL_PATH ${ROOT_PATH}/build/nndeploy_${NNDEPLOY_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}_${CMAKE_BUILD_TYPE}_${CMAKE_CXX_COMPILER_ID})

if(ENABLE_NNDEPLOY_BUILD_SHARED)
  set(NNDEPLOY_INSTALL_TYPE LIBRARY)
else()
  set(NNDEPLOY_INSTALL_TYPE ARCHIVE)
endif()

if (SYSTEM_Windows)
  set(CMAKE_INSTALL_PREFIX ${NNDEPLOY_INSTALL_PATH})
  set(NNDEPLOY_INSTALL_LIB_PATH ${NNDEPLOY_INSTALL_PATH}/bin)
  set(NNDEPLOY_INSTALL_WIN_LIB_PATH ${NNDEPLOY_INSTALL_PATH}/lib)
else()
  set(CMAKE_INSTALL_PREFIX ${NNDEPLOY_INSTALL_PATH})
  # set(CMAKE_INSTALL_PREFIX ${ROOT_PATH}/build/nndeploy_${NNDEPLOY_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}_${CMAKE_BUILD_TYPE}_${CMAKE_CXX_COMPILER_ID}_complete)
  set(NNDEPLOY_INSTALL_LIB_PATH ${NNDEPLOY_INSTALL_PATH}/lib)
endif()
set(NNDEPLOY_INSTALL_INCLUDE_PATH ${NNDEPLOY_INSTALL_PATH}/include)
set(NNDEPLOY_INSTALL_DEMO_PATH ${NNDEPLOY_INSTALL_PATH}/demo)
set(NNDEPLOY_INSTALL_TEST_PATH ${NNDEPLOY_INSTALL_PATH}/test)
set(NNDEPLOY_INSTALL_THIRD_PARTY_PATH ${NNDEPLOY_INSTALL_PATH}/third_party)

set(EXECUTABLE_OUTPUT_PATH ${ROOT_PATH}/build)

# build type
if(ENABLE_NNDEPLOY_BUILD_SHARED)
  set(NNDEPLOY_LIB_TYPE SHARED)
else()
  set(NNDEPLOY_LIB_TYPE STATIC)
endif()

# define
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DNNDEPLOY_DEBUG)
endif()

if(${ENABLE_NNDEPLOY_DEVICE_ARM} MATCHES "OFF")
else()
  add_definitions(-DENABLE_NNDEPLOY_DEVICE_ARM)
endif()

if(${ENABLE_NNDEPLOY_DEVICE_X86} MATCHES "OFF")
else()
  add_definitions(-DENABLE_NNDEPLOY_DEVICE_X86)
endif()

if(${ENABLE_NNDEPLOY_TIME_PROFILER} MATCHES "OFF")
else()
  add_definitions(-DENABLE_NNDEPLOY_TIME_PROFILER)
endif()

if(${ENABLE_NNDEPLOY_GPU_MEM_TRACKER} MATCHES "OFF")
else()
  add_definitions(-DENABLE_NNDEPLOY_GPU_MEM_TRACKER)
endif()

if(${ENABLE_NNDEPLOY_OPENCV} MATCHES "OFF")
else()
  add_definitions(-DENABLE_NNDEPLOY_OPENCV)
endif()

set(INFERENCE_SUFFIX "")
if(ENABLE_NNDEPLOY_INFERENCE)
  if(ENABLE_NNDEPLOY_INFERENCE_DEFAULT)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_DEFAULT)
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TENSORRT)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_TENSORRT)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_trt")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TNN)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_TNN)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_tnn")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_MNN)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_MNN)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_mnn")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_OPENVINO)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_OPENVINO)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_openvino")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_COREML)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_COREML)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_coreml")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_onnxruntime")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TFLITE)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_TFLITE)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_tflite")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_NCNN)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_NCNN)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_ncnn")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_PADDLELITE)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_paddlelite")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_RKNN)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_RKNN)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_rknn")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_ASCEND_CL)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_ascend_cl")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_SNPE)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_SNPE)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_snpe")
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TVM)
    add_definitions(-DENABLE_NNDEPLOY_INFERENCE_TVM)
    set(INFERENCE_SUFFIX "${INFERENCE_SUFFIX}_tvm")
  endif()
endif()
message(STATUS "INFERENCE_SUFFIX: ${INFERENCE_SUFFIX}")


# ignore warning
if(MSVC)
    # Disable the following MSVC compiler warnings:
    # /wd4003: macro expansion operator lacks operands
    # /wd4819: file contains characters that cannot be represented in the current code page
    # /wd4244: conversion from 'type1' to 'type2', possible loss of data
    # /wd4018: signed/unsigned mismatch
    # /utf-8: set both source and execution character sets to UTF-8
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4003 /wd4819 /wd4244 /wd4018 /utf-8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003 /wd4819 /wd4244 /wd4018 /utf-8")
    # Disable the following MSVC compiler warnings:
    # /wd4251: needs to have dll-interface to be used by clients warning
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4251")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
endif()

# ignore loop-vectorize warning
if(SYSTEM_Windows)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-pass-failed")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-pass-failed")
endif()

# ignore deprecated warning
if(SYSTEM_Windows)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations -Wno-ignored-attributes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-ignored-attributes")
endif()

# BUILD_SHARED
if(SYSTEM_Windows AND ENABLE_NNDEPLOY_BUILD_SHARED)
  add_definitions(-DENABLE_NNDEPLOY_BUILDING_DLL)
endif()

if(!SYSTEM_Windows)
  # you must have return sentence for un-void function
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# for pypi
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64")
    message(STATUS "Building for x86_64 architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    message(STATUS "Building for ARM64 architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a")
    message(STATUS "Building for ARMv7 architecture")
endif()

# Platform-specific configuration
if(WIN32)
    # Windows-specific configuration
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
elseif(APPLE)
    # macOS-specific configuration
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    set(CMAKE_MACOSX_RPATH ON)
elseif(UNIX AND NOT ANDROID AND NOT IOS)
    # Linux-specific configuration (excluding Android and iOS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # Only add these options when using GCC compiler
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
    endif()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
# for pypi

if(ENABLE_NNDEPLOY_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)

  if(OPENMP_FOUND)
    add_definitions(-DENABLE_NNDEPLOY_OPENMP)

    if(MSVC)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    else()
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      include_directories(${OpenMP_C_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})

      if(${ANDROID_NDK_MAJOR})
        if(${ANDROID_NDK_MAJOR} GREATER 20)
        else()
          link_libraries(${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
        endif()
      else()
        link_libraries(${OpenMP_C_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
      endif()
    endif()
  else()
    error("OpenMP Not Found.")
  endif()
endif()

if(ENABLE_NNDEPLOY_CXX20_ABI)
  set(CMAKE_CXX_STANDARD 20)
  set(ENABLE_NNDEPLOY_CXX17_ABI OFF)
  set(ENABLE_NNDEPLOY_CXX14_ABI OFF)
  set(ENABLE_NNDEPLOY_CXX11_ABI OFF)
endif()

if(ENABLE_NNDEPLOY_CXX17_ABI)
  set(CMAKE_CXX_STANDARD 17)
  set(ENABLE_NNDEPLOY_CXX14_ABI OFF)
  set(ENABLE_NNDEPLOY_CXX11_ABI OFF)
endif()

if(ENABLE_NNDEPLOY_CXX14_ABI)
  if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
    set(CMAKE_CXX_STANDARD 17)
  else()
    set(CMAKE_CXX_STANDARD 14)
    set(ENABLE_NNDEPLOY_CXX11_ABI OFF)
  endif()
endif()

if(ENABLE_NNDEPLOY_CXX11_ABI)
  if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
    set(CMAKE_CXX_STANDARD 17)
  else()
    set(CMAKE_CXX_STANDARD 11)
  endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(ENABLE_NNDEPLOY_DEVICE_METAL OR ENABLE_NNDEPLOY_DEVICE_APPLE_NPU)
  # compile the file according to file type
  # add_compile_options(-x objective-c++)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fobjc-arc -Wno-shorten-64-to-32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -x objective-c++ -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
endif()

if(SYSTEM_Linux AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm" AND ANDROID_API_LEVAL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_C99_MATH_TR1")
  add_definitions(-D__ANDROID_API__=${ANDROID_API_LEVAL})
endif()

if(ENABLE_NNDEPLOY_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -coverage -lgcov")
  endif()
endif()

if(ENABLE_NNDEPLOY_ADDRESS_SANTIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

# Check endianness
include(TestBigEndian)
test_big_endian(NNDEPLOY_BIG_ENDIAN)
if(NNDEPLOY_BIG_ENDIAN)
  set(NNDEPLOY_CMAKE_LITTLE_ENDIAN 0)
else()
  set(NNDEPLOY_CMAKE_LITTLE_ENDIAN 1)
endif()

nndeploy_option(ENABLE_NNDEPLOY_SAFETENSORS_CPP "Enable safetensors-cpp" ON)
if(ENABLE_NNDEPLOY_SAFETENSORS_CPP)
  add_definitions(-DENABLE_NNDEPLOY_SAFETENSORS_CPP)
endif()
if(ENABLE_NNDEPLOY_IR)
  # set(ENABLE_NNDEPLOY_SAFETENSORS_CPP ON)
  # add_definitions(-DENABLE_NNDEPLOY_SAFETENSORS_CPP)
  if(ENABLE_NNDEPLOY_IR_ONNX)
    # add_definitions(-DNOMINMAX)
    # add_definitions(-DWIN32_LEAN_AND_MEAN)
    set(ENABLE_NNDEPLOY_PROTOBUF ON)
    set(ENABLE_NNDEPLOY_ONNX ON)
  endif()
endif()

if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
  # set(ENABLE_NNDEPLOY_PROTOBUF ON)
endif()

# include
include_directories(${ROOT_PATH}/framework/include)
include_directories(${ROOT_PATH}/framework/source)
include_directories(third_party)

# make
set(NNDEPLOY_FRAMEWORK_SOURCE)
set(NNDEPLOY_FRAMEWORK_OBJECT)
set(NNDEPLOY_FRAMEWORK_BINARY nndeploy_framework)
set(NNDEPLOY_FRAMEWORK_DIRECTORY nndeploy_framework)
set(NNDEPLOY_DEPEND_LIBRARY)
set(NNDEPLOY_SYSTEM_LIBRARY)
set(NNDEPLOY_THIRD_PARTY_LIBRARY)
set(NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY)
set(NNDEPLOY_TEST_THIRD_PARTY_LIBRARY)
set(NNDEPLOY_DEMO_DEPEND_LIBRARY)
set(NNDEPLOY_DEMO_SYSTEM_LIBRARY)
set(NNDEPLOY_DEMO_THIRD_PARTY_LIBRARY)

# nndeploy system library
if(SYSTEM_iOS)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  set(NNDEPLOY_SYSTEM_LIBRARY ${NNDEPLOY_SYSTEM_LIBRARY} ${ACCELERATE_FRAMEWORK})
endif()

# nndeploy third party lib
include(${ROOT_PATH}/cmake/nndeploy.cmake)

# nndeploy framework
set(FRAMEWORK_ROOT_PATH ${ROOT_PATH}/framework)

# nndeploy source
# # attention GLOB or GLOB_RECURSE
file(GLOB NNDEPLOY_FRAMEWORK_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/*.cc"
    )

if(ENABLE_NNDEPLOY_BASE)
  file(GLOB_RECURSE BASE_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/base/*.h"
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/base/*.hpp"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/base/*.cc"
  )
  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${BASE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_THREAD_POOL)
  file(GLOB_RECURSE THREAD_POOL_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/thread_pool/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/thread_pool/*.cc"
  )
  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${THREAD_POOL_SOURCE})
endif()

if(ENABLE_NNDEPLOY_CRYPTION)
  file(GLOB CRYPTION_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/cryption/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/cryption/*.cc"
  )
  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${CRYPTION_SOURCE})
endif()

if(ENABLE_NNDEPLOY_DEVICE)
  file(GLOB DEVICE_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/device/*.cc"
  )

  if(ENABLE_NNDEPLOY_DEVICE_CPU)
    file(GLOB_RECURSE DEVICE_CPU_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/cpu/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/device/cpu/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_CPU_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_X86)
    file(GLOB_RECURSE DEVICE_X86_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/x86/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/device/x86/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_X86_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_ARM)
    file(GLOB_RECURSE DEVICE_ARM_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/arm/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/device/arm/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_ARM_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_CUDA)
    file(GLOB_RECURSE DEVICE_CUDA_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/cuda/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/device/cuda/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_CUDA_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL)
    file(GLOB_RECURSE DEVICE_ASCEND_CL_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/ascend_cl/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/device/ascend_cl/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_ASCEND_CL_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_OPENCL)
    add_definitions(-DNNDEPLOY_USE_OPENCL_WRAPPER)
    file(GLOB_RECURSE DEVICE_OPENCL_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/opencl/*.h"
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/device/opencl/*.hpp"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/device/opencl/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_OPENCL_SOURCE})
  endif()

  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${DEVICE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_OP)
  file(GLOB OP_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/op/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/*.cc"
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/kernel/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/kernel/*.cc"
  )

  # if(ENABLE_NNDEPLOY_DEVICE_CPU)
  #   file(GLOB_RECURSE OP_CPU_SOURCE
  #     "${FRAMEWORK_ROOT_PATH}/include/nndeploy/op/cpu/*.h"
  #     "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/cpu/*.cc"
  #   )
  #   set(OP_SOURCE ${OP_SOURCE} ${OP_CPU_SOURCE})
  # endif()

  if(ENABLE_NNDEPLOY_DEVICE_X86 AND ENABLE_NNDEPLOY_DEVICE_X86_ONEDNN)
    file(GLOB_RECURSE OP_X86_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/op/x86/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/x86/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_X86_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_ARM AND ENABLE_NNDEPLOY_DEVICE_ARM_XNNPACK AND ENABLE_NNDEPLOY_DEVICE_ARM_QNNPACK)
    file(GLOB_RECURSE OP_ARM_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/op/arm/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/arm/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_ARM_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_CUDA AND ENABLE_NNDEPLOY_DEVICE_CUDNN)
    file(GLOB_RECURSE OP_CUDA_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/op/cuda/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/cuda/*.cc"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/*.cu"
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/kernel/cuda/*.h"
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/kernel/cuda/*.cuh"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/kernel/cuda/*.cu"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_CUDA_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL)
    file(GLOB OP_ASCEND_CL_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/op/ascend_cl/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/ascend_cl/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_ASCEND_CL_SOURCE})
    if (ENABLE_NNDEPLOY_OP_ASCEND_C)
      add_definitions(-DENABLE_NNDEPLOY_OP_ASCEND_C)
      file(GLOB_RECURSE OP_ASCEND_C_SOURCE
          "${FRAMEWORK_ROOT_PATH}/source/nndeploy/op/ascend_cl/ascend_c/*.cc"
      )
      ascendc_library(nndeploy_framework_ascend_c_kernels STATIC ${OP_ASCEND_C_SOURCE})
      set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} nndeploy_framework_ascend_c_kernels)  
    endif()
  endif()
  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${OP_SOURCE})
endif()

if(ENABLE_NNDEPLOY_IR)
  file(GLOB IR_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/ir/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/ir/*.cc"
  )

  if(ENABLE_NNDEPLOY_IR_ONNX)
    file(GLOB_RECURSE IR_ONNX_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/ir/onnx/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/ir/onnx/*.cc"
    )
    set(IR_SOURCE ${IR_SOURCE} ${IR_ONNX_SOURCE})
  endif()

  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${IR_SOURCE})
endif()

if(ENABLE_NNDEPLOY_NET)
  file(GLOB_RECURSE NET_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/net/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/net/*.cc"
  )

  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${NET_SOURCE})
endif()

if(ENABLE_NNDEPLOY_INFERENCE)
  file(GLOB INFERENCE_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/*.h"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/*.cc"
  )

  if(ENABLE_NNDEPLOY_INFERENCE_DEFAULT)
    file(GLOB_RECURSE INFERENCE_DEFAULT_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/default/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/default/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_DEFAULT_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TENSORRT)
    file(GLOB_RECURSE INFERENCE_TENSORRT_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/tensorrt/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/tensorrt/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TENSORRT_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TNN)
    file(GLOB_RECURSE INFERENCE_TNN_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/tnn/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/tnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_MNN)
    file(GLOB_RECURSE INFERENCE_MNN_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/mnn/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/mnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_MNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_OPENVINO)
    file(GLOB_RECURSE INFERENCE_OPENVINO_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/openvino/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/openvino/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_OPENVINO_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_COREML)
    file(GLOB_RECURSE INFERENCE_COREML_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/coreml/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/coreml/*.cc"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/coreml/*.mm"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_COREML_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME)
    file(GLOB_RECURSE INFERENCE_ONNXRUNTIME_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/onnxruntime/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/onnxruntime/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_ONNXRUNTIME_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TFLITE)
    file(GLOB_RECURSE INFERENCE_TFLITE_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/tflite/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/tflite/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TFLITE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_NCNN)
    file(GLOB_RECURSE INFERENCE_NCNN_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/ncnn/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/ncnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_NCNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE)
    file(GLOB_RECURSE INFERENCE_PADDLELITE_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/paddlelite/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/paddlelite/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_PADDLELITE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_RKNN)
    file(GLOB_RECURSE INFERENCE_RKNN_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/rknn/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/rknn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_RKNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL)
    file(GLOB_RECURSE INFERENCE_ASCEND_CL_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/ascend_cl/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/ascend_cl/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_ASCEND_CL_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_SNPE)
    file(GLOB_RECURSE INFERENCE_SNPE_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/snpe/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/snpe/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_SNPE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TVM)
    file(GLOB_RECURSE INFERENCE_TVM_SOURCE
      "${FRAMEWORK_ROOT_PATH}/include/nndeploy/inference/tvm/*.h"
      "${FRAMEWORK_ROOT_PATH}/source/nndeploy/inference/tvm/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TVM_SOURCE})
  endif()

  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${INFERENCE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_DAG)
  file(GLOB_RECURSE DAG_SOURCE
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/dag/*.h"
    "${FRAMEWORK_ROOT_PATH}/include/nndeploy/dag/*.hpp"
    "${FRAMEWORK_ROOT_PATH}/source/nndeploy/dag/*.cc"
  )
  set(NNDEPLOY_FRAMEWORK_SOURCE ${NNDEPLOY_FRAMEWORK_SOURCE} ${DAG_SOURCE})
endif()

# nndeploy
# # TARGET
add_library(${NNDEPLOY_FRAMEWORK_BINARY} ${NNDEPLOY_LIB_TYPE} ${NNDEPLOY_FRAMEWORK_SOURCE} ${NNDEPLOY_FRAMEWORK_OBJECT})

# # DIRECTORY
set_property(TARGET ${NNDEPLOY_FRAMEWORK_BINARY} PROPERTY FOLDER ${NNDEPLOY_FRAMEWORK_DIRECTORY})

# # DEPEND_LIBRARY
target_link_libraries(${NNDEPLOY_FRAMEWORK_BINARY} ${NNDEPLOY_DEPEND_LIBRARY})
message(STATUS "NNDEPLOY_DEPEND_LIBRARY: ${NNDEPLOY_DEPEND_LIBRARY}")

# # SYSTEM_LIBRARY
target_link_libraries(${NNDEPLOY_FRAMEWORK_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY})
message(STATUS "NNDEPLOY_SYSTEM_LIBRARY: ${NNDEPLOY_SYSTEM_LIBRARY}")

# # THIRD_PARTY_LIBRARY
target_link_libraries(${NNDEPLOY_FRAMEWORK_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY})
message(STATUS "NNDEPLOY_THIRD_PARTY_LIBRARY: ${NNDEPLOY_THIRD_PARTY_LIBRARY}")

# # install
if(SYSTEM_Windows)
  install(TARGETS ${NNDEPLOY_FRAMEWORK_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
  install(DIRECTORY ${FRAMEWORK_ROOT_PATH}/include/nndeploy DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH})
else()
  install(TARGETS ${NNDEPLOY_FRAMEWORK_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${FRAMEWORK_ROOT_PATH}/include/nndeploy DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH})
endif()

# plugin
if(ENABLE_NNDEPLOY_PLUGIN)
  include(${ROOT_PATH}/plugin/config.cmake)
endif()

# GFLAGS
set(ENABLE_NNDEPLOY_GFLAGS OFF)

if(ENABLE_NNDEPLOY_TEST)
  set(ENABLE_NNDEPLOY_GFLAGS ON)
endif()

if(ENABLE_NNDEPLOY_DEMO)
  set(ENABLE_NNDEPLOY_GFLAGS ON)
endif()

if(ENABLE_NNDEPLOY_GFLAGS)
  if(APPLE)
    find_package(gflags QUIET)
    if(gflags_FOUND)
      # message(STATUS "gflags library found:")
      # message(STATUS "    version: ${gflags_VERSION}")
      # message(STATUS "    include path: ${gflags_INCLUDE_DIRS}")
      # message(STATUS "    libraries: ${gflags_LIBRARIES}")

      include_directories(${gflags_INCLUDE_DIRS})
      set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${gflags_LIBRARIES})
    else()
      add_subdirectory(third_party/gflags)
      get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
      include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
    endif()
  else()
    add_subdirectory(third_party/gflags)
    get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
    include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
  endif()
endif()

# test
if(ENABLE_NNDEPLOY_TEST)
  if(MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
  # Disable googletest from installing its components to install directory
  # set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  # set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/googletest) # this does make the build slow. I wonder if there is a way to avoid this. 
  include(${ROOT_PATH}/test/config.cmake)
endif()

# demo
if(ENABLE_NNDEPLOY_DEMO)
  include(${ROOT_PATH}/demo/config.cmake)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tool/script/ DESTINATION ${NNDEPLOY_INSTALL_PATH} PATTERN "download" EXCLUDE PATTERN "third_party" EXCLUDE)
endif()

# python
if(ENABLE_NNDEPLOY_PYTHON)
  include(${ROOT_PATH}/python/config.cmake)
endif()

# java
if(ENABLE_NNDEPLOY_FFI_JAVA)
  include(${ROOT_PATH}/ffi/java/config.cmake)
endif()

# swift
if(ENABLE_NNDEPLOY_FFI_SWIFT)
  include(${ROOT_PATH}/ffi/swift/config.cmake)
endif()

# go
if(ENABLE_NNDEPLOY_FFI_GO)
  include(${ROOT_PATH}/ffi/go/config.cmake)
endif()

# csharp
if(ENABLE_NNDEPLOY_FFI_CSHARP)
  include(${ROOT_PATH}/ffi/csharp/config.cmake)
endif()

# cpack
include(${ROOT_PATH}/cmake/cpack.cmake)

message(STATUS "CMAKE END. NNDEPLOY_VERSION: ${NNDEPLOY_VERSION}")

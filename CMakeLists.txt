cmake_minimum_required(VERSION 3.12)

project(nndeploy)

ENABLE_LANGUAGE(ASM C CXX)

set(NNDEPLOY_MAJOR_VERSION 2)
set(NNDEPLOY_MINOR_VERSION 0)
set(NNDEPLOY_PATCH_VERSION 0)
set(NNDEPLOY_BUILD_VERSION 0)
set(NNDEPLOY_VERSION "${NNDEPLOY_MAJOR_VERSION}.${NNDEPLOY_MINOR_VERSION}.${NNDEPLOY_PATCH_VERSION}.${NNDEPLOY_BUILD_VERSION}")

include(cmake/util.cmake)
include(cmake/summary.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
endif()

# common
nndeploy_option(ENABLE_NNDEPLOY_BUILD_SHARED "ENABLE_NNDEPLOY_BUILD_SHARED" ON)
nndeploy_option(ENABLE_NNDEPLOY_SYMBOL_HIDE "ENABLE_NNDEPLOY_SYMBOL_HIDE" OFF)
nndeploy_option(ENABLE_NNDEPLOY_COVERAGE "ENABLE_NNDEPLOY_COVERAGE" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX11_ABI "ENABLE_NNDEPLOY_CXX11_ABI" ON)
nndeploy_option(ENABLE_NNDEPLOY_CXX14_ABI "ENABLE_NNDEPLOY_CXX14_ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX17_ABI "ENABLE_NNDEPLOY_CXX17_ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX20_ABI "ENABLE_NNDEPLOY_CXX20_ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_OPENMP "ENABLE_NNDEPLOY_OPENMP" OFF)
nndeploy_option(ENABLE_NNDEPLOY_ADDRESS_SANTIZER "ENABLE_NNDEPLOY_ADDRESS_SANTIZER" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DOCS "ENABLE_NNDEPLOY_DOCS" OFF)
nndeploy_option(ENABLE_NNDEPLOY_TIME_PROFILER "ENABLE_NNDEPLOY_TIME_PROFILER" ON)
nndeploy_option(ENABLE_NNDEPLOY_OPENCV "ENABLE_NNDEPLOY_OPENCV" OFF)
## base
nndeploy_option(ENABLE_NNDEPLOY_BASE "ENABLE_NNDEPLOY_BASE" ON)
## thread
nndeploy_option(ENABLE_NNDEPLOY_THREAD_POOL "ENABLE_NNDEPLOY_THREAD_POOL" ON)
## cryption
nndeploy_option(ENABLE_NNDEPLOY_CRYPTION "ENABLE_NNDEPLOY_CRYPTION" OFF)
## device
nndeploy_option(ENABLE_NNDEPLOY_DEVICE "ENABLE_NNDEPLOY_DEVICE" ON)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_CPU "ENABLE_NNDEPLOY_DEVICE_CPU" ON)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ARM "ENABLE_NNDEPLOY_DEVICE_ARM" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_X86 "ENABLE_NNDEPLOY_DEVICE_X86" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_CUDA "ENABLE_NNDEPLOY_DEVICE_CUDA" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_CUDNN "ENABLE_NNDEPLOY_DEVICE_CUDNN" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_OPENCL "ENABLE_NNDEPLOY_DEVICE_OPENCL" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_OPENGL "ENABLE_NNDEPLOY_DEVICE_OPENGL" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_METAL "ENABLE_NNDEPLOY_DEVICE_METAL" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_APPLE_NPU "ENABLE_NNDEPLOY_DEVICE_APPLE_NPU" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_HVX "ENABLE_NNDEPLOY_DEVICE_HVX" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_MTK_VPU "ENABLE_NNDEPLOY_DEVICE_MTK_VPU" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_ASCEND_CL "ENABLE_NNDEPLOY_DEVICE_ASCEND_CL" OFF)
## op
nndeploy_option(ENABLE_NNDEPLOY_OP "ENABLE_NNDEPLOY_OP" OFF)
## net
nndeploy_option(ENABLE_NNDEPLOY_NET "ENABLE_NNDEPLOY_NET" OFF)
## inference
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE "ENABLE_NNDEPLOY_INFERENCE" ON)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TENSORRT "ENABLE_NNDEPLOY_INFERENCE_TENSORRT" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_OPENVINO "ENABLE_NNDEPLOY_INFERENCE_OPENVINO" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_COREML "ENABLE_NNDEPLOY_INFERENCE_COREML" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TFLITE "ENABLE_NNDEPLOY_INFERENCE_TFLITE" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_NCNN "ENABLE_NNDEPLOY_INFERENCE_NCNN" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TNN "ENABLE_NNDEPLOY_INFERENCE_TNN" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_MNN "ENABLE_NNDEPLOY_INFERENCE_MNN" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_TVM "ENABLE_NNDEPLOY_INFERENCE_TVM" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE "ENABLE_NNDEPLOY_INFERENCE_PADDLELITE" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_1 "ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOKIT_1" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOLKIT_2 "ENABLE_NNDEPLOY_INFERENCE_RKNN_TOOKIT_2" OFF)
nndeploy_option(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL "ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL" OFF)
## dag
nndeploy_option(ENABLE_NNDEPLOY_DAG "ENABLE_NNDEPLOY_DAG" ON)
## codec
nndeploy_option(ENABLE_NNDEPLOY_CODEC "ENABLE_NNDEPLOY_CODEC" ON)
## model
nndeploy_option(ENABLE_NNDEPLOY_MODEL "ENABLE_NNDEPLOY_MODEL" ON)
nndeploy_option(ENABLE_NNDEPLOY_MODEL_TOKENIZER "ENABLE_NNDEPLOY_MODEL_TOKENIZER" OFF)
nndeploy_option(ENABLE_NNDEPLOY_MODEL_TOKENIZER_CPP "ENABLE_NNDEPLOY_MODEL_TOKENIZER_CPP" OFF)
## test
nndeploy_option(ENABLE_NNDEPLOY_TEST "ENABLE_NNDEPLOY_TEST" OFF)
## demo
nndeploy_option(ENABLE_NNDEPLOY_DEMO "ENABLE_NNDEPLOY_DEMO" OFF)

# print option
print_summary()

# set
set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# install path
set(NNDEPLOY_INSTALL_PATH ${ROOT_PATH}/build/install)
if(ENABLE_NNDEPLOY_BUILD_SHARED)
  set(NNDEPLOY_INSTALL_TYPE LIBRARY)
else()
  set(NNDEPLOY_INSTALL_TYPE ARCHIVE)
endif()
set(CMAKE_INSTALL_PREFIX ${NNDEPLOY_INSTALL_PATH})
set(NNDEPLOY_INSTALL_LIB_PATH ${NNDEPLOY_INSTALL_PATH}/lib)
set(NNDEPLOY_INSTALL_BIN_PATH ${NNDEPLOY_INSTALL_PATH}/bin)

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
if(${ENABLE_NNDEPLOY_OPENCV} MATCHES "OFF")
else()
  add_definitions(-DENABLE_NNDEPLOY_OPENCV)
endif()

# general
set(NNDEPLOY_LIB_PREFIX "lib")
set(NNDEPLOY_LIB_SUFFIX ".so")
if(CMAKE_SYSTEM_NAME MATCHES "^Android")
  set(SYSTEM.Android 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Linux")
  set(SYSTEM.Linux 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Darwin")
  set(SYSTEM.Darwin 1)
  set(NNDEPLOY_LIB_SUFFIX ".dylib")
elseif(CMAKE_SYSTEM_NAME MATCHES "^iOS")
  set(SYSTEM.iOS 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Windows")
  set(SYSTEM.Windows 1)
  set(NNDEPLOY_LIB_PREFIX "")
  set(NNDEPLOY_LIB_SUFFIX ".lib")
endif()

if(SYSTEM.Windows AND ENABLE_NNDEPLOY_BUILD_SHARED)
  add_definitions(-DENABLE_NNDEPLOY_BUILDING_DLL)
endif()

if(!SYSTEM.Windows)
  # you must have return sentence for un-void function
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

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
  if(ENABLE_NNDEPLOY_MODEL_TOKENIZER_CPP)
    set(CMAKE_CXX_STANDARD 17)
  else()
    set(CMAKE_CXX_STANDARD 14)
    set(ENABLE_NNDEPLOY_CXX11_ABI OFF)
  endif()
endif()
if(ENABLE_NNDEPLOY_CXX11_ABI)
  if(ENABLE_NNDEPLOY_MODEL_TOKENIZER_CPP)
    set(CMAKE_CXX_STANDARD 17)
  else()
    set(CMAKE_CXX_STANDARD 11)
  endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(ENABLE_NNDEPLOY_DEVICE_METAL OR ENABLE_NNDEPLOY_DEVICE_APPLE_NPU)
  # compile the file according to file type
  # add_compile_nndeploy_options(-x objective-c++)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fobjc-arc -Wno-shorten-64-to-32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -x objective-c++ -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
endif()

if(SYSTEM.Linux AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm" AND ANDROID_API_LEVAL)
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

if (ENABLE_NNDEPLOY_ADDRESS_SANTIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

# include
include_directories(include)
include_directories(source)
include_directories(third_party)

# make
set(NNDEPLOY_SOURCE)
set(NNDEPLOY_OBJECT)
set(NNDEPLOY_BINARY nndeploy)
set(NNDEPLOY_DIRECTORY nndeploy)
set(NNDEPLOY_DEPEND_LIBRARY)
set(NNDEPLOY_SYSTEM_LIBRARY)
set(NNDEPLOY_THIRD_PARTY_LIBRARY)
# nndeploy third party lib
include(${ROOT_PATH}/cmake/nndeploy.cmake)
# nndeploy source
## attention GLOB or GLOB_RECURSE
if(ENABLE_NNDEPLOY_BASE)
  file(GLOB_RECURSE BASE_SOURCE
    "${ROOT_PATH}/include/nndeploy/base/*.h"
    "${ROOT_PATH}/include/nndeploy/base/*.hpp"
    "${ROOT_PATH}/source/nndeploy/base/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${BASE_SOURCE})
endif()
if(ENABLE_NNDEPLOY_THREAD_POOL)
  file(GLOB_RECURSE THREAD_POOL_SOURCE
    "${ROOT_PATH}/include/nndeploy/thread_pool/*.h"
    "${ROOT_PATH}/source/nndeploy/thread_pool/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${THREAD_POOL_SOURCE})
endif()
if(ENABLE_NNDEPLOY_CRYPTION)
  file(GLOB CRYPTION_SOURCE
    "${ROOT_PATH}/include/nndeploy/cryption/*.h"
    "${ROOT_PATH}/source/nndeploy/cryption/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${CRYPTION_SOURCE})
endif()
if(ENABLE_NNDEPLOY_DEVICE)
  file(GLOB DEVICE_SOURCE
    "${ROOT_PATH}/include/nndeploy/device/*.h"
    "${ROOT_PATH}/source/nndeploy/device/*.cc"
  )
  if (ENABLE_NNDEPLOY_DEVICE_CPU)
    file(GLOB_RECURSE DEVICE_CPU_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/cpu/*.h"
      "${ROOT_PATH}/source/nndeploy/device/cpu/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_CPU_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_X86)
    file(GLOB_RECURSE DEVICE_X86_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/x86/*.h"
      "${ROOT_PATH}/source/nndeploy/device/x86/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_X86_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_ARM)
    file(GLOB_RECURSE DEVICE_ARM_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/arm/*.h"
      "${ROOT_PATH}/source/nndeploy/device/arm/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_ARM_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_CUDA)
    file(GLOB_RECURSE DEVICE_CUDA_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/cuda/*.h"
      "${ROOT_PATH}/source/nndeploy/device/cuda/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_CUDA_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_ASCEND_CL)
    file(GLOB_RECURSE DEVICE_ASCEND_CL_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/ascend_cl/*.h"
      "${ROOT_PATH}/source/nndeploy/device/ascend_cl/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_ASCEND_CL_SOURCE})
  endif()
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${DEVICE_SOURCE})
endif()
if(ENABLE_NNDEPLOY_OP)
  file(GLOB OP_SOURCE
    "${ROOT_PATH}/include/nndeploy/op/*.h"
    "${ROOT_PATH}/source/nndeploy/op/*.cc"
  )
  if (ENABLE_NNDEPLOY_DEVICE_CPU)
    file(GLOB_RECURSE OP_CPU_SOURCE
      "${ROOT_PATH}/include/nndeploy/op/cpu/*.h"
      "${ROOT_PATH}/source/nndeploy/op/cpu/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_CPU_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_X86)
    file(GLOB_RECURSE OP_X86_SOURCE
      "${ROOT_PATH}/include/nndeploy/op/x86/*.h"
      "${ROOT_PATH}/source/nndeploy/op/x86/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_X86_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_ARM)
    file(GLOB_RECURSE OP_ARM_SOURCE
      "${ROOT_PATH}/include/nndeploy/op/arm/*.h"
      "${ROOT_PATH}/source/nndeploy/op/arm/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_ARM_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_CUDA)
    file(GLOB_RECURSE OP_CUDA_SOURCE
      "${ROOT_PATH}/include/nndeploy/op/cuda/*.h"
      "${ROOT_PATH}/source/nndeploy/op/cuda/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_CUDA_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_ASCEND_CL)
    file(GLOB_RECURSE OP_ASCEND_CL_SOURCE
      "${ROOT_PATH}/include/nndeploy/op/ascend_cl/*.h"
      "${ROOT_PATH}/source/nndeploy/op/ascend_cl/*.cc"
    )
    set(OP_SOURCE ${OP_SOURCE} ${OP_ASCEND_CL_SOURCE})
  endif()
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${OP_SOURCE})
endif()
if(ENABLE_NNDEPLOY_NET)
  file(GLOB NET_SOURCE
    "${ROOT_PATH}/include/nndeploy/net/*.h"
    "${ROOT_PATH}/source/nndeploy/net/*.cc"
  )
  if (ENABLE_NNDEPLOY_DEVICE_CPU)
    file(GLOB_RECURSE NET_CPU_SOURCE
      "${ROOT_PATH}/include/nndeploy/net/cpu/*.h"
      "${ROOT_PATH}/source/nndeploy/net/cpu/*.cc"
    )
    set(NET_SOURCE ${NET_SOURCE} ${NET_CPU_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_X86)
    file(GLOB_RECURSE NET_X86_SOURCE
      "${ROOT_PATH}/include/nndeploy/net/x86/*.h"
      "${ROOT_PATH}/source/nndeploy/net/x86/*.cc"
    )
    set(NET_SOURCE ${NET_SOURCE} ${NET_X86_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_ARM)
    file(GLOB_RECURSE NET_ARM_SOURCE
      "${ROOT_PATH}/include/nndeploy/net/arm/*.h"
      "${ROOT_PATH}/source/nndeploy/net/arm/*.cc"
    )
    set(NET_SOURCE ${NET_SOURCE} ${NET_ARM_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_CUDA)
    file(GLOB_RECURSE NET_CUDA_SOURCE
      "${ROOT_PATH}/include/nndeploy/net/cuda/*.h"
      "${ROOT_PATH}/source/nndeploy/net/cuda/*.cc"
    )
    set(NET_SOURCE ${NET_SOURCE} ${NET_CUDA_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_ASCEND_CL)
    file(GLOB_RECURSE NET_ASCEND_CL_SOURCE
      "${ROOT_PATH}/include/nndeploy/net/ascend_cl/*.h"
      "${ROOT_PATH}/source/nndeploy/net/ascend_cl/*.cc"
    )
    set(NET_SOURCE ${NET_SOURCE} ${NET_ASCEND_CL_SOURCE})
  endif()
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${NET_SOURCE})
endif()
if(ENABLE_NNDEPLOY_INFERENCE)
  file(GLOB INFERENCE_SOURCE
    "${ROOT_PATH}/include/nndeploy/inference/*.h"
    "${ROOT_PATH}/source/nndeploy/inference/*.cc"
  )
  if (ENABLE_NNDEPLOY_INFERENCE_TENSORRT)
    file(GLOB_RECURSE INFERENCE_TENSORRT_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/tensorrt/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/tensorrt/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TENSORRT_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_TNN)
    file(GLOB_RECURSE INFERENCE_TNN_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/tnn/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/tnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TNN_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_MNN)
    file(GLOB_RECURSE INFERENCE_MNN_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/mnn/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/mnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_MNN_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_OPENVINO)
    file(GLOB_RECURSE INFERENCE_OPENVINO_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/openvino/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/openvino/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_OPENVINO_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_COREML)
    file(GLOB_RECURSE INFERENCE_COREML_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/coreml/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/coreml/*.cc"
      "${ROOT_PATH}/source/nndeploy/inference/coreml/*.mm"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_COREML_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME)
    file(GLOB_RECURSE INFERENCE_ONNXRUNTIME_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/onnxruntime/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/onnxruntime/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_ONNXRUNTIME_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_TFLITE)
    file(GLOB_RECURSE INFERENCE_TFLITE_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/tflite/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/tflite/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TFLITE_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_NCNN)
    file(GLOB_RECURSE INFERENCE_NCNN_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/ncnn/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/ncnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_NCNN_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_PADDLELITE)
    file(GLOB_RECURSE INFERENCE_PADDLELITE_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/paddlelite/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/paddlelite/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_PADDLELITE_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_RKNN)
    file(GLOB_RECURSE INFERENCE_RKNN_SOURCE
            "${ROOT_PATH}/include/nndeploy/inference/rknn/*.h"
            "${ROOT_PATH}/source/nndeploy/inference/rknn/*.cc"
            )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_RKNN_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL)
    file(GLOB_RECURSE INFERENCE_ASCEND_CL_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/ascend_cl/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/ascend_cl/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_ASCEND_CL_SOURCE})
  endif()
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${INFERENCE_SOURCE})
endif()
if(ENABLE_NNDEPLOY_DAG)
  file(GLOB_RECURSE DAG_SOURCE
    "${ROOT_PATH}/include/nndeploy/dag/*.h"
    "${ROOT_PATH}/include/nndeploy/dag/*.hpp"
    "${ROOT_PATH}/source/nndeploy/dag/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${DAG_SOURCE})
endif()
if(ENABLE_NNDEPLOY_CODEC)
  file(GLOB CODEC_SOURCE
    "${ROOT_PATH}/include/nndeploy/codec/*.h"
    "${ROOT_PATH}/source/nndeploy/codec/*.cc"
  )
  if (ENABLE_NNDEPLOY_OPENCV)
    file(GLOB_RECURSE CODEC_OPENCV_SOURCE
      "${ROOT_PATH}/include/nndeploy/codec/opencv/*.h"
      "${ROOT_PATH}/source/nndeploy/codec/opencv/*.cc"
    )
    set(CODEC_SOURCE ${CODEC_SOURCE} ${CODEC_OPENCV_SOURCE})
  endif()
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${CODEC_SOURCE})
endif()
if(ENABLE_NNDEPLOY_MODEL)
  set(MODEL_SOURCE)
  include(${ROOT_PATH}/source/nndeploy/model/config.cmake)
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${MODEL_SOURCE})
  # message(STATUS "MODEL_SOURCE: ${MODEL_SOURCE}")
endif()
# nndeploy
## TARGET
add_library(${NNDEPLOY_BINARY} ${NNDEPLOY_LIB_TYPE} ${NNDEPLOY_SOURCE} ${NNDEPLOY_OBJECT})
## DIRECTORY
set_property(TARGET ${NNDEPLOY_BINARY} PROPERTY FOLDER ${NNDEPLOY_DIRECTORY})
## DEPEND_LIBRARY
target_link_libraries(${NNDEPLOY_BINARY} ${NNDEPLOY_DEPEND_LIBRARY}) 
## SYSTEM_LIBRARY
target_link_libraries(${NNDEPLOY_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY}) 
## THIRD_PARTY_LIBRARY
target_link_libraries(${NNDEPLOY_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY}) 
## install
if(SYSTEM.Windows)
  install(TARGETS ${NNDEPLOY_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
else() 
  install(TARGETS ${NNDEPLOY_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
endif()
 
# GFLAGS
set(ENABLE_NNDEPLOY_GFLAGS)
if(ENABLE_NNDEPLOY_TEST)
  set(ENABLE_NNDEPLOY_GFLAGS ON)
endif()
if (ENABLE_NNDEPLOY_DEMO)
  set(ENABLE_NNDEPLOY_GFLAGS ON)
endif()
if(ENABLE_NNDEPLOY_GFLAGS)
  add_subdirectory(third_party/gflags)
  get_target_property(GFLAGS_INCLUDE_DIRS gflags INTERFACE_INCLUDE_DIRECTORIES)
  include_directories(BEFORE "${GFLAGS_INCLUDE_DIRS}")
endif()

# nndeploy_test
if(ENABLE_NNDEPLOY_TEST)
  include(${ROOT_PATH}/test/config.cmake)
endif()

# nndeploy_demo
if (ENABLE_NNDEPLOY_DEMO)
  include(${ROOT_PATH}/demo/config.cmake)
endif()

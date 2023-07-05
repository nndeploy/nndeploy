cmake_minimum_required(VERSION 3.12)

project(nndeploy)

ENABLE_LANGUAGE(ASM C CXX)

set(NNDEPLOY_MAJOR_VERSION 1)
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
nndeploy_option(ENABLE_NNDEPLOY_BUILD_SHARED "ENABLE_NNDEPLOY_BUILD_SHARED" OFF)
nndeploy_option(ENABLE_NNDEPLOY_SYMBOL_HIDE "ENABLE_NNDEPLOY_SYMBOL_HIDE" OFF)
nndeploy_option(ENABLE_NNDEPLOY_COVERAGE "ENABLE_NNDEPLOY_COVERAGE" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX11_ABI "ENABLE_NNDEPLOY_CXX11_ABI" ON)
nndeploy_option(ENABLE_NNDEPLOY_CXX14_ABI "ENABLE_NNDEPLOY_CXX14_ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_CXX17_ABI "ENABLE_NNDEPLOY_CXX17_ABI" OFF)
nndeploy_option(ENABLE_NNDEPLOY_OPENMP "ENABLE_NNDEPLOY_OPENMP" OFF)
nndeploy_option(ENABLE_NNDEPLOY_VALGRIND "ENABLE_NNDEPLOY_VALGRIND" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DOCS "ENABLE_NNDEPLOY_DOCS" OFF)
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
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_OPENCL "ENABLE_NNDEPLOY_DEVICE_OPENCL" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_OPENGL "ENABLE_NNDEPLOY_DEVICE_OPENGL" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_METAL "ENABLE_NNDEPLOY_DEVICE_METAL" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_APPLE_NPU "ENABLE_NNDEPLOY_DEVICE_APPLE_NPU" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_HVX "ENABLE_NNDEPLOY_DEVICE_HVX" OFF)
nndeploy_option(ENABLE_NNDEPLOY_DEVICE_MTK_VPU "ENABLE_NNDEPLOY_DEVICE_MTK_VPU" OFF)
## op
nndeploy_option(ENABLE_NNDEPLOY_OP "ENABLE_NNDEPLOY_OP" OFF)
nndeploy_option(ENABLE_NNDEPLOY_OP_NN "ENABLE_NNDEPLOY_OP_NN" OFF)
nndeploy_option(ENABLE_NNDEPLOY_OP_CV "ENABLE_NNDEPLOY_OP_CV" OFF)
nndeploy_option(ENABLE_NNDEPLOY_OP_AUDIO "ENABLE_NNDEPLOY_OP_AUDIO" OFF)
## forward
nndeploy_option(ENABLE_NNDEPLOY_FORWARD "ENABLE_NNDEPLOY_FORWARD" OFF)
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
nndeploy_option(ENABLE_NNDEPLOY_AICOMPILER_TVM "ENABLE_NNDEPLOY_INFERENCE_TVM" OFF)
## pipeline
nndeploy_option(ENABLE_NNDEPLOY_PIPELINE "ENABLE_NNDEPLOY_PIPELINE" ON)
## model
nndeploy_option(ENABLE_NNDEPLOY_MODEL "ENABLE_NNDEPLOY_MODEL" ON)
nndeploy_option(ENABLE_NNDEPLOY_MODEL_DETECT "ENABLE_NNDEPLOY_MODEL_DETECT" OFF)
nndeploy_option(ENABLE_NNDEPLOY_MODEL_DETECT_YOLOV5 "ENABLE_NNDEPLOY_MODEL_DETECT_YOLOV5" OFF)
nndeploy_option(ENABLE_NNDEPLOY_MODEL_DETECT_DETR "ENABLE_NNDEPLOY_MODEL_DETECT_DETR" OFF)
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

# build type
if(ENABLE_NNDEPLOY_BUILD_SHARED)
  set(NNDEPLOY_LIB_TYPE SHARED)
  set(NNDEPLOY_LIB_TYPE SHARED)
else()
  set(NNDEPLOY_LIB_TYPE STATIC)
endif()

# define
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DNNDEPLOY_DEBUG)
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
elseif(CMAKE_SYSTEM_NAME MATCHES "^iOS")
  set(SYSTEM.iOS 1)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Windows")
  set(SYSTEM.Windows 1)
  set(NNDEPLOY_LIB_SUFFIX ".dll")
endif()
set(NNDEPLOY_LIB_PATH)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
  set(PROCESSOR.arm 1)
  set(NNDEPLOY_LIB_PATH "lib/armeabi-v7a/Release")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
  set(PROCESSOR.aarch64 1)
  set(NNDEPLOY_LIB_PATH "lib/arm64-v8a/Release")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86")
  set(PROCESSOR.x86 1)
  set(NNDEPLOY_LIB_PATH "lib/x86/Release")
endif()

if(SYSTEM.Windows AND ENABLE_NNDEPLOY_BUILD_SHARED)
  add_definitions(-DENABLE_NNDEPLOY_BUILDING_DLL)
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

if(ENABLE_NNDEPLOY_CXX17_ABI)
  set(CMAKE_CXX_STANDARD 17)
  set(ENABLE_NNDEPLOY_CXX14_ABI OFF)
  set(ENABLE_NNDEPLOY_CXX11_ABI OFF)
endif()
if(ENABLE_NNDEPLOY_CXX14_ABI)
  set(CMAKE_CXX_STANDARD 14)
  set(ENABLE_NNDEPLOY_CXX11_ABI OFF)
endif()
if(ENABLE_NNDEPLOY_CXX11_ABI)
  set(CMAKE_CXX_STANDARD 11)
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

# include
include_directories(include)
include_directories(source)

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
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${DEVICE_SOURCE})
endif()
if(ENABLE_NNDEPLOY_OP)
  file(GLOB OP_SOURCE
    "${ROOT_PATH}/include/nndeploy/op/*.h"
    "${ROOT_PATH}/source/nndeploy/op/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${OP_SOURCE})
endif()
if(ENABLE_NNDEPLOY_FORWARD)
  file(GLOB FORWARD_SOURCE
    "${ROOT_PATH}/include/nndeploy/forward/*.h"
    "${ROOT_PATH}/source/nndeploy/forward/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${FORWARD_SOURCE})
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
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${INFERENCE_SOURCE})
endif()
if(ENABLE_NNDEPLOY_PIPELINE)
  file(GLOB PIPELINE_SOURCE
    "${ROOT_PATH}/include/nndeploy/pipeline/*.h"
    "${ROOT_PATH}/source/nndeploy/pipeline/*.cc"
  )
  file(GLOB PIPELINE_PREPROCESS_SOURCE
    "${ROOT_PATH}/include/nndeploy/pipeline/preprocess/*.h"
    "${ROOT_PATH}/source/nndeploy/pipeline/preprocess/*.cc"
  )
  if(ENABLE_NNDEPLOY_OPENCV)
    file(GLOB_RECURSE PIPELINE_PREPROCESS_OPENCV_SOURCE
      "${ROOT_PATH}/include/nndeploy/pipeline/preprocess/opencv/*.h"
      "${ROOT_PATH}/source/nndeploy/pipeline/preprocess/opencv/*.cc"
    )
    set(PIPELINE_PREPROCESS_SOURCE ${PIPELINE_PREPROCESS_SOURCE} ${PIPELINE_PREPROCESS_OPENCV_SOURCE})
  endif()
  set(PIPELINE_SOURCE ${PIPELINE_SOURCE} ${PIPELINE_PREPROCESS_SOURCE})
  file(GLOB PIPELINE_POSTPROCESS_SOURCE
    "${ROOT_PATH}/include/nndeploy/pipeline/postprocess/*.h"
    "${ROOT_PATH}/source/nndeploy/pipeline/postprocess/*.cc"
  )
  if(ENABLE_NNDEPLOY_OPENCV)
    file(GLOB_RECURSE PIPELINE_POSTPROCESS_OPENCV_SOURCE
      "${ROOT_PATH}/include/nndeploy/pipeline/postprocess/opencv/*.h"
      "${ROOT_PATH}/source/nndeploy/pipeline/postprocess/opencv/*.cc"
    )
    set(PIPELINE_POSTPROCESS_SOURCE ${PIPELINE_POSTPROCESS_SOURCE} ${PIPELINE_POSTPROCESS_OPENCV_SOURCE})
  endif()
  set(PIPELINE_SOURCE ${PIPELINE_SOURCE} ${PIPELINE_POSTPROCESS_SOURCE})
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${PIPELINE_SOURCE})
  message(STATUS "PIPELINE_SOURCE: ${PIPELINE_SOURCE}")
endif()
if(ENABLE_NNDEPLOY_MODEL)
  set(MODEL_SOURCE)
  include(${ROOT_PATH}/source/nndeploy/model/config.cmake)
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${MODEL_SOURCE})
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
install(TARGETS ${NNDEPLOY_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})

# nndeploy_test
if(ENABLE_NNDEPLOY_TEST)
  add_subdirectory(${ROOT_PATH}/test)
endif()

# nndeploy_demo
if (ENABLE_NNDEPLOY_DEMO)
  add_subdirectory(${ROOT_PATH}/demo/cc)
endif()


cmake_minimum_required(VERSION 3.0)

project(nnkit)

ENABLE_LANGUAGE(ASM)

set(NNKIT_MAJOR_VERSION 1)
set(NNKIT_MINOR_VERSION 0)
set(NNKIT_PATCH_VERSION 0)
set(NNKIT_BUILD_VERSION 0)
set(NNKIT_VERSION "${NNKIT_MAJOR_VERSION}.${NNKIT_MINOR_VERSION}.${NNKIT_PATCH_VERSION}.${NNKIT_BUILD_VERSION}")

# common
option(NNKIT_ENABLE_BUILD_SHARED "" ON)
option(NNKIT_ENABLE_SYMBOL_HIDE "" ON)
option(NNKIT_ENABLE_COVERAGE "" ON)
option(NNKIT_ENABLE_CXX11_ABI "" ON)
option(NNKIT_ENABLE_CXX14_ABI "" OFF)
option(NNKIT_ENABLE_OPENMP "" ON)
option(NNKIT_ENABLE_DEBUG "" OFF)
# device
option(NNKIT_ENABLE_DEVICE_CPU "" ON)
option(NNKIT_ENABLE_DEVICE_ARM "" OFF)
option(NNKIT_ENABLE_DEVICE_X86 "" OFF)
option(NNKIT_ENABLE_DEVICE_CUDA "" OFF)
option(NNKIT_ENABLE_DEVICE_OPENCL "" OFF)
option(NNKIT_ENABLE_DEVICE_METAL "" OFF)
option(NNKIT_ENABLE_DEVICE_APPLE_NPU "" OFF)
# inference
option(NNKIT_ENABLE_INFERENCE_TENSORRT "" OFF)
option(NNKIT_ENABLE_INFERENCE_OPENVINO "" OFF)
option(NNKIT_ENABLE_INFERENCE_COREML "" OFF)
option(NNKIT_ENABLE_INFERENCE_TFLITE  OFF)
option(NNKIT_ENABLE_INFERENCE_TVM  "" OFF)
option(NNKIT_ENABLE_INFERENCE_TNN "" OFF)
# audio
option(NNKIT_ENABLE_AUDIO "" ON)
# vision
option(NNKIT_ENABLE_VISION "" ON)
# nntask
option(NNKIT_ENABLE_NNTASK "" ON)
# test
option(NNKIT_ENABLE_UNITTEST "" OFF)

# set
set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# define
if (NNKIT_ENABLE_DEBUG) 
  add_definitions(-DNNKIT_ENABLE_DEBUG)
endif()

# general 
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
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
  set(PROCESSOR.arm 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
  set(PROCESSOR.aarch64 1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86")
  set(PROCESSOR.x86 1)
endif()

if (SYSTEM.Windows AND NNKIT_ENABLE_BUILD_SHARED)
  add_definitions(-DNNKIT_ENABLE_BUILDING_DLL)
endif()

if(NNKIT_ENABLE_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  if(OPENMP_FOUND)
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

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if (NNKIT_ENABLE_CXX14_ABI)
  set(CMAKE_CXX_STANDARD 14)
  set(NNKIT_ENABLE_CXX11_ABI OFF)
endif()
if (NNKIT_ENABLE_CXX11_ABI)
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NNKIT_ENABLE_DEVICE_METAL OR NNKIT_ENABLE_DEVICE_APPLE_NPU)
  # compile the file according to file type
  # add_compile_options(-x objective-c++)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fobjc-arc -Wno-shorten-64-to-32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -x objective-c++ -fobjc-arc -Wno-shorten-64-to-32 -Wno-null-character")
endif()

if(SYSTEM.Linux AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm" AND ANDROID_API_LEVAL)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_C99_MATH_TR1")
  add_definitions(-D__ANDROID_API__=${ANDROID_API_LEVAL})
endif()

if(NNKIT_ENABLE_COVERAGE)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -coverage -lgcov")
  endif()
endif()

# nndeploy
add_subdirectory(nndeploy)

# nntask
if(NNKIT_ENABLE_NNTASK)
  add_subdirectory(nntask)
endif()

# test
if(NNKIT_ENABLE_UNITTEST)
  add_subdirectory(test)
endif()

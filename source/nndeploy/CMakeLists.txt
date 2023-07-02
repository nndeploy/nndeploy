# 注意 有些地方使用了 GLOB， 有些地方使用了 GLOB_RECURSE
if(ENABLE_NNDEPLOY_BASE)
  file(GLOB_RECURSE BASE_SOURCE
    "${ROOT_PATH}/include/nndeploy/base/*.h"
    "${ROOT_PATH}/source/nndeploy/base/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${BASE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_THREAD)
  file(GLOB_RECURSE THREAD_SOURCE
    "${ROOT_PATH}/include/nndeploy/thread/*.h"
    "${ROOT_PATH}/source/nndeploy/thread/*.cc"
  )
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${THREAD_SOURCE})
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
    file(GLOB DEVICE_CPU_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/cpu/*.h"
      "${ROOT_PATH}/source/nndeploy/device/cpu/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_CPU_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_X86)
    file(GLOB DEVICE_X86_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/x86/*.h"
      "${ROOT_PATH}/source/nndeploy/device/x86/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_X86_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_ARM)
    file(GLOB DEVICE_ARM_SOURCE
      "${ROOT_PATH}/include/nndeploy/device/arm/*.h"
      "${ROOT_PATH}/source/nndeploy/device/arm/*.cc"
    )
    set(DEVICE_SOURCE ${DEVICE_SOURCE} ${DEVICE_ARM_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_DEVICE_CUDA)
    file(GLOB DEVICE_CUDA_SOURCE
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
    file(GLOB INFERENCE_TENSORRT_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/tensorrt/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/tensorrt/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TENSORRT_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_TNN)
    file(GLOB INFERENCE_TNN_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/tnn/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/tnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_TNN_SOURCE})
  endif()
  if (ENABLE_NNDEPLOY_INFERENCE_MNN)
    file(GLOB INFERENCE_MNN_SOURCE
      "${ROOT_PATH}/include/nndeploy/inference/mnn/*.h"
      "${ROOT_PATH}/source/nndeploy/inference/mnn/*.cc"
    )
    set(INFERENCE_SOURCE ${INFERENCE_SOURCE} ${INFERENCE_MNN_SOURCE})
  endif()
  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${INFERENCE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_TASK)
  file(GLOB TASK_SOURCE
    "${ROOT_PATH}/include/nndeploy/task/*.h"
    "${ROOT_PATH}/source/nndeploy/task/*.cc"
  )

  file(GLOB TASK_PRE_PROCESS_SOURCE
    "${ROOT_PATH}/include/nndeploy/task/pre_process/*.h"
    "${ROOT_PATH}/source/nndeploy/task/pre_process/*.cc"
  )
  if(ENABLE_NNDEPLOY_OPENCV)
    file(GLOB_RECURSE TASK_PRE_PROCESS_OPENCV_SOURCE
      "${ROOT_PATH}/include/nndeploy/task/pre_process/opencv/*.h"
      "${ROOT_PATH}/source/nndeploy/task/pre_process/opencv/*.cc"
    )
    set(TASK_PRE_PROCESS_SOURCE ${TASK_PRE_PROCESS_SOURCE} ${TASK_PRE_PROCESS_OPENCV_SOURCE})
  endif()
  set(TASK_SOURCE ${TASK_SOURCE} ${TASK_PRE_PROCESS_SOURCE})

  file(GLOB_RECURSE TASK_TEMPLATE_INFERENCE_SOURCE
    "${ROOT_PATH}/include/nndeploy/task/template_inference/*.h"
    "${ROOT_PATH}/source/nndeploy/task/template_inference/*.cc"
  )
  set(TASK_SOURCE ${TASK_SOURCE} ${TASK_TEMPLATE_INFERENCE_SOURCE})

  file(GLOB TASK_POST_PROCESS_SOURCE
    "${ROOT_PATH}/include/nndeploy/task/post_process/*.h"
    "${ROOT_PATH}/source/nndeploy/task/post_process/*.cc"
  )
  if(ENABLE_NNDEPLOY_OPENCV)
    file(GLOB_RECURSE TASKPOST_PROCESS_OPENCV_SOURCE
      "${ROOT_PATH}/include/nndeploy/task/post_process/opencv/*.h"
      "${ROOT_PATH}/source/nndeploy/task/post_process/opencv/*.cc"
    )
    set(TASK_POST_PROCESS_SOURCE ${TASK_POST_PROCESS_SOURCE} ${TASK_POST_PROCESS_OPENCV_SOURCE})
  endif()
  set(TASK_SOURCE ${TASK_SOURCE} ${TASK_POST_PROCESS_SOURCE})

  set(NNDEPLOY_SOURCE ${NNDEPLOY_SOURCE} ${TASK_SOURCE})
endif()

# nndeploy third party lib
include(${ROOT_PATH}/cmake/nndeploy.cmake)

# TARGET
add_library(${NNDEPLOY_BINARY} ${NNDEPLOY_LIB_TYPE} ${NNDEPLOY_SOURCE} ${NNDEPLOY_OBJECT})
# DIRECTORY
set_property(TARGET ${NNDEPLOY_BINARY} PROPERTY FOLDER ${NNDEPLOY_DIRECTORY})
# DEPEND_LIBRARY
target_link_libraries(${NNDEPLOY_BINARY} ${NNDEPLOY_DEPEND_LIBRARY}) 
# SYSTEM_LIBRARY
target_link_libraries(${NNDEPLOY_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY}) 
# THIRD_PARTY_LIBRARY
target_link_libraries(${NNDEPLOY_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY}) 
# install
install(TARGETS ${NNDEPLOY_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})

# nndeploy task 
## dedect
nndeploy_option(ENABLE_NNDEPLOY_TASK_DEDECT "ENABLE_NNDEPLOY_TASK_DEDECT" OFF)
nndeploy_option(ENABLE_NNDEPLOY_TASK_DEDECT_YOLOV5 "ENABLE_NNDEPLOY_TASK_DEDECT_YOLOV5" OFF)
nndeploy_option(ENABLE_NNDEPLOY_TASK_DEDECT_DETR "ENABLE_NNDEPLOY_TASK_DEDECT_DETR" OFF)
if(ENABLE_NNDEPLOY_TASK_DEDECT)
  add_subdirectory(${ROOT_PATH}/source/nndeploy/task/dedect)
endif()

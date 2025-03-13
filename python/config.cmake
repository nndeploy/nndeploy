message(STATUS "python")

# set
set(SOURCE)
set(OBJECT)
set(BINARY pynndeploy)
set(DIRECTORY python)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

# 添加版本信息定义
set(PACKAGE_VERSION ${NNDEPLOY_VERSION})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES)

# 添加 pybind11 子目录
add_subdirectory(${ROOT_PATH}/third_party/pybind11)

# include
include_directories(${ROOT_PATH}/python/src)
include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

# TODO:SOURCE 
# nndeploy source
file(GLOB PYTHON_SOURCE
  "${ROOT_PATH}/python/src/*.h"
  "${ROOT_PATH}/python/src/*.cc"
)
set(SOURCE ${SOURCE} ${PYTHON_SOURCE})

# framework
if(ENABLE_NNDEPLOY_BASE)
  file(GLOB_RECURSE PYTHON_BASE_SOURCE
    "${ROOT_PATH}/python/src/base/*.h"
    "${ROOT_PATH}/python/src/base/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_BASE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_THREAD_POOL)
  file(GLOB_RECURSE PYTHON_THREAD_POOL_SOURCE
    "${ROOT_PATH}/python/src/thread_pool/*.h"
    "${ROOT_PATH}/python/src/thread_pool/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_THREAD_POOL_SOURCE})
endif()

if(ENABLE_NNDEPLOY_CRYPTION)
  file(GLOB_RECURSE PYTHON_CRYPTION_SOURCE
    "${ROOT_PATH}/python/src/cryption/*.h"
    "${ROOT_PATH}/python/src/cryption/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_CRYPTION_SOURCE})
endif()

if(ENABLE_NNDEPLOY_DEVICE)
  file(GLOB_RECURSE PYTHON_DEVICE_SOURCE
    "${ROOT_PATH}/python/src/device/*.h"
    "${ROOT_PATH}/python/src/device/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_DEVICE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_IR)
  file(GLOB_RECURSE PYTHON_IR_SOURCE
    "${ROOT_PATH}/python/src/ir/*.h"
    "${ROOT_PATH}/python/src/ir/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_IR_SOURCE})
endif()

if(ENABLE_NNDEPLOY_OP)
  file(GLOB_RECURSE PYTHON_OP_SOURCE
    "${ROOT_PATH}/python/src/op/*.h"
    "${ROOT_PATH}/python/src/op/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_OP_SOURCE})
endif()

# if(ENABLE_NNDEPLOY_NET)
#   file(GLOB_RECURSE PYTHON_NET_SOURCE
#     "${ROOT_PATH}/python/src/net/*.h"
#     "${ROOT_PATH}/python/src/net/*.cc"
#   )
#   set(SOURCE ${SOURCE} ${PYTHON_NET_SOURCE})
# endif()

if(ENABLE_NNDEPLOY_INFERENCE)
  file(GLOB_RECURSE PYTHON_INFERENCE_SOURCE
    "${ROOT_PATH}/python/src/inference/*.h"
    "${ROOT_PATH}/python/src/inference/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_INFERENCE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_DAG)
  file(GLOB_RECURSE PYTHON_DAG_SOURCE
    "${ROOT_PATH}/python/src/dag/*.h"
    "${ROOT_PATH}/python/src/dag/*.cc"
  )
  list(REMOVE_ITEM PYTHON_DAG_SOURCE 
    "${ROOT_PATH}/python/src/dag/node.cc"
    "${ROOT_PATH}/python/src/dag/graph.cc"
  )
  set(PYTHON_DAG_SOURCE ${PYTHON_DAG_SOURCE}
    # 依赖于node.cc的文件后列出
    "${ROOT_PATH}/python/src/dag/node.cc"
    "${ROOT_PATH}/python/src/dag/graph.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_DAG_SOURCE})
  message(STATUS "PYTHON_DAG_SOURCE: ${PYTHON_DAG_SOURCE}")
endif()

# plugin
if(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS)
  file(GLOB_RECURSE PYTHON_PREPROCESS_SOURCE
    "${ROOT_PATH}/python/src/preprocess/*.h"
    "${ROOT_PATH}/python/src/preprocess/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_PREPROCESS_SOURCE})
endif()

if(ENABLE_NNDEPLOY_PLUGIN_CODEC)
  file(GLOB_RECURSE PYTHON_CODEC_SOURCE
    "${ROOT_PATH}/python/src/codec/*.h"
    "${ROOT_PATH}/python/src/codec/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_CODEC_SOURCE})
  message(STATUS "PYTHON_CODEC_SOURCE: ${PYTHON_CODEC_SOURCE}")
endif()

if(ENABLE_NNDEPLOY_PLUGIN_INFER)
  file(GLOB_RECURSE PYTHON_INFER_SOURCE
    "${ROOT_PATH}/python/src/infer/*.h"
    "${ROOT_PATH}/python/src/infer/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_INFER_SOURCE})
endif()

if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER)
  file(GLOB_RECURSE PYTHON_TOKENIZER_SOURCE
    "${ROOT_PATH}/python/src/tokenizer/*.h"
    "${ROOT_PATH}/python/src/tokenizer/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_TOKENIZER_SOURCE})
endif()

if(ENABLE_NNDEPLOY_PLUGIN_DETECT)
  file(GLOB_RECURSE PYTHON_DETECT_SOURCE
    "${ROOT_PATH}/python/src/detect/*.h"
    "${ROOT_PATH}/python/src/detect/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_DETECT_SOURCE})
endif()

if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT)
  file(GLOB_RECURSE PYTHON_SEGMENT_SOURCE
    "${ROOT_PATH}/python/src/segment/*.h"
    "${ROOT_PATH}/python/src/segment/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_SEGMENT_SOURCE})
endif()

if(ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION)
  file(GLOB_RECURSE PYTHON_CLASSIFICATION_SOURCE
    "${ROOT_PATH}/python/src/classification/*.h"
    "${ROOT_PATH}/python/src/classification/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_CLASSIFICATION_SOURCE})
endif()

if(ENABLE_NNDEPLOY_PLUGIN_LLM)
  file(GLOB_RECURSE PYTHON_LLM_SOURCE
    "${ROOT_PATH}/python/src/llm/*.h"
    "${ROOT_PATH}/python/src/llm/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_LLM_SOURCE})
endif()

if(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION)
  file(GLOB_RECURSE PYTHON_STABLE_DIFFUSION_SOURCE
    "${ROOT_PATH}/python/src/stable_diffusion/*.h"
    "${ROOT_PATH}/python/src/stable_diffusion/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_STABLE_DIFFUSION_SOURCE})
endif()

# 创建 Python 模块
pybind11_add_module(${BINARY} ${SOURCE})

# 属性
set_target_properties(${BINARY} PROPERTIES OUTPUT_NAME "_nndeploy_internal")
set_target_properties(${BINARY} PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nndeploy")
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# link
# target_link_libraries(${BINARY} PUBLIC nndeploy_framework)
# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_FRAMEWORK_BINARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_DEPEND_LIBRARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_PYTHON_DEPEND_LIBRARY})
target_link_libraries(${BINARY} PUBLIC  ${DEPEND_LIBRARY})

# SYSTEM_LIBRARY
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_SYSTEM_LIBRARY})
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_PYTHON_SYSTEM_LIBRARY})
target_link_libraries(${BINARY} PUBLIC  ${SYSTEM_LIBRARY})

# THIRD_PARTY_LIBRARY
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PYTHON_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_LIST})
target_link_libraries(${BINARY} PUBLIC ${THIRD_PARTY_LIBRARY})

# install
#if(SYSTEM.Windows)
#  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_BIN_PATH})
#else()
#  install(TARGETS ${BINARY} RUNTIME DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
#endif()

# unkown
if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    add_custom_command(TARGET ${BINARY} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION} 
        ${PROJECT_SOURCE_DIR}/python/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION})
        message(STATUS "Copying ${CMAKE_CURRENT_BINARY_DIR}/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION} to ${PROJECT_SOURCE_DIR}/python/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}")
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

# 生成python pip package安装脚本
configure_file(${ROOT_PATH}/python/setup.py.cfg ${PROJECT_SOURCE_DIR}/python/setup.py)

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)
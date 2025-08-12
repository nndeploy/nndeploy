message(STATUS "Building nndeploy python")

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
# add_subdirectory(${ROOT_PATH}/third_party/pybind11)

# include
include_directories(${ROOT_PATH}/python/src)
# include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

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

if(ENABLE_NNDEPLOY_NET)
  file(GLOB_RECURSE PYTHON_NET_SOURCE
    "${ROOT_PATH}/python/src/net/*.h"
    "${ROOT_PATH}/python/src/net/*.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_NET_SOURCE})
endif()

if(ENABLE_NNDEPLOY_INFERENCE)
  file(GLOB PYTHON_INFERENCE_SOURCE
    "${ROOT_PATH}/python/src/inference/*.h"
    "${ROOT_PATH}/python/src/inference/*.cc"
  )
  
  if(ENABLE_NNDEPLOY_INFERENCE_DEFAULT)
    file(GLOB_RECURSE INFERENCE_DEFAULT_SOURCE
      "${ROOT_PATH}/python/src/inference/default/*.h"
      "${ROOT_PATH}/python/src/inference/default/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_DEFAULT_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TENSORRT)
    file(GLOB_RECURSE INFERENCE_TENSORRT_SOURCE
      "${ROOT_PATH}/python/src/inference/tensorrt/*.h"
      "${ROOT_PATH}/python/src/inference/tensorrt/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_TENSORRT_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TNN)
    file(GLOB_RECURSE INFERENCE_TNN_SOURCE
      "${ROOT_PATH}/python/src/inference/tnn/*.h"
      "${ROOT_PATH}/python/src/inference/tnn/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_TNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_MNN)
    file(GLOB_RECURSE INFERENCE_MNN_SOURCE
      "${ROOT_PATH}/python/src/inference/mnn/*.h"
      "${ROOT_PATH}/python/src/inference/mnn/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_MNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_OPENVINO)
    file(GLOB_RECURSE INFERENCE_OPENVINO_SOURCE
      "${ROOT_PATH}/python/src/inference/openvino/*.h"
      "${ROOT_PATH}/python/src/inference/openvino/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_OPENVINO_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_COREML)
    file(GLOB_RECURSE INFERENCE_COREML_SOURCE
      "${ROOT_PATH}/python/src/inference/coreml/*.h"
      "${ROOT_PATH}/python/src/inference/coreml/*.cc"
      "${ROOT_PATH}/python/src/inference/coreml/*.mm"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_COREML_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME)
    file(GLOB_RECURSE INFERENCE_ONNXRUNTIME_SOURCE
      "${ROOT_PATH}/python/src/inference/onnxruntime/*.h"
      "${ROOT_PATH}/python/src/inference/onnxruntime/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_ONNXRUNTIME_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TFLITE)
    file(GLOB_RECURSE INFERENCE_TFLITE_SOURCE
      "${ROOT_PATH}/python/src/inference/tflite/*.h"
      "${ROOT_PATH}/python/src/inference/tflite/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_TFLITE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_NCNN)
    file(GLOB_RECURSE INFERENCE_NCNN_SOURCE
      "${ROOT_PATH}/python/src/inference/ncnn/*.h"
      "${ROOT_PATH}/python/src/inference/ncnn/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_NCNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE)
    file(GLOB_RECURSE INFERENCE_PADDLELITE_SOURCE
      "${ROOT_PATH}/python/src/inference/paddlelite/*.h"
      "${ROOT_PATH}/python/src/inference/paddlelite/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_PADDLELITE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_RKNN)
    file(GLOB_RECURSE INFERENCE_RKNN_SOURCE
      "${ROOT_PATH}/python/src/inference/rknn/*.h"
      "${ROOT_PATH}/python/src/inference/rknn/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_RKNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL)
    file(GLOB_RECURSE INFERENCE_ASCEND_CL_SOURCE
      "${ROOT_PATH}/python/src/inference/ascend_cl/*.h"
      "${ROOT_PATH}/python/src/inference/ascend_cl/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_ASCEND_CL_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_SNPE)
    file(GLOB_RECURSE INFERENCE_SNPE_SOURCE
      "${ROOT_PATH}/python/src/inference/snpe/*.h"
      "${ROOT_PATH}/python/src/inference/snpe/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_SNPE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TVM)
    file(GLOB_RECURSE INFERENCE_TVM_SOURCE
      "${ROOT_PATH}/python/src/inference/tvm/*.h"
      "${ROOT_PATH}/python/src/inference/tvm/*.cc"
    )
    set(PYTHON_INFERENCE_SOURCE ${PYTHON_INFERENCE_SOURCE} ${INFERENCE_TVM_SOURCE})
  endif()

  set(SOURCE ${SOURCE} ${PYTHON_INFERENCE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_DAG)
  file(GLOB_RECURSE PYTHON_DAG_SOURCE
    "${ROOT_PATH}/python/src/dag/*.h"
    "${ROOT_PATH}/python/src/dag/*.cc"
  )
  list(REMOVE_ITEM PYTHON_DAG_SOURCE 
    "${ROOT_PATH}/python/src/dag/node.cc"
    "${ROOT_PATH}/python/src/dag/composite_node.cc"
    "${ROOT_PATH}/python/src/dag/const_node.cc"
    "${ROOT_PATH}/python/src/dag/graph.cc"
    "${ROOT_PATH}/python/src/dag/loop.cc"
    "${ROOT_PATH}/python/src/dag/condition.cc"
    "${ROOT_PATH}/python/src/dag/running_condition.cc"
  )
  set(PYTHON_DAG_SOURCE ${PYTHON_DAG_SOURCE}
    # 依赖于node.cc的文件后列出
    "${ROOT_PATH}/python/src/dag/node.cc"
    "${ROOT_PATH}/python/src/dag/composite_node.cc"
    "${ROOT_PATH}/python/src/dag/const_node.cc"
    "${ROOT_PATH}/python/src/dag/graph.cc"
    "${ROOT_PATH}/python/src/dag/loop.cc"
    "${ROOT_PATH}/python/src/dag/condition.cc"
    "${ROOT_PATH}/python/src/dag/running_condition.cc"
  )
  set(SOURCE ${SOURCE} ${PYTHON_DAG_SOURCE})
endif()

# plugin
if(ENABLE_NNDEPLOY_PLUGIN)
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
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_INFER)
    file(GLOB_RECURSE PYTHON_INFER_SOURCE
      "${ROOT_PATH}/python/src/infer/*.h"
      "${ROOT_PATH}/python/src/infer/*.cc"
    )
    set(SOURCE ${SOURCE} ${PYTHON_INFER_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER)
    file(GLOB PYTHON_TOKENIZER_SOURCE
      "${ROOT_PATH}/python/src/tokenizer/*.h"
      "${ROOT_PATH}/python/src/tokenizer/*.cc"
    )
    if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
      file(GLOB_RECURSE PYTHON_TOKENIZER_CPP_SOURCE
        "${ROOT_PATH}/python/src/tokenizer/tokenizer_cpp/*.h"
        "${ROOT_PATH}/python/src/tokenizer/tokenizer_cpp/*.cc"
      )
      set(PYTHON_TOKENIZER_SOURCE ${PYTHON_TOKENIZER_SOURCE} ${PYTHON_TOKENIZER_CPP_SOURCE})
    endif()
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
  
  if(ENABLE_NNDEPLOY_PLUGIN_TRACK)
    file(GLOB_RECURSE PYTHON_TRACK_SOURCE
      "${ROOT_PATH}/python/src/track/*.h"
      "${ROOT_PATH}/python/src/track/*.cc"
    )
    set(SOURCE ${SOURCE} ${PYTHON_TRACK_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_MATTING)
    file(GLOB_RECURSE PYTHON_MATTING_SOURCE
      "${ROOT_PATH}/python/src/matting/*.h"
      "${ROOT_PATH}/python/src/matting/*.cc"
    )
    set(SOURCE ${SOURCE} ${PYTHON_MATTING_SOURCE})
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
endif()

# 创建 Python 模块
# 使用pybind11创建Python模块
# ${BINARY}是前面定义的pynndeploy
# ${SOURCE}是前面收集的所有源文件
# 该命令会生成一个Python扩展模块,将C++代码暴露给Python
pybind11_add_module(${BINARY} ${SOURCE})

# 属性
set_target_properties(${BINARY} PROPERTIES OUTPUT_NAME "_nndeploy_internal")
set_target_properties(${BINARY} PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# link
# target_link_libraries(${BINARY} PUBLIC nndeploy_framework)
# DEPEND_LIBRARY
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_FRAMEWORK_BINARY})
# foreach(framework ${NNDEPLOY_FRAMEWORK_BINARY})
#   if(WIN32)
#     add_custom_command(TARGET ${BINARY} POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${CMAKE_CURRENT_BINARY_DIR}/${framework}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       ${PROJECT_SOURCE_DIR}/python/nndeploy/${framework}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       COMMENT "Copying framework ${framework}${CMAKE_SHARED_LIBRARY_SUFFIX} to python/nndeploy/nndeploy/ directory"
#     )
#   elseif(APPLE)
#     add_custom_command(TARGET ${BINARY} POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${CMAKE_CURRENT_BINARY_DIR}/lib${framework}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       ${PROJECT_SOURCE_DIR}/python/nndeploy/lib${framework}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       COMMENT "Copying framework lib${framework}${CMAKE_SHARED_LIBRARY_SUFFIX} to python/nndeploy/nndeploy/ directory"
#     )
#   else()
#     add_custom_command(TARGET ${BINARY} POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${CMAKE_CURRENT_BINARY_DIR}/lib${framework}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       ${PROJECT_SOURCE_DIR}/python/nndeploy/lib${framework}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       COMMENT "Copying framework lib${framework}${CMAKE_SHARED_LIBRARY_SUFFIX} to python/nndeploy/nndeploy/ directory"
#     )
#   endif()
# endforeach()
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
message(STATUS "NNDEPLOY_PLUGIN_LIST: ${NNDEPLOY_PLUGIN_LIST}")
# # 添加构建后命令，将NNDEPLOY_PLUGIN_LIST中的so文件复制到python/nndeploy/目录下
# foreach(plugin ${NNDEPLOY_PLUGIN_LIST})
#   if(WIN32)
#     add_custom_command(TARGET ${BINARY} POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${CMAKE_CURRENT_BINARY_DIR}/${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       ${PROJECT_SOURCE_DIR}/python/nndeploy/${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       COMMENT "Copying plugin ${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX} to python/nndeploy/nndeploy/ directory"
#     )
#   elseif(APPLE)
#     add_custom_command(TARGET ${BINARY} POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${CMAKE_CURRENT_BINARY_DIR}/lib${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       ${PROJECT_SOURCE_DIR}/python/nndeploy/lib${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       COMMENT "Copying plugin lib${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX} to python/nndeploy/nndeploy/ directory"
#     )
#   else()
#     add_custom_command(TARGET ${BINARY} POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${CMAKE_CURRENT_BINARY_DIR}/lib${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       ${PROJECT_SOURCE_DIR}/python/nndeploy/lib${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX}
#       COMMENT "Copying plugin lib${plugin}${CMAKE_SHARED_LIBRARY_SUFFIX} to python/nndeploy/nndeploy/ directory"
#     )
#   endif()
# endforeach()
target_link_libraries(${BINARY} PUBLIC ${THIRD_PARTY_LIBRARY})

if (APPLE)
  set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
elseif (UNIX)
  set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
elseif(WIN32)
  if(MSVC)
    # target_link_options(${BINARY} PRIVATE /WHOLEARCHIVE)
  elseif(MINGW)
    set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")
  endif()
endif()

# copy python module to python/nndeploy/
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "")
# if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
#     if (SYSTEM.Windows)
#         # Windows系统下的处理:
#         # 1. 添加一个构建后命令,在目标${BINARY}构建完成后执行
#         # 2. 使用CMAKE的copy_if_different命令复制文件
#         # 3. 根据构建类型(Debug/Release)选择对应目录下的Python模块文件
#         # 4. 将文件复制到python/nndeploy/目录下
#         add_custom_command(TARGET ${BINARY} POST_BUILD
#             COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                 $<$<CONFIG:Debug>:${CMAKE_CURRENT_BINARY_DIR}/nndeploy/Debug/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}>
#                 $<$<CONFIG:Release>:${CMAKE_CURRENT_BINARY_DIR}/nndeploy/Release/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}>
#                 ${PROJECT_SOURCE_DIR}/python/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}
#             COMMENT "Copying Python module to output directory")
#         message(STATUS "Windows: Copying Python module to ${PROJECT_SOURCE_DIR}/python/nndeploy/")
#     else()
#         # Unix系统下的处理:
#         # 1. 添加一个构建后命令
#         # 2. 直接复制nndeploy目录下的Python模块文件到目标位置
#         add_custom_command(TARGET ${BINARY} POST_BUILD
#             COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                 ${CMAKE_CURRENT_BINARY_DIR}/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}
#                 ${PROJECT_SOURCE_DIR}/python/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION}
#             COMMENT "Copying Python module to output directory")
#         message(STATUS "Unix: Copying ${CMAKE_CURRENT_BINARY_DIR}/nndeploy/_nndeploy_internal${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION} to ${PROJECT_SOURCE_DIR}/python/nndeploy/")
#     endif()
# endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

# 生成python pip package安装脚本
# configure_file(${ROOT_PATH}/python/setup.py.cfg ${PROJECT_SOURCE_DIR}/python/setup.py)

# install
if(SYSTEM.Windows)
  install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
else()
  install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
endif()

# post install
# post install - 在安装后递归拷贝所有动态库
install(CODE "
  # 定义不同平台的动态库扩展名
  if(WIN32)
    set(LIB_EXTENSIONS \"*.dll\" \"*.dll.*\" \"*.lib\" \"*.lib.*\" \"*.pyd\" \"*.pyd.*\")
  elseif(APPLE)
    set(LIB_EXTENSIONS \"*.dylib\" \"*.dylib.*\" \"*.so\" \"*.so.*\")
  else()
    set(LIB_EXTENSIONS \"*.so\" \"*.so.*\")
  endif()
  set(SEARCH_PATHS \"${NNDEPLOY_INSTALL_PATH}\")
  
  # 确保目标目录存在
  file(MAKE_DIRECTORY \"${PROJECT_SOURCE_DIR}/python/nndeploy\")
  
  # 递归搜索并拷贝动态库
  foreach(search_path IN LISTS SEARCH_PATHS)
    if(EXISTS \"\${search_path}\")
      foreach(ext IN LISTS LIB_EXTENSIONS)
        file(GLOB_RECURSE DYNAMIC_LIBS \"\${search_path}/\${ext}\")
        foreach(lib_file IN LISTS DYNAMIC_LIBS)
          # 检查是否为软连接，如果是则跳过
          if(NOT IS_SYMLINK \"\${lib_file}\")
            get_filename_component(lib_name \"\${lib_file}\" NAME)
            message(STATUS \"Copying dynamic library: \${lib_name}\")
            file(COPY \"\${lib_file}\" 
                 DESTINATION \"${PROJECT_SOURCE_DIR}/python/nndeploy\")
          else()
            get_filename_component(lib_name \"\${lib_file}\" NAME)
            message(STATUS \"Skipping symlink: \${lib_name}\")
          endif()
        endforeach()
      endforeach()
    else()
      message(WARNING \"Search path does not exist: \${search_path}\")
    endif()
  endforeach()
  
  message(STATUS \"Finished copying dynamic libraries to python/nndeploy directory\")
")

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)
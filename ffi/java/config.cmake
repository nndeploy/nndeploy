message(STATUS "Building nndeploy java")

# set
set(SOURCE)
set(OBJECT)
set(BINARY javanndeploy)
set(DIRECTORY java)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

# 添加版本信息定义
set(PACKAGE_VERSION ${NNDEPLOY_VERSION})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

# include
include_directories(${ROOT_PATH}/ffi/java/jni)

# nndeploy source
file(GLOB JAVA_SOURCE
  "${ROOT_PATH}/ffi/java/jni/*.h"
  "${ROOT_PATH}/ffi/java/jni/*.cc"
)
set(SOURCE ${SOURCE} ${JAVA_SOURCE})

# framework
if(ENABLE_NNDEPLOY_BASE)
  file(GLOB_RECURSE JAVA_BASE_SOURCE
    "${ROOT_PATH}/ffi/java/jni/base/*.h"
    "${ROOT_PATH}/ffi/java/jni/base/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_BASE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_THREAD_POOL)
  file(GLOB_RECURSE JAVA_THREAD_POOL_SOURCE
    "${ROOT_PATH}/ffi/java/jni/thread_pool/*.h"
    "${ROOT_PATH}/ffi/java/jni/thread_pool/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_THREAD_POOL_SOURCE})
endif()

if(ENABLE_NNDEPLOY_CRYPTION)
  file(GLOB_RECURSE JAVA_CRYPTION_SOURCE
    "${ROOT_PATH}/ffi/java/jni/cryption/*.h"
    "${ROOT_PATH}/ffi/java/jni/cryption/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_CRYPTION_SOURCE})
endif()

if(ENABLE_NNDEPLOY_DEVICE)
  file(GLOB_RECURSE JAVA_DEVICE_SOURCE
    "${ROOT_PATH}/ffi/java/jni/device/*.h"
    "${ROOT_PATH}/ffi/java/jni/device/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_DEVICE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_IR)
  file(GLOB_RECURSE JAVA_IR_SOURCE
    "${ROOT_PATH}/ffi/java/jni/ir/*.h"
    "${ROOT_PATH}/ffi/java/jni/ir/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_IR_SOURCE})
endif()

if(ENABLE_NNDEPLOY_OP)
  file(GLOB_RECURSE JAVA_OP_SOURCE
    "${ROOT_PATH}/ffi/java/jni/op/*.h"
    "${ROOT_PATH}/ffi/java/jni/op/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_OP_SOURCE})
endif()

if(ENABLE_NNDEPLOY_NET)
  file(GLOB_RECURSE JAVA_NET_SOURCE
    "${ROOT_PATH}/ffi/java/jni/net/*.h"
    "${ROOT_PATH}/ffi/java/jni/net/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_NET_SOURCE})
endif()

if(ENABLE_NNDEPLOY_INFERENCE)
  file(GLOB JAVA_INFERENCE_SOURCE
    "${ROOT_PATH}/ffi/java/jni/inference/*.h"
    "${ROOT_PATH}/ffi/java/jni/inference/*.cc"
  )
  
  if(ENABLE_NNDEPLOY_INFERENCE_DEFAULT)
    file(GLOB_RECURSE INFERENCE_DEFAULT_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/default/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/default/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_DEFAULT_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TENSORRT)
    file(GLOB_RECURSE INFERENCE_TENSORRT_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/tensorrt/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/tensorrt/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_TENSORRT_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TNN)
    file(GLOB_RECURSE INFERENCE_TNN_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/tnn/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/tnn/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_TNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_MNN)
    file(GLOB_RECURSE INFERENCE_MNN_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/mnn/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/mnn/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_MNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_OPENVINO)
    file(GLOB_RECURSE INFERENCE_OPENVINO_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/openvino/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/openvino/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_OPENVINO_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_COREML)
    file(GLOB_RECURSE INFERENCE_COREML_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/coreml/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/coreml/*.cc"
      "${ROOT_PATH}/ffi/java/jni/inference/coreml/*.mm"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_COREML_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME)
    file(GLOB_RECURSE INFERENCE_ONNXRUNTIME_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/onnxruntime/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/onnxruntime/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_ONNXRUNTIME_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TFLITE)
    file(GLOB_RECURSE INFERENCE_TFLITE_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/tflite/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/tflite/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_TFLITE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_NCNN)
    file(GLOB_RECURSE INFERENCE_NCNN_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/ncnn/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/ncnn/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_NCNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_PADDLELITE)
    file(GLOB_RECURSE INFERENCE_PADDLELITE_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/paddlelite/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/paddlelite/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_PADDLELITE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_RKNN)
    file(GLOB_RECURSE INFERENCE_RKNN_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/rknn/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/rknn/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_RKNN_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_ASCEND_CL)
    file(GLOB_RECURSE INFERENCE_ASCEND_CL_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/ascend_cl/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/ascend_cl/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_ASCEND_CL_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_SNPE)
    file(GLOB_RECURSE INFERENCE_SNPE_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/snpe/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/snpe/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_SNPE_SOURCE})
  endif()

  if(ENABLE_NNDEPLOY_INFERENCE_TVM)
    file(GLOB_RECURSE INFERENCE_TVM_SOURCE
      "${ROOT_PATH}/ffi/java/jni/inference/tvm/*.h"
      "${ROOT_PATH}/ffi/java/jni/inference/tvm/*.cc"
    )
    set(JAVA_INFERENCE_SOURCE ${JAVA_INFERENCE_SOURCE} ${INFERENCE_TVM_SOURCE})
  endif()

  set(SOURCE ${SOURCE} ${JAVA_INFERENCE_SOURCE})
endif()

if(ENABLE_NNDEPLOY_DAG)
  file(GLOB_RECURSE JAVA_DAG_SOURCE
    "${ROOT_PATH}/ffi/java/jni/dag/*.h"
    "${ROOT_PATH}/ffi/java/jni/dag/*.cc"
  )
  set(SOURCE ${SOURCE} ${JAVA_DAG_SOURCE})
endif()

# plugin
if(ENABLE_NNDEPLOY_PLUGIN)
  if(ENABLE_NNDEPLOY_PLUGIN_PREPROCESS)
    file(GLOB_RECURSE JAVA_PREPROCESS_SOURCE
      "${ROOT_PATH}/ffi/java/jni/preprocess/*.h"
      "${ROOT_PATH}/ffi/java/jni/preprocess/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_PREPROCESS_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_CODEC)
    file(GLOB_RECURSE JAVA_CODEC_SOURCE
      "${ROOT_PATH}/ffi/java/jni/codec/*.h"
      "${ROOT_PATH}/ffi/java/jni/codec/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_CODEC_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_INFER)
    file(GLOB_RECURSE JAVA_INFER_SOURCE
      "${ROOT_PATH}/ffi/java/jni/infer/*.h"
      "${ROOT_PATH}/ffi/java/jni/infer/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_INFER_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER)
    file(GLOB JAVA_TOKENIZER_SOURCE
      "${ROOT_PATH}/ffi/java/jni/tokenizer/*.h"
      "${ROOT_PATH}/ffi/java/jni/tokenizer/*.cc"
    )
    if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
      file(GLOB_RECURSE JAVA_TOKENIZER_CPP_SOURCE
        "${ROOT_PATH}/ffi/java/jni/tokenizer/tokenizer_cpp/*.h"
        "${ROOT_PATH}/ffi/java/jni/tokenizer/tokenizer_cpp/*.cc"
      )
      set(JAVA_TOKENIZER_SOURCE ${JAVA_TOKENIZER_SOURCE} ${JAVA_TOKENIZER_CPP_SOURCE})
    endif()
    set(SOURCE ${SOURCE} ${JAVA_TOKENIZER_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_DETECT)
    file(GLOB_RECURSE JAVA_DETECT_SOURCE
      "${ROOT_PATH}/ffi/java/jni/detect/*.h"
      "${ROOT_PATH}/ffi/java/jni/detect/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_DETECT_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_SEGMENT)
    file(GLOB_RECURSE JAVA_SEGMENT_SOURCE
      "${ROOT_PATH}/ffi/java/jni/segment/*.h"
      "${ROOT_PATH}/ffi/java/jni/segment/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_SEGMENT_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_CLASSIFICATION)
    file(GLOB_RECURSE JAVA_CLASSIFICATION_SOURCE
      "${ROOT_PATH}/ffi/java/jni/classification/*.h"
      "${ROOT_PATH}/ffi/java/jni/classification/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_CLASSIFICATION_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_TRACK)
    file(GLOB_RECURSE JAVA_TRACK_SOURCE
      "${ROOT_PATH}/ffi/java/jni/track/*.h"
      "${ROOT_PATH}/ffi/java/jni/track/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_TRACK_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_MATTING)
    file(GLOB_RECURSE JAVA_MATTING_SOURCE
      "${ROOT_PATH}/ffi/java/jni/matting/*.h"
      "${ROOT_PATH}/ffi/java/jni/matting/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_MATTING_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_LLM)
    file(GLOB_RECURSE JAVA_LLM_SOURCE
      "${ROOT_PATH}/ffi/java/jni/llm/*.h"
      "${ROOT_PATH}/ffi/java/jni/llm/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_LLM_SOURCE})
  endif()
  
  if(ENABLE_NNDEPLOY_PLUGIN_STABLE_DIFFUSION)
    file(GLOB_RECURSE JAVA_STABLE_DIFFUSION_SOURCE
      "${ROOT_PATH}/ffi/java/jni/stable_diffusion/*.h"
      "${ROOT_PATH}/ffi/java/jni/stable_diffusion/*.cc"
    )
    set(SOURCE ${SOURCE} ${JAVA_STABLE_DIFFUSION_SOURCE})
  endif()
endif()

# 创建 Java JNI 动态库
add_library(${BINARY} SHARED ${SOURCE})

# 设置 JNI 相关的编译选项
target_include_directories(${BINARY} PRIVATE ${JNI_INCLUDE_DIRS})

list(APPEND DEPEND_LIBRARY ${NNDEPLOY_DEPEND_LIBRARY})
list(APPEND DEPEND_LIBRARY ${NNDEPLOY_JAVA_DEPEND_LIBRARY})
target_link_libraries(${BINARY} PUBLIC  ${DEPEND_LIBRARY})

# SYSTEM_LIBRARY
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_SYSTEM_LIBRARY})
list(APPEND SYSTEM_LIBRARY ${NNDEPLOY_JAVA_SYSTEM_LIBRARY})
target_link_libraries(${BINARY} PUBLIC  ${SYSTEM_LIBRARY})

# THIRD_PARTY_LIBRARY
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_JAVA_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})
list(APPEND THIRD_PARTY_LIBRARY ${NNDEPLOY_PLUGIN_LIST})
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

# install
if(SYSTEM.Windows)
  install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
else()
  install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
endif()

# java
# Java 包构建配置
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)

# 设置 Java 项目名称和版本
set(PROJECT_NAME javanndeploy_package)
set(JAVA_PACKAGE_VERSION ${NNDEPLOY_VERSION})

# 收集所有 Java 源文件
file(GLOB_RECURSE PURE_JAVA_SOURCE
  "${ROOT_PATH}/ffi/java/nndeploy/*.java"
)

# 检查是否找到 Java 源文件
if(NOT PURE_JAVA_SOURCE)
  message(WARNING "No Java source files found in ${ROOT_PATH}/ffi/java/nndeploy/")
endif()

# 创建 Java JAR 包
add_jar(${PROJECT_NAME} 
  SOURCES ${PURE_JAVA_SOURCE}
  OUTPUT_NAME nndeploy
  OUTPUT_DIR ${NNDEPLOY_INSTALL_PATH}
  VERSION ${JAVA_PACKAGE_VERSION}
  # MANIFEST ${ROOT_PATH}/ffi/java/MANIFEST.MF
)

# 设置 JAR 包属性
set_target_properties(${PROJECT_NAME} PROPERTIES
  JAR_FILE "${NNDEPLOY_INSTALL_PATH}/nndeploy-${JAVA_PACKAGE_VERSION}.jar"
)

# 安装 JAR 包
install_jar(${PROJECT_NAME} DESTINATION ${NNDEPLOY_INSTALL_PATH})

# 生成 Java 文档（可选）
if(BUILD_JAVA_DOCS)
  find_package(Java COMPONENTS Development)
  if(Java_JAVADOC_EXECUTABLE)
    add_custom_target(java_docs
      COMMAND ${Java_JAVADOC_EXECUTABLE} 
        -d ${CMAKE_BINARY_DIR}/java_docs
        -sourcepath ${ROOT_PATH}/ffi/java
        -subpackages nndeploy
      COMMENT "Generating Java documentation"
    )
  endif()
endif()

# 添加依赖关系，确保 JNI 库先构建完成
add_dependencies(${PROJECT_NAME} ${BINARY})

message(STATUS "Java package configuration completed: ${PROJECT_NAME}")

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)
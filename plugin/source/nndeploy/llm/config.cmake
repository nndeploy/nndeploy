message(STATUS "plugin/llm")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_llm)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/*.h"
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/*.hpp"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/*.cc"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/*.cpp"
)

file(GLOB_RECURSE PLUGIN_SOURCE_EMBEDDING
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/embedding/*.h"
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/embedding/*.hpp"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/embedding/*.cc"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/embedding/*.cpp"
)
set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${PLUGIN_SOURCE_EMBEDDING})

file(GLOB_RECURSE PLUGIN_SOURCE_PROMPT
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/prompt/*.h"
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/prompt/*.hpp"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/prompt/*.cc"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/prompt/*.cpp"
)
set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${PLUGIN_SOURCE_PROMPT})

file(GLOB_RECURSE PLUGIN_SOURCE_SAMPLE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/sample/*.h"
  "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/sample/*.hpp"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/sample/*.cc"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/sample/*.cpp"
)
set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${PLUGIN_SOURCE_SAMPLE})

if (ENABLE_NNDEPLOY_INFERENCE_MNN STREQUAL "OFF")
else()
  file(GLOB_RECURSE PLUGIN_SOURCE_MNN
    "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/mnn/*.h"
    "${PLUGIN_ROOT_PATH}/include/nndeploy/llm/mnn/*.hpp"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/mnn/*.cc"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/llm/mnn/*.cpp"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${PLUGIN_SOURCE_MNN})
endif()

# message(STATUS "PLUGIN_SOURCE: ${PLUGIN_SOURCE}")

# # TARGET
add_library(${PLUGIN_BINARY} ${NNDEPLOY_LIB_TYPE} ${PLUGIN_SOURCE} ${PLUGIN_OBJECT})

# # DIRECTORY
set_property(TARGET ${PLUGIN_BINARY} PROPERTY FOLDER ${NNDEPLOY_PLUGIN_DIRECTORY})

# # DEPEND_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_DEPEND_LIBRARY})

# # SYSTEM_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY})

# # THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY})

# # NNDEPLOY_FRAMEWORK_BINARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_FRAMEWORK_BINARY})
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_preprocess)
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_infer)
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_tokenizer)

# # NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})

# # install
if(SYSTEM_Windows)
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/llm DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/llm DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)

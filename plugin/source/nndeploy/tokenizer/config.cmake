message(STATUS "plugin/tokenizer")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_tokenizer)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/tokenizer/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/tokenizer/*.cc"
)

if (ENABLE_NNDEPLOY_INFERENCE_MNN STREQUAL "OFF")
else()
  file(GLOB_RECURSE TOKENIZER_MNN_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/tokenizer/tokenizer_mnn/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/tokenizer/tokenizer_mnn/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${TOKENIZER_MNN_SOURCE})
endif()

# file(GLOB TOKENIZER_MNN_SOURCE
#     "${PLUGIN_ROOT_PATH}/include/nndeploy/tokenizer/tokenizer_mnn/*.h"
#     "${PLUGIN_ROOT_PATH}/source/nndeploy/tokenizer/tokenizer_mnn/*.cc"
#   )
# set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${TOKENIZER_MNN_SOURCE})

if(ENABLE_NNDEPLOY_PLUGIN_TOKENIZER_CPP)
  file(GLOB_RECURSE TOKENIZER_CPP_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/tokenizer/tokenizer_cpp/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/tokenizer/tokenizer_cpp/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${TOKENIZER_CPP_SOURCE})
endif()

## TARGET
add_library(${PLUGIN_BINARY} ${NNDEPLOY_LIB_TYPE} ${PLUGIN_SOURCE} ${PLUGIN_OBJECT})
## DIRECTORY
set_property(TARGET ${PLUGIN_BINARY} PROPERTY FOLDER ${NNDEPLOY_PLUGIN_DIRECTORY})
## DEPEND_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_DEPEND_LIBRARY})
## SYSTEM_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY}) 
## THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY}) 
## NNDEPLOY_FRAMEWORK_BINARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_FRAMEWORK_BINARY}) 
## NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY}) 
## install
if(SYSTEM.Windows)
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/tokenizer DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else() 
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/tokenizer DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)



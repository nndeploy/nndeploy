message(STATUS "plugin/codec")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_codec)

# Always include base codec source
list(APPEND PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/codec/codec.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/codec/codec.cc"
)

# Always include OpenCV codec source
if(ENABLE_NNDEPLOY_OPENCV)
  file(GLOB_RECURSE OPENCV_CODEC_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/codec/opencv/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/codec/opencv/*.cc"
  )
  list(APPEND PLUGIN_SOURCE ${OPENCV_CODEC_SOURCE})
  message(STATUS "  + OpenCV codec backend")
else()
  message(STATUS "  - OpenCV codec backend (disabled)")
endif()


## TARGET
add_library(${PLUGIN_BINARY} ${NNDEPLOY_LIB_TYPE} ${PLUGIN_SOURCE} ${PLUGIN_OBJECT})
## DIRECTORY
set_property(TARGET ${PLUGIN_BINARY} PROPERTY FOLDER ${NNDEPLOY_PLUGIN_DIRECTORY})
## DEPEND_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_DEPEND_LIBRARY})
## SYSTEM_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_SYSTEM_LIBRARY})
## THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_THIRD_PARTY_LIBRARY})
## NNDEPLOY_FRAMEWORK_BINARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_FRAMEWORK_BINARY})
## NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} PRIVATE ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})
## install
if(SYSTEM_Windows)
  nndeploy_install_target(${PLUGIN_BINARY})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/codec DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/codec DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)

